use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone)]
pub enum KernelType {
    Poly { degree: u32, coef0: f64, gamma: f64 },
    RBF { gamma: f64 },
    Linear,
}

impl KernelType {
    fn compute(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        match self {
            KernelType::Poly { degree, coef0, gamma } =>
                (gamma * x.dot(y) + *coef0).powi(*degree as i32),
            KernelType::RBF { gamma } => {
                let diff = x - y;
                (-gamma * diff.dot(&diff)).exp()
            }
            KernelType::Linear => x.dot(y),
        }
    }
}

#[derive(Clone)]
pub struct DualSVM {
    alphas: Option<Array1<f64>>,
    support_vectors: Option<Array2<f64>>,
    support_labels: Option<Array1<f64>>,
    bias: f64,
    c: f64,
    kernel: KernelType,
}

impl DualSVM {
    pub fn new(kernel: KernelType, c: f64) -> Self {
        Self {
            alphas: None,
            support_vectors: None,
            support_labels: None,
            bias: 0.0,
            c,
            kernel,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, max_iter: usize, tol: f64) {
        let n = x.nrows();
        let mut alphas = Array1::<f64>::zeros(n);
        let mut bias = 0.0;
        let c = self.c;

        let mut kernel_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                kernel_mat[[i, j]] = self.kernel.compute(&x.row(i).to_owned(), &x.row(j).to_owned());
            }
        }

        let mut rng = thread_rng();
        let mut passes = 0;
        let max_passes = 5;
        let eps = tol;
        let mut iter = 0;
        while passes < max_passes && iter < max_iter {
            let mut alpha_changed = 0;
            let idx: Vec<usize> = (0..n).collect();
            for &i in idx.choose_multiple(&mut rng, n) {
                let fxi = (0..n)
                    .map(|k| alphas[k] * y[k] * kernel_mat[[k, i]])
                    .sum::<f64>()
                    + bias;
                let ei = fxi - y[i];

                if !((y[i] * ei < -eps && alphas[i] < c - eps) ||
                     (y[i] * ei > eps && alphas[i] > eps)) {
                    continue;
                }

                let mut j = i;
                while j == i {
                    j = rng.gen_range(0..n);
                }

                let fxj = (0..n)
                    .map(|k| alphas[k] * y[k] * kernel_mat[[k, j]])
                    .sum::<f64>()
                    + bias;
                let ej = fxj - y[j];

                let (yi, yj) = (y[i], y[j]);
                let (ai_old, aj_old) = (alphas[i], alphas[j]);

                let (l, h) = if yi != yj {
                    (f64::max(0.0, aj_old - ai_old), f64::min(c, c + aj_old - ai_old))
                } else {
                    (f64::max(0.0, ai_old + aj_old - c), f64::min(c, ai_old + aj_old))
                };
                if (l - h).abs() < 1e-12 { continue; }

                let eta = 2.0 * kernel_mat[[i, j]] - kernel_mat[[i, i]] - kernel_mat[[j, j]];
                if eta >= 0.0 { continue; }

                let mut aj_new = aj_old - yj * (ei - ej) / eta;
                aj_new = aj_new.clamp(l, h);
                if (aj_new - aj_old).abs() < 1e-6 { continue; }

                let ai_new = ai_old + yi * yj * (aj_old - aj_new);

                let b1 = bias - ei
                    - yi * (ai_new - ai_old) * kernel_mat[[i, i]]
                    - yj * (aj_new - aj_old) * kernel_mat[[i, j]];
                let b2 = bias - ej
                    - yi * (ai_new - ai_old) * kernel_mat[[i, j]]
                    - yj * (aj_new - aj_old) * kernel_mat[[j, j]];
                bias = if ai_new > eps && ai_new < c - eps {
                    b1
                } else if aj_new > eps && aj_new < c - eps {
                    b2
                } else {
                    (b1 + b2) / 2.0
                };

                alphas[i] = ai_new;
                alphas[j] = aj_new;
                alpha_changed += 1;
            }
            if alpha_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
            iter += 1;
        }

        let sv_idx: Vec<usize> = alphas
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > 1e-8)
            .map(|(i, _)| i)
            .collect();

        let support_vectors = Array2::from_shape_fn((sv_idx.len(), x.ncols()), |(i, j)| x[[sv_idx[i], j]]);
        let support_labels = Array1::from_iter(sv_idx.iter().map(|&i| y[i]));
        let support_alphas = Array1::from_iter(sv_idx.iter().map(|&i| alphas[i]));

        self.support_vectors = Some(support_vectors);
        self.support_labels = Some(support_labels);
        self.alphas = Some(support_alphas);
        self.bias = bias;
    }

    pub fn decision_function(&self, x: &Array1<f64>) -> f64 {
        let sv = self.support_vectors.as_ref().unwrap();
        let sl = self.support_labels.as_ref().unwrap();
        let al = self.alphas.as_ref().unwrap();
        let mut sum = 0.0;
        for i in 0..sv.nrows() {
            sum += al[i] * sl[i] * self.kernel.compute(&sv.row(i).to_owned(), x);
        }
        sum + self.bias
    }
}

#[pyclass]
pub struct SVM {
    classifiers: Vec<(f64, f64, DualSVM)>,
    classes: Vec<f64>,
    c: f64,
    #[pyo3(get)]
    kernel_type: String,
    degree: u32,
    coef0: f64,
    gamma: f64,
}

#[pymethods]
impl SVM {
    #[staticmethod]
    pub fn poly(degree: u32, coef0: f64, c: f64, gamma: Option<f64>) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
            c,
            kernel_type: "poly".to_string(),
            degree,
            coef0,
            gamma: gamma.unwrap_or(1.0),
        }
    }


    #[staticmethod]
    pub fn rbf(gamma: f64, c: f64) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
            c,
            kernel_type: "rbf".to_string(),
            degree: 0,
            coef0: 0.0,
            gamma,
        }
    }

    #[staticmethod]
    pub fn linear(c: f64) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
            c,
            kernel_type: "linear".to_string(),
            degree: 1,
            coef0: 0.0,
            gamma: 0.0,
        }
    }

pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<()> {
        let n_samples = x.len();
        if n_samples == 0 || y.len() != n_samples {
            return Err(PyValueError::new_err("Empty data or label size mismatch"));
        }
        let n_features = x[0].len();
        let x_flat: Vec<f64> = x.iter().flatten().copied().collect();
        let x_nd = Array2::from_shape_vec((n_samples, n_features), x_flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut classes: Vec<f64> = y.clone();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup_by(|a, b| a == b);
        self.classes = classes.clone();
        self.classifiers.clear();

        let kernel = match self.kernel_type.as_str() {
            "poly" => KernelType::Poly { degree: self.degree, coef0: self.coef0, gamma: self.gamma },
            "rbf" => KernelType::RBF { gamma: self.gamma },
            "linear" => KernelType::Linear,
            _ => KernelType::Linear,
        };

        let pairs: Vec<(f64, f64)> = classes.iter().enumerate()
            .flat_map(|(i, &a)| classes.iter().skip(i + 1).map(move |&b| (a, b)))
            .collect();

        let classifiers: Vec<(f64, f64, DualSVM)> = pairs
            .par_iter()
            .map(|&(class_a, class_b)| {
                let mask: Vec<usize> = y.iter().enumerate()
                    .filter(|(_, &label)| label == class_a || label == class_b)
                    .map(|(idx, _)| idx)
                    .collect();
                let x_bin = Array2::from_shape_fn((mask.len(), n_features), |(i, j)| x_nd[[mask[i], j]]);
                let y_bin = Array1::from_iter(mask.iter().map(|&i| if y[i] == class_a { 1.0 } else { -1.0 }));
                let mut svm = DualSVM::new(kernel.clone(), self.c);
                svm.fit(&x_bin, &y_bin, max_iter, tol);
                (class_a, class_b, svm)
            })
            .collect();

        self.classifiers = classifiers;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let n_samples = x.len();
        if n_samples == 0 { return Ok(vec![]); }
        let n_features = if let Some(sv) = self.classifiers.first() {
            sv.2.support_vectors.as_ref().unwrap().ncols()
        } else {
            return Err(PyValueError::new_err("No classifiers trained"));
        };
        let x_flat: Vec<f64> = x.iter().flatten().copied().collect();
        let x_nd = Array2::from_shape_vec((n_samples, n_features), x_flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let preds: Vec<f64> = x_nd.outer_iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|xi| {
                let mut votes = vec![0; self.classes.len()];
                for (class_a, class_b, svm) in &self.classifiers {
                    let pred = svm.decision_function(&xi.to_owned());
                    let idx = if pred >= 0.0 {
                        self.classes.iter().position(|c| c == class_a).unwrap()
                    } else {
                        self.classes.iter().position(|c| c == class_b).unwrap()
                    };
                    votes[idx] += 1;
                }
                let (idx, _) = votes.iter().enumerate().max_by_key(|&(_, cnt)| cnt).unwrap();
                self.classes[idx]
            })
            .collect();
        Ok(preds)
    }
}