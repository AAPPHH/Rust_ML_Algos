use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use crate::svm_kernel::KernelType;
use crate::dual_svm::DualSVM;

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

        let mut classes: Vec<f64> = y.clone();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        self.classes = classes.clone();

        let kernel_def = match self.kernel_type.as_str() {
            "poly" => KernelType::Poly {
                degree: self.degree,
                coef0: self.coef0,
                gamma: self.gamma,
            },
            "rbf" => KernelType::RBF { gamma: self.gamma },
            "linear" => KernelType::Linear,
            _ => KernelType::Linear,
        };
        let c_val = self.c;

        let x_flat: Vec<f64> = x.iter().flatten().copied().collect();
        let y_vec = y;

        let classifiers: Vec<(f64, f64, DualSVM)> = Python::with_gil(|py| {
            py.allow_threads(|| {
                let x_nd = Array2::from_shape_vec((n_samples, n_features), x_flat)
                    .expect("shape checked above");
                let y_arr = Array1::from_vec(y_vec);

                let pairs: Vec<(f64, f64)> = classes
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &a)| classes.iter().skip(i + 1).map(move |&b| (a, b)))
                    .collect();

                pairs
                    .par_iter()
                    .map(|&(class_a, class_b)| {
                        let idx: Vec<usize> = y_arr
                            .iter()
                            .enumerate()
                            .filter(|(_, &lab)| lab == class_a || lab == class_b)
                            .map(|(i, _)| i)
                            .collect();

                        let x_bin = Array2::from_shape_fn((idx.len(), n_features), |(r, c)| {
                            x_nd[[idx[r], c]]
                        });
                        let y_bin = Array1::from_iter(
                            idx.iter().map(|&i| if y_arr[i] == class_a { 1.0 } else { -1.0 }),
                        );

                        let kernel_mat = kernel_def.compute_kernel(&x_bin, &x_bin);

                        let mut svm = DualSVM::new(kernel_def.clone(), c_val);
                        svm.fit(&kernel_mat, &y_bin, max_iter, tol);

                        (class_a, class_b, svm)
                    })
                    .collect()
            })
        });
        self.classifiers = classifiers;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let n_samples = x.len();
        if n_samples == 0 {
            return Ok(vec![]);
        }
        let n_features = if let Some(sv) = self.classifiers.first() {
            sv.2.support_vectors.as_ref().map(|s| s.ncols()).unwrap_or(0)
        } else {
            return Err(PyValueError::new_err("No classifiers trained"));
        };
        let x_flat: Vec<f64> = x.iter().flatten().copied().collect();
        let x_nd = Array2::from_shape_vec((n_samples, n_features), x_flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let n_classes = self.classes.len();
        let mut votes = Array2::<usize>::zeros((n_samples, n_classes));

        for (class_a, class_b, svm) in &self.classifiers {
            let preds = svm.decision_function_batch(&x_nd);
            let idx_a = self.classes.iter().position(|c| c == class_a).unwrap();
            let idx_b = self.classes.iter().position(|c| c == class_b).unwrap();
            for (i, &score) in preds.iter().enumerate() {
                if score >= 0.0 {
                    votes[[i, idx_a]] += 1;
                } else {
                    votes[[i, idx_b]] += 1;
                }
            }
        }
        let preds = votes
            .outer_iter()
            .map(|row| {
                let (idx, _) = row.iter().enumerate().max_by_key(|&(_, cnt)| cnt).unwrap();
                self.classes[idx]
            })
            .collect();
        Ok(preds)
    }
}

