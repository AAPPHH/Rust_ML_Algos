use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

use crate::svm_kernel::KernelType;
use crate::dual_svm::DualSVM;
use crate::flat_dataset::FlatDataset;
use faer::{Mat, prelude::*};
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

        let dataset = FlatDataset::from_nested(x);
        let c_val = self.c;

        let classifiers: Vec<(f64, f64, DualSVM)> = Python::with_gil(|py| {
            py.allow_threads(|| {
                let y_arr = y;

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

                        // faer Matrix für Subset bauen
                        let mut x_bin_mat = Mat::<f64>::zeros(idx.len(), n_features);
                        for (row_idx, &i) in idx.iter().enumerate() {
                            for j in 0..n_features {
                                let val = dataset.data.read(i, j);
                                x_bin_mat.write(row_idx, j, val);
                            }
                        }

                        let x_bin = FlatDataset { data: x_bin_mat };

                        let y_bin: Vec<f64> = idx.iter()
                            .map(|&i| if y_arr[i] == class_a { 1.0 } else { -1.0 })
                            .collect();

                        let mut svm = DualSVM::new(kernel_def.clone(), c_val);
                        svm.fit(x_bin, y_bin, max_iter, tol);

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
        let dataset = FlatDataset::from_nested(x);
        let n_classes = self.classes.len();
        let mut votes = vec![vec![0usize; n_classes]; n_samples];

        for (class_a, class_b, svm) in &self.classifiers {
            let preds = svm.decision_function_batch(&dataset);
            let idx_a = self.classes.iter().position(|c| c == class_a).unwrap();
            let idx_b = self.classes.iter().position(|c| c == class_b).unwrap();
            for (i, &score) in preds.iter().enumerate() {
                if score >= 0.0 {
                    votes[i][idx_a] += 1;
                } else {
                    votes[i][idx_b] += 1;
                }
            }
        }

        let preds = votes
            .iter()
            .map(|row| {
                let (idx, _) = row.iter().enumerate().max_by_key(|&(_, cnt)| cnt).unwrap();
                self.classes[idx]
            })
            .collect();

        Ok(preds)
    }
}
