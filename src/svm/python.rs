use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use crate::svm::multiclass::SVM;
use crate::svm::dataset::{FlatDataset, DatasetStorage, SvmDataset}; // KORREKTUR: SvmDataset importiert

#[pyclass]
pub struct PySVM {
    inner: SVM,
}

#[pymethods]
impl PySVM {
    // ... (unveränderter Code bis n_support)
    #[staticmethod]
    pub fn poly(degree: u32, coef0: f64, c: f64, gamma: Option<f64>) -> Self {
        PySVM { inner: SVM::poly(degree, coef0, c, gamma) }
    }

    #[staticmethod]
    pub fn rbf(gamma: f64, c: f64) -> Self {
        PySVM { inner: SVM::rbf(gamma, c) }
    }

    #[staticmethod]
    pub fn linear(c: f64) -> Self {
        PySVM { inner: SVM::linear(c) }
    }

    pub fn fit(
        &mut self, 
        x: PyReadonlyArray2<f64>, 
        y: PyReadonlyArray1<f64>, 
        max_iter: usize, 
        tol: f64
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        
        let n_samples = x_array.shape()[0];
        
        if n_samples != y_array.len() {
            return Err(PyValueError::new_err("X and y must have the same number of samples"));
        }
        
        let dataset = FlatDataset::from_numpy_array(x_array);
        let y_vec: Vec<f64> = y_array.iter().copied().collect();
        
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.inner.fit_dataset(dataset, y_vec, max_iter, tol)
            })
        }).map_err(|e| PyValueError::new_err(e))
    }

    pub fn predict(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let x_array = x.as_array();
        let dataset = FlatDataset::from_numpy_array(x_array);
        
        let predictions = Python::with_gil(|py| {
            py.allow_threads(|| {
                self.inner.predict_dataset(&dataset)
            })
        });

        Python::with_gil(|py| {
            Ok(PyArray1::from_vec(py, predictions).to_owned())
        })
    }

    pub fn predict_proba(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x_array = x.as_array();
        let n_samples = x_array.shape()[0];
        let dataset = FlatDataset::from_numpy_array(x_array);

        let proba = Python::with_gil(|py| {
            py.allow_threads(|| {
                self.inner.predict_proba_dataset(&dataset)
            })
        });

        Python::with_gil(|py| {
            let n_classes = if proba.is_empty() { 0 } else { proba[0].len() };
            let flat: Vec<f64> = proba.into_iter().flatten().collect();
            let array = PyArray1::from_vec(py, flat);
            Ok(array.reshape((n_samples, n_classes))?.to_owned())
        })
    }
    
    pub fn decision_function(&self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<Py<PyArray1<f64>>>> {
        let x_array = x.as_array();
        let dataset = FlatDataset::from_numpy_array(x_array);
        let n_classifiers = self.inner.classifiers.len();
        let mut all_decisions = vec![vec![]; n_classifiers];
        
        Python::with_gil(|py| {
            py.allow_threads(|| {
                for (idx, (_, _, svm)) in self.inner.classifiers.iter().enumerate() {
                    all_decisions[idx] = svm.decision_function_batch(&dataset);
                }
            });
            
            let result: Vec<Py<PyArray1<f64>>> = all_decisions
                .into_iter()
                .map(|decisions| PyArray1::from_vec(py, decisions).to_owned())
                .collect();
            
            Ok(result)
        })
    }

    #[getter]
    pub fn kernel_type(&self) -> String {
        self.inner.kernel_type.clone()
    }
    
    #[getter]
    pub fn classes(&self) -> Vec<f64> {
        self.inner.classes.clone()
    }
    
    #[getter]
    pub fn n_support(&self) -> Vec<usize> {
        let mut counts = vec![0; self.inner.classes.len()];
        
        for (class_a, class_b, svm) in &self.inner.classifiers {
            if let Some(ref sv) = svm.support_vectors {
                // KORREKTUR: n_samples() ist jetzt eine Trait-Methode und SvmDataset muss im Scope sein.
                let n_sv = sv.n_samples();
                let idx_a = self.inner.classes.iter().position(|&c| c == *class_a).unwrap();
                let idx_b = self.inner.classes.iter().position(|&c| c == *class_b).unwrap();
                counts[idx_a] += n_sv / 2;
                counts[idx_b] += (n_sv + 1) / 2;
            }
        }
        
        counts
    }
    
    // Rest der Datei ist unverändert...
    #[getter]
    pub fn support_vectors(&self) -> PyResult<Vec<Py<PyArray2<f64>>>> {
        Python::with_gil(|py| {
            let mut result = Vec::new();
            
            for (_, _, svm) in &self.inner.classifiers {
                if let Some(ref sv) = svm.support_vectors {
                    match &sv.storage {
                        DatasetStorage::Owned(mat) => {
                            let n_samples = mat.nrows();
                            let n_features = mat.ncols();
                            
                            let mut flat_data = Vec::with_capacity(n_samples * n_features);
                            for i in 0..n_samples {
                                for j in 0..n_features {
                                    flat_data.push(mat[(i, j)]);
                                }
                            }
                            
                            let array = PyArray1::from_vec(py, flat_data);
                            let reshaped = array.reshape((n_samples, n_features))?;
                            result.push(reshaped.to_owned());
                        }
                        _ => {
                            return Err(PyValueError::new_err("Support vectors not properly stored"));
                        }
                    }
                }
            }
            
            Ok(result)
        })
    }
    
    #[getter]
    pub fn support_labels(&self) -> PyResult<Vec<Py<PyArray1<f64>>>> {
        Python::with_gil(|py| {
            let mut result = Vec::new();
            
            for (_, _, svm) in &self.inner.classifiers {
                if let Some(ref sl) = svm.support_labels {
                    let n_samples = sl.nrows();
                    let mut labels = Vec::with_capacity(n_samples);
                    
                    for i in 0..n_samples {
                        labels.push(sl[(i, 0)]);
                    }
                    
                    result.push(PyArray1::from_vec(py, labels).to_owned());
                }
            }
            
            Ok(result)
        })
    }
    
    #[getter]
    pub fn alphas(&self) -> PyResult<Vec<Py<PyArray1<f64>>>> {
        Python::with_gil(|py| {
            let mut result = Vec::new();
            
            for (_, _, svm) in &self.inner.classifiers {
                if let Some(ref al) = svm.alphas {
                    let n_samples = al.nrows();
                    let mut alphas = Vec::with_capacity(n_samples);
                    
                    for i in 0..n_samples {
                        alphas.push(al[(i, 0)]);
                    }
                    
                    result.push(PyArray1::from_vec(py, alphas).to_owned());
                }
            }
            
            Ok(result)
        })
    }
    
    #[getter]
    pub fn bias(&self) -> Vec<f64> {
        self.inner.classifiers
            .iter()
            .map(|(_, _, svm)| svm.bias)
            .collect()
    }
    
    pub fn copy(&self) -> PyResult<Self> {
        Ok(PySVM {
            inner: SVM {
                classifiers: self.inner.classifiers.clone(),
                classes: self.inner.classes.clone(),
                c: self.inner.c,
                kernel_type: self.inner.kernel_type.clone(),
                degree: self.inner.degree,
                coef0: self.inner.coef0,
                gamma: self.inner.gamma,
            }
        })
    }
    
    pub fn __str__(&self) -> String {
        format!(
            "SVM(kernel='{}', C={}, n_classes={}, n_classifiers={})",
            self.inner.kernel_type,
            self.inner.c,
            self.inner.classes.len(),
            self.inner.classifiers.len()
        )
    }
    
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}