use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::svm::svm::SVM;

#[pyclass]
pub struct PySVM {
    inner: SVM,
}

#[pymethods]
impl PySVM {
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

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>, max_iter: usize, tol: f64) -> PyResult<()> {
        self.inner.fit(x, y, max_iter, tol).map_err(|e| PyValueError::new_err(e))
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        Ok(self.inner.predict(x))
    }

    #[getter]
    pub fn kernel_type(&self) -> String {
        self.inner.kernel_type.clone()
    }
}
