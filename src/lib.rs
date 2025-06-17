pub mod svm;
pub mod knn;
pub mod logistic_regression;
pub mod svm_pegaso;

use pyo3::prelude::*;

use crate::logistic_regression::LogisticRegression;
use crate::knn::KNNClassifier;
use crate::svm_pegaso::SVMClassifier;
use crate::svm::svm_py_wrapper::PySVM;

#[pymodule]
fn my_rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LogisticRegression>()?;
    m.add_class::<KNNClassifier>()?;
    m.add_class::<SVMClassifier>()?;
    m.add_class::<PySVM>()?;
    Ok(())
}
