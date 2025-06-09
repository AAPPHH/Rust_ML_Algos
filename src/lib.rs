mod logistic_regression;
mod knn;
mod svm_pegaso;
mod svm;

use pyo3::prelude::*;

#[pymodule]
fn my_rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<logistic_regression::LogisticRegression>()?;
    m.add_class::<knn::KNNClassifier>()?;
    m.add_class::<svm_pegaso::SVMClassifier>()?;
    m.add_class::<svm::SVM>()?;
    Ok(())
}
