use pyo3::prelude::*;
use ndarray::{Array2, Array1};
use pyo3::types::PyList;

#[pyclass]
pub struct KNNClassifier {
    k: usize,
    X: Vec<Vec<f64>>,
    y: Vec<usize>,
}

#[pymethods]
impl KNNClassifier {
    #[new]
    pub fn new(k: usize) -> Self {
        KNNClassifier { k, X: Vec::new(), y: Vec::new() }
    }

    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<usize>) {
        self.X = X;
        self.y = y;
    }

    pub fn predict(&self, X_test: Vec<Vec<f64>>) -> Vec<usize> {
        X_test.iter().map(|x| self.predict_one(x)).collect()
    }
}

impl KNNClassifier {
    fn predict_one(&self, x: &Vec<f64>) -> usize {
        let mut dists: Vec<(f64, usize)> = self.X
            .iter()
            .zip(&self.y)
            .map(|(xi, &yi)| {
                let dist = xi.iter().zip(x).map(|(a, b)| (a-b).powi(2)).sum::<f64>().sqrt();
                (dist, yi)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut votes = std::collections::HashMap::new();
        for &(_, label) in dists.iter().take(self.k) {
            *votes.entry(label).or_insert(0) += 1;
        }
        *votes.iter().max_by_key(|(_, count)| *count).unwrap().0
    }
}