use pyo3::prelude::*;
use faer::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct KNNClassifier {
    k: usize,
    X: Option<Mat<f64>>,
    y: Option<Vec<usize>>,
}

#[pymethods]
impl KNNClassifier {
    #[new]
    pub fn new(k: usize) -> Self {
        KNNClassifier { k, X: None, y: None }
    }

    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<usize>) {
        let n_samples = X.len();
        let n_features = X[0].len();
        let mat = Mat::from_fn(n_samples, n_features, |i, j| X[i][j]);

        self.X = Some(mat);
        self.y = Some(y);
    }

    pub fn predict(&self, X_test: Vec<Vec<f64>>) -> Vec<usize> {
        X_test.iter().map(|x| self.predict_one(x)).collect()
    }
}

impl KNNClassifier {
    fn predict_one(&self, x: &Vec<f64>) -> usize {
        let train_X = self.X.as_ref().expect("Model not fitted");
        let train_y = self.y.as_ref().expect("Model not fitted");

        let mut dists: Vec<(f64, usize)> = (0..train_X.nrows())
            .map(|i| {
                let row = train_X.row(i);
                let dist = row.iter()
                    .zip(x.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (dist, train_y[i])
            })
            .collect();

        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut votes = HashMap::new();
        for &(_, label) in dists.iter().take(self.k) {
            *votes.entry(label).or_insert(0) += 1;
        }
        *votes.iter().max_by_key(|(_, count)| *count).unwrap().0
    }
}
