use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array2, Array1, Axis, s};
use rand::prelude::*;
use pyo3::prelude::*; 

/// Einfache logistische Regression (Multiclass, one-vs-rest)
#[pyclass]
pub struct LogisticRegression {
    weights: Option<Array2<f64>>,
    n_classes: usize,
}

#[pymethods]
impl LogisticRegression {
    #[new]
    fn new() -> Self {
        LogisticRegression { weights: None, n_classes: 0 }
    }

    fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<usize>, n_iter: usize, lr: f64) {
        let X = Array2::from_shape_vec((X.len(), X[0].len()), X.concat()).unwrap();
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let y = Array1::from_vec(y);
        let n_classes = *y.iter().max().unwrap() + 1;
        self.n_classes = n_classes;

        let X = X.insert_axis(Axis(1)).to_owned();
        let X = X.slice(s![.., 0, ..]).to_owned();
        let mut Xb = Array2::<f64>::ones((n_samples, n_features + 1));
        Xb.slice_mut(s![.., ..n_features]).assign(&X);

        let mut weights = Array2::<f64>::zeros((n_classes, n_features + 1));
        let mut rng = rand::thread_rng();
        weights.mapv_inplace(|_| rng.gen_range(-0.01..0.01));

        for class in 0..n_classes {
            let mut w = weights.slice_mut(s![class, ..]);
            let y_bin = y.mapv(|v| if v == class { 1.0 } else { 0.0 });
            for _ in 0..n_iter {
                let logits = Xb.dot(&w.to_owned());
                let preds = logits.mapv(|z| 1.0 / (1.0 + (-z).exp()));
                let grad = Xb.t().dot(&(preds - &y_bin)) / n_samples as f64;
                w -= &(lr * grad);
            }
        }
        self.weights = Some(weights);
    }

    fn predict<'py>(&self, py: Python<'py>, X: Vec<Vec<f64>>) -> &'py PyArray1<usize> {
        let X = Array2::from_shape_vec((X.len(), X[0].len()), X.concat()).unwrap();
        let n_samples = X.nrows();
        let n_features = X.ncols();

        let mut Xb = Array2::<f64>::ones((n_samples, n_features + 1));
        Xb.slice_mut(s![.., ..n_features]).assign(&X);

        let weights = self.weights.as_ref().expect("Must call fit() first");
        let logits = Xb.dot(&weights.t());
        let preds = logits.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        });
        preds.into_pyarray(py)
    }
}