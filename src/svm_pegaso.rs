use pyo3::prelude::*;
use ndarray::{Array2, Array1, Axis, s};
use rand::{seq::SliceRandom, thread_rng, Rng};

#[pyclass]
pub struct SVMClassifier {
    weights: Option<Array2<f64>>,
    n_classes: usize,
}

#[pymethods]
impl SVMClassifier {
    #[new]
    fn new() -> Self {
        Self { weights: None, n_classes: 0 }
    }
    fn fit(&mut self,
           X: Vec<Vec<f64>>,
           y: Vec<usize>,
           n_iter: usize,
           C: f64) -> PyResult<()> {

        let n_samples = X.len();
        if n_samples == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Empty X"));
        }
        let n_features = X[0].len();
        let flat: Vec<f64> = X.into_iter().flatten().collect();
        let mut X = Array2::from_shape_vec((n_samples, n_features), flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let y = Array1::<usize>::from_vec(y);
        self.n_classes = *y.iter().max().unwrap() + 1;
        let k = self.n_classes;

        for mut col in X.gencolumns_mut() {
            let mean = col.mean().unwrap();
            let std = col.std(0.0);
            if std > 0.0 {
                col -= mean;
                col /= std;
            }
        }

        let mut Xb = Array2::<f64>::ones((n_samples, n_features + 1));
        Xb.slice_mut(s![.., ..n_features]).assign(&X);

        let mut rng = thread_rng();
        let mut W = Array2::<f64>::zeros((k, n_features + 1));
        let scale = 1.0_f64 / (n_features as f64).sqrt();
        for mut row in W.axis_iter_mut(Axis(0)) {
            row.mapv_inplace(|_| rng.gen_range(-scale..scale));
        }

        let lambda = 1.0 / (C * n_samples as f64);

        let mut index: Vec<usize> = (0..n_samples).collect();
        let mut t = 1usize;

        for _epoch in 0..n_iter {
            index.shuffle(&mut rng);

            for &i in &index {
                let xi = Xb.slice(s![i, ..]);
                let yi = y[i];

                for class in 0..k {
                    let y_bin = if class == yi { 1.0 } else { -1.0 };
                    let mut w = W.row(class).to_owned();

                    let eta = 1.0 / (lambda * t as f64);

                    let margin = y_bin * w.dot(&xi);

                    let mut update = w.slice_mut(s![..n_features]);
                    update.mapv_inplace(|v| (1.0 - eta * lambda) * v);
                    if margin < 1.0 {
                        for j in 0..n_features {
                            update[j] += eta * y_bin * xi[j];
                        }
                        w[n_features] += eta * y_bin;
                    }
                    W.row_mut(class).assign(&w);
                }
                t += 1;
            }
        }

        self.weights = Some(W);
        Ok(())
    }

    fn predict(&self, X: Vec<Vec<f64>>) -> PyResult<Vec<usize>> {
        let w = self.weights.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Call fit() first"))?;

        let n_samples = X.len();
        if n_samples == 0 { return Ok(vec![]); }

        let n_features = w.shape()[1] - 1;
        let flat: Vec<f64> = X.into_iter().flatten().collect();
        let mut X = Array2::from_shape_vec((n_samples, n_features), flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        for mut col in X.gencolumns_mut() {
            let std = col.std(0.0);
            if std > 0.0 { col /= std; }
        }

        let mut Xb = Array2::<f64>::ones((n_samples, n_features + 1));
        Xb.slice_mut(s![.., ..n_features]).assign(&X);

        let logits = Xb.dot(&w.t());
        let preds = logits
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();
        Ok(preds)
    }
}
