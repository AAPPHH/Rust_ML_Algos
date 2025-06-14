use std::collections::{HashMap, VecDeque};
use crate::flat_svm::FlatDataset;

use ndarray::{Array1, Array2, ArrayView1, Zip};

#[derive(Clone)]
pub enum KernelType {
    Poly { degree: u32, coef0: f64, gamma: f64 },
    RBF { gamma: f64 },
    Linear,
}

impl KernelType {
    pub fn compute_pair_flat(&self, x: &[f64], y: &[f64]) -> f64 {
        match self {
            KernelType::Poly { degree, coef0, gamma } => {
                let dot = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
                (gamma * dot + *coef0).powi(*degree as i32)
            }
            KernelType::RBF { gamma } => {
                let diff = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                (-gamma * diff).exp()
            }
            KernelType::Linear => {
                x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
            }
        }
    }

    pub fn compute_kernel(&self, x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        match self {
            KernelType::Linear => {
                x.dot(&y.t())
            }
            KernelType::Poly { degree, coef0, gamma } => {
                let mut mat = x.dot(&y.t());
                mat.mapv_inplace(|v| (gamma * v + coef0).powi(*degree as i32));
                mat
            }
            KernelType::RBF { gamma } => {
                let x_norms = x
                    .rows()
                    .into_iter()
                    .map(|row| row.dot(&row))
                    .collect::<Array1<_>>();
                let y_norms = y
                    .rows()
                    .into_iter()
                    .map(|row| row.dot(&row))
                    .collect::<Array1<_>>();
                let dot = x.dot(&y.t());
                let mut k_mat = dot;
                Zip::from(k_mat.rows_mut())
                    .and(x_norms.view())
                    .for_each(|mut row, &xn| {
                        Zip::from(&mut row)
                            .and(y_norms.view())
                            .for_each(|k_ij, &yn| {
                                *k_ij = (-gamma * (xn + yn - 2.0 * *k_ij)).exp();
                            });
                    });
                k_mat
            }
        }
    }
}






pub struct FlatKernelCache {
    kernel: KernelType,
    dataset: FlatDataset,
    cache: HashMap<usize, Vec<f64>>,
    order: VecDeque<usize>,
    max_size: usize,
}

#[inline]
pub fn faer_dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

#[inline]
pub fn faer_sqdist(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| {
        let d = a - b;
        d * d
    }).sum()
}


impl FlatKernelCache {
    pub fn new(kernel: KernelType, dataset: FlatDataset, max_size: usize) -> Self {
        Self {
            kernel,
            dataset,
            cache: HashMap::new(),
            order: VecDeque::new(),
            max_size,
        }
    }

    pub fn get_row(&mut self, i: usize) -> &[f64] {
        if !self.cache.contains_key(&i) {
            let row = self.compute_kernel_row(i);
            self.insert(i, row);
        }
        self.cache.get(&i).unwrap()
    }

    fn compute_kernel_row(&self, i: usize) -> Vec<f64> {
        let xi = self.dataset.get_row(i);
        let mut row = Vec::with_capacity(self.dataset.n_samples);
        for j in 0..self.dataset.n_samples {
            let xj = self.dataset.get_row(j);
            let kij = self.kernel.compute_pair_flat(xi, xj);
            row.push(kij);
        }
        row
    }

    fn insert(&mut self, i: usize, row: Vec<f64>) {
        self.cache.insert(i, row);
        self.order.push_back(i);
        if self.cache.len() > self.max_size {
            if let Some(old_i) = self.order.pop_front() {
                self.cache.remove(&old_i);
            }
        }
    }

    pub fn get(&mut self, i: usize, j: usize) -> f64 {
        self.get_row(i)[j]
    }
}