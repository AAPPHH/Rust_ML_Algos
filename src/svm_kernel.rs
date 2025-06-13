use ndarray::{Array2, ArrayView1};
use ndarray::{Array1, Zip};
use std::collections::{HashMap, VecDeque};


#[derive(Clone)]
pub enum KernelType {
    Poly { degree: u32, coef0: f64, gamma: f64 },
    RBF { gamma: f64 },
    Linear,
}

impl KernelType {
    pub fn compute_pair(&self, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
        match self {
            KernelType::Poly { degree, coef0, gamma } =>
                (gamma * x.dot(&y) + *coef0).powi(*degree as i32),
            KernelType::RBF { gamma } => {
                let diff = &x - &y;
                (-gamma * diff.dot(&diff)).exp()
            }
            KernelType::Linear => x.dot(&y),
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


pub struct KernelCache {
    kernel: KernelType,
    x: Array2<f64>,
    cache: HashMap<(usize, usize), f64>,
    order: VecDeque<(usize, usize)>,
    max_size: usize,
}

impl KernelCache {
    pub fn new(kernel: KernelType, x: Array2<f64>, max_size: usize) -> Self {
        Self {
            kernel,
            x,
            cache: HashMap::new(),
            order: VecDeque::new(),
            max_size,
        }
    }

    pub fn get(&mut self, i: usize, j: usize) -> f64 {
        let key = if i <= j { (i, j) } else { (j, i) };
        if let Some(&val) = self.cache.get(&key) {
            val
        } else {
            let xi = self.x.row(i);
            let xj = self.x.row(j);
            let val = self.kernel.compute_pair(xi, xj);
            self.insert(key, val);
            val
        }
    }

    fn insert(&mut self, key: (usize, usize), val: f64) {
        self.cache.insert(key, val);
        self.order.push_back(key);
        if self.cache.len() > self.max_size {
            if let Some(old_key) = self.order.pop_front() {
                self.cache.remove(&old_key);
            }
        }
    }
}

