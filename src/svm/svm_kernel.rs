use faer::{Mat, MatRef, RowRef};

#[derive(Clone)]
pub enum KernelType {
    Poly { degree: u32, coef0: f64, gamma: f64 },
    RBF { gamma: f64 },
    Linear,
}

impl KernelType {
    #[inline(always)]
    pub fn compute_pair_row(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        match self {
            KernelType::Poly { degree, coef0, gamma } => {
                let dot = (0..x.ncols())
                    .map(|i| x[i] * y[i])
                    .sum::<f64>();
                (gamma * dot + *coef0).powi(*degree as i32)
            }
            KernelType::RBF { gamma } => {
                let diff_sq = (0..x.ncols())
                    .map(|i| {
                        let d = x[i] - y[i];
                        d * d
                    })
                    .sum::<f64>();
                (-gamma * diff_sq).exp()
            }
            KernelType::Linear => {
                (0..x.ncols())
                    .map(|i| x[i] * y[i])
                    .sum()
            }
        }
    }

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

    pub fn compute_kernel(&self, x: &Mat<f64>, y: &Mat<f64>) -> Mat<f64> {
        match self {
            KernelType::Linear => x * y.transpose(),
            KernelType::Poly { degree, coef0, gamma } => {
                let mut mat = x * y.transpose();
                for i in 0..mat.nrows() {
                    for j in 0..mat.ncols() {
                        let v = mat[(i, j)];
                        mat[(i, j)] = (gamma * v + *coef0).powi(*degree as i32);
                    }
                }
                mat
            }
            KernelType::RBF { gamma } => {
                let x_norms: Vec<f64> = (0..x.nrows())
                    .map(|i| {
                        let row = x.row(i);
                        (0..row.ncols()).map(|j| row[j].powi(2)).sum()
                    })
                    .collect();

                let y_norms: Vec<f64> = (0..y.nrows())
                    .map(|i| {
                        let row = y.row(i);
                        (0..row.ncols()).map(|j| row[j].powi(2)).sum()
                    })
                    .collect();

                let mut dot = x * y.transpose();

                for i in 0..dot.nrows() {
                    for j in 0..dot.ncols() {
                        let v = dot[(i, j)];
                        dot[(i, j)] = (-gamma * (x_norms[i] + y_norms[j] - 2.0 * v)).exp();
                    }
                }
                dot
            }
        }
    }
}