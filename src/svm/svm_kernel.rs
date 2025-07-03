use faer::{Mat, RowRef};
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};

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
                let dot = self.dot_product_optimized(x, y);
                (gamma * dot + *coef0).powi(*degree as i32)
            }
            KernelType::RBF { gamma } => {
                let diff_sq = self.squared_distance_optimized(x, y);
                (-gamma * diff_sq).exp()
            }
            KernelType::Linear => {
                self.dot_product_optimized(x, y)
            }
        }
    }

    #[inline(always)]
    fn dot_product_optimized(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        let n = x.ncols();
        let mut sum = 0.0;
        
        // Unroll für bessere Performance
        let mut i = 0;
        while i + 8 <= n {
            sum += x[i] * y[i] + x[i+1] * y[i+1] + 
                   x[i+2] * y[i+2] + x[i+3] * y[i+3] +
                   x[i+4] * y[i+4] + x[i+5] * y[i+5] +
                   x[i+6] * y[i+6] + x[i+7] * y[i+7];
            i += 8;
        }
        
        // Rest
        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }
        
        sum
    }

    #[inline(always)]
    fn squared_distance_optimized(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        let n = x.ncols();
        let mut sum = 0.0;
        
        // Unroll für bessere Performance
        let mut i = 0;
        while i + 4 <= n {
            let d0 = x[i] - y[i];
            let d1 = x[i+1] - y[i+1];
            let d2 = x[i+2] - y[i+2];
            let d3 = x[i+3] - y[i+3];
            
            sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
            i += 4;
        }
        
        // Rest
        while i < n {
            let d = x[i] - y[i];
            sum += d * d;
            i += 1;
        }
        
        sum
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
                mat.par_row_chunks_mut(64).for_each(|mut chunk| {
                    for i in 0..chunk.nrows() {
                        for j in 0..chunk.ncols() {
                            let v = chunk[(i, j)];
                            chunk[(i, j)] = (gamma * v + *coef0).powi(*degree as i32);
                        }
                    }
                });
                mat
            }
            KernelType::RBF { gamma } => {
                let x_norms: Vec<f64> = (0..x.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let row = x.row(i);
                        (0..row.ncols()).map(|j| row[j].powi(2)).sum()
                    })
                    .collect();

                let y_norms: Vec<f64> = (0..y.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let row = y.row(i);
                        (0..row.ncols()).map(|j| row[j].powi(2)).sum()
                    })
                    .collect();

                let mut dot = x * y.transpose();

                dot.par_row_chunks_mut(64).enumerate().for_each(|(chunk_idx, mut chunk)| {
                    let row_start = chunk_idx * 64;
                    for i in 0..chunk.nrows() {
                        let row_idx = row_start + i;
                        if row_idx < x_norms.len() {
                            for j in 0..chunk.ncols() {
                                let v = chunk[(i, j)];
                                chunk[(i, j)] = (-gamma * (x_norms[row_idx] + y_norms[j] - 2.0 * v)).exp();
                            }
                        }
                    }
                });
                
                dot
            }
        }
    }
}