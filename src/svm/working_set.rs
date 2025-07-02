use crate::svm::flat_kernel_cache::FlatKernelCache;
use faer::{Mat, MatRef};

pub struct WorkingSetSelector {
    violations: Vec<f64>,
    indices: Vec<usize>,
    candidates: Vec<(usize, f64)>,
}

impl WorkingSetSelector {
    pub fn new(n: usize) -> Self {
        Self {
            violations: Vec::with_capacity(n),
            indices: Vec::with_capacity(n),
            candidates: Vec::with_capacity(100),
        }
    }

    pub fn select_working_set(
        &mut self,
        alphas: &[f64],
        y: &[f64],
        grad: &[f64],
        c: f64,
        kernel_cache: &mut FlatKernelCache,
        active_indices: &[usize],
    ) -> Option<((usize, usize), f64)> {
        let tol = 1e-3;
        
        // Vektorisierte KKT-Verletzungsberechnung
        self.violations.clear();
        self.indices.clear();
        
        for &i in active_indices {
            let ai = alphas[i];
            let yi_gi = y[i] * grad[i];
            
            let violation = if (ai < c && yi_gi < -tol) || (ai > 0.0 && yi_gi > tol) {
                yi_gi.abs()
            } else {
                0.0
            };
            
            if violation > tol {
                self.violations.push(violation);
                self.indices.push(i);
            }
        }
        
        if self.violations.is_empty() {
            return None;
        }
        
        // Finde maximale Verletzung
        let (max_idx, &max_violation) = self.violations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;
        
        let i = self.indices[max_idx];
        let gi = grad[i];
        let kii = kernel_cache.get_diagonal(i);
        
        // Finde bestes j
        let mut max_gain = -1.0;
        let mut j_best = None;
        
        for (idx, &j) in self.indices.iter().enumerate() {
            if j == i { continue; }
            
            let kjj = kernel_cache.get_diagonal(j);
            let kij = kernel_cache.get(i, j);
            let eta = kii + kjj - 2.0 * kij;
            
            if eta > 0.0 {
                let gain = (gi - grad[j]).powi(2) / eta;
                if gain > max_gain {
                    max_gain = gain;
                    j_best = Some(idx);
                }
            }
        }
        
        let j_idx = j_best?;
        let j = self.indices[j_idx];
        
        // Zurück zu active_indices Position
        let i_pos = active_indices.iter().position(|&x| x == i)?;
        let j_pos = active_indices.iter().position(|&x| x == j)?;
        
        Some(((i_pos, j_pos), max_violation))
    }
}

// Spezialisierte Version für Shrinking
pub struct ShrinkingWorkingSet {
    active_mask: Mat<f64>,
    grad_violations: Mat<f64>,
    sorted_indices: Vec<usize>,
}

impl ShrinkingWorkingSet {
    pub fn new(n: usize) -> Self {
        Self {
            active_mask: Mat::from_fn(n, 1, |_, _| 1.0),
            grad_violations: Mat::zeros(n, 1),
            sorted_indices: Vec::with_capacity(n),
        }
    }
    
    pub fn update_active_set(
        &mut self,
        alphas_mat: MatRef<f64>,
        y_mat: MatRef<f64>,
        grad_mat: MatRef<f64>,
        c: f64,
        tol: f64,
    ) -> Vec<usize> {
        let n = alphas_mat.nrows();
        
        for i in 0..n {
            let ai = alphas_mat[(i, 0)];
            let yi = y_mat[(i, 0)];
            let gi = grad_mat[(i, 0)];
            let yi_gi = yi * gi;
            
            let should_shrink = (ai <= 0.0 && yi_gi >= 1.0 - tol) || 
                               (ai >= c && yi_gi <= -1.0 + tol);
            
            self.active_mask[(i, 0)] = if should_shrink { 0.0 } else { 1.0 };
            self.grad_violations[(i, 0)] = if should_shrink { 0.0 } else { yi_gi.abs() };
        }
        
        self.sorted_indices.clear();
        for i in 0..n {
            if self.active_mask[(i, 0)] > 0.5 {
                self.sorted_indices.push(i);
            }
        }
        
        self.sorted_indices.sort_unstable_by(|&a, &b| {
            self.grad_violations[(b, 0)]
                .partial_cmp(&self.grad_violations[(a, 0)])
                .unwrap()
        });
        
        self.sorted_indices.clone()
    }
}

pub fn select_working_set_wss2_flat_cache(
    alphas: &[f64],
    y: &[f64],
    grad: &[f64],
    c: f64,
    kernel_cache: &mut FlatKernelCache,
    active_indices: &[usize],
) -> Option<((usize, usize), f64)> {
    let mut selector = WorkingSetSelector::new(active_indices.len());
    selector.select_working_set(alphas, y, grad, c, kernel_cache, active_indices)
}