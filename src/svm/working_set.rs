use crate::svm::flat_kernel_cache::FlatKernelCache;
use faer::MatRef;
use std::cmp::Ordering;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub struct WorkingSetSelector {
    violations: Vec<f64>,
    indices: Vec<usize>,
    candidates: Vec<(usize, f64)>,
    grad_cache: Vec<f64>,
    active_mask: Vec<bool>,
    top_pairs: Vec<(usize, usize, f64)>,
    pair_update_counter: usize,
}

impl WorkingSetSelector {
    pub fn new(n: usize) -> Self {
        Self {
            violations: Vec::with_capacity(n),
            indices: Vec::with_capacity(n),
            candidates: Vec::with_capacity(100),
            grad_cache: vec![0.0; n],
            active_mask: vec![true; n],
            top_pairs: Vec::with_capacity(20),
            pair_update_counter: 0,
        }
    }

    #[inline(always)]
    pub fn select_working_set(
        &mut self,
        alphas: &[f64],
        y: &[f64],
        grad: &[f64],
        c: f64,
        kernel_cache: &mut FlatKernelCache,
        active_indices: &[usize],
    ) -> Option<((usize, usize), f64)> {
        const TOL: f64 = 1e-3;
        
        self.pair_update_counter += 1;
        if self.pair_update_counter % 10 != 0 && !self.top_pairs.is_empty() {
            for &(i, j, expected_violation) in &self.top_pairs {
                if active_indices.contains(&i) && active_indices.contains(&j) {
                    let ai = alphas[i];
                    let aj = alphas[j];
                    let yi = y[i];
                    let yj = y[j];
                    
                    let valid_i = (ai > TOL && ai < c - TOL) || 
                                  (ai <= TOL && yi * grad[i] < -TOL) ||
                                  (ai >= c - TOL && yi * grad[i] > TOL);
                    let valid_j = (aj > TOL && aj < c - TOL) || 
                                  (aj <= TOL && yj * grad[j] < -TOL) ||
                                  (aj >= c - TOL && yj * grad[j] > TOL);
                    
                    if valid_i && valid_j {
                        let i_pos = active_indices.iter().position(|&x| x == i)?;
                        let j_pos = active_indices.iter().position(|&x| x == j)?;
                        return Some(((i_pos, j_pos), expected_violation));
                    }
                }
            }
        }
        
        for &i in active_indices {
            self.grad_cache[i] = grad[i];
        }
        
        self.violations.clear();
        self.indices.clear();

        if active_indices.len() > 1000 {
            let violations_par: Vec<(usize, f64)> = active_indices
                .par_iter()
                .filter_map(|&i| {
                    let ai = unsafe { *alphas.get_unchecked(i) };
                    let yi_gi = unsafe { y.get_unchecked(i) * self.grad_cache.get_unchecked(i) };
                    
                    let at_lower = ai < c - TOL;
                    let at_upper = ai > TOL;
                    let violates_lower = at_lower && yi_gi < -TOL;
                    let violates_upper = at_upper && yi_gi > TOL;
                    
                    if violates_lower || violates_upper {
                        Some((i, yi_gi.abs()))
                    } else {
                        None
                    }
                })
                .collect();
            
            for (idx, viol) in violations_par {
                self.violations.push(viol);
                self.indices.push(idx);
            }
        } else {
            for &i in active_indices {
                let ai = unsafe { *alphas.get_unchecked(i) };
                let yi_gi = unsafe { y.get_unchecked(i) * self.grad_cache.get_unchecked(i) };
                
                let at_lower = ai < c - TOL;
                let at_upper = ai > TOL;
                let violates_lower = at_lower && yi_gi < -TOL;
                let violates_upper = at_upper && yi_gi > TOL;
                
                if violates_lower || violates_upper {
                    self.violations.push(yi_gi.abs());
                    self.indices.push(i);
                }
            }
        }
        
        if self.violations.is_empty() {
            return None;
        }
        
        let mut max_violation = 0.0;
        let mut max_idx = 0;
        
        let mut idx = 0;
        while idx + 4 <= self.violations.len() {
            let v0 = self.violations[idx];
            let v1 = self.violations[idx + 1];
            let v2 = self.violations[idx + 2];
            let v3 = self.violations[idx + 3];
            
            if v0 > max_violation { max_violation = v0; max_idx = idx; }
            if v1 > max_violation { max_violation = v1; max_idx = idx + 1; }
            if v2 > max_violation { max_violation = v2; max_idx = idx + 2; }
            if v3 > max_violation { max_violation = v3; max_idx = idx + 3; }
            
            idx += 4;
        }
        
        while idx < self.violations.len() {
            if self.violations[idx] > max_violation {
                max_violation = self.violations[idx];
                max_idx = idx;
            }
            idx += 1;
        }
        
        let i = self.indices[max_idx];
        let gi = self.grad_cache[i];
        let kii = kernel_cache.get_diagonal(i);
        
        let mut max_gain = -1.0;
        let mut best_j_idx = None;
        let mut second_best_gain = -1.0;
        let mut second_best_j_idx = None;

        const BATCH_SIZE: usize = 32;
        
        for batch_start in (0..self.indices.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(self.indices.len());
            
            for idx in batch_start..batch_end {
                let j = unsafe { *self.indices.get_unchecked(idx) };
                if j == i { continue; }
                
                let gj = unsafe { *self.grad_cache.get_unchecked(j) };
                let diff = gi - gj;
                
                if diff * diff <= max_gain * 1e-8 {
                    continue;
                }
                
                let kjj = kernel_cache.get_diagonal(j);
                let kij = kernel_cache.get(i, j);
                let eta = kii + kjj - 2.0 * kij;
                
                if eta > 1e-8 {
                    let gain = diff * diff / eta;
                    if gain > max_gain {
                        second_best_gain = max_gain;
                        second_best_j_idx = best_j_idx;
                        max_gain = gain;
                        best_j_idx = Some(idx);
                    } else if gain > second_best_gain {
                        second_best_gain = gain;
                        second_best_j_idx = Some(idx);
                    }
                }
            }
        }
        
        let j_idx = best_j_idx?;
        let j = self.indices[j_idx];
        
        if self.pair_update_counter % 10 == 0 {
            self.top_pairs.clear();
            self.top_pairs.push((i, j, max_violation));
            if let Some(idx2) = second_best_j_idx {
                self.top_pairs.push((i, self.indices[idx2], max_violation * 0.9));
            }
        }
        
        let i_pos = active_indices.iter().position(|&x| x == i)?;
        let j_pos = active_indices.iter().position(|&x| x == j)?;
        
        Some(((i_pos, j_pos), max_violation))
    }
}

pub struct ShrinkingWorkingSet {
    active_mask: Vec<bool>,
    violations: Vec<f64>,
    sorted_indices: Vec<usize>,
    shrink_counter: usize,
}

impl ShrinkingWorkingSet {
    pub fn new(n: usize) -> Self {
        Self {
            active_mask: vec![true; n],
            violations: vec![0.0; n],
            sorted_indices: Vec::with_capacity(n),
            shrink_counter: 0,
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
        self.shrink_counter += 1;
        
        let shrink_tol = if self.shrink_counter > 100 {
            tol * 10.0
        } else {
            tol
        };
        

        let mut num_active = 0;
        
        for i in 0..n {
            let ai = unsafe { *alphas_mat.get_unchecked(i, 0) };
            let yi = unsafe { *y_mat.get_unchecked(i, 0) };
            let gi = unsafe { *grad_mat.get_unchecked(i, 0) };
            let yi_gi = yi * gi;
            
            let at_bound_zero = ai <= 1e-8;
            let at_bound_c = ai >= c - 1e-8;
            let should_shrink = (at_bound_zero && yi_gi >= 1.0 - shrink_tol) || 
                               (at_bound_c && yi_gi <= -1.0 + shrink_tol);
            
            self.active_mask[i] = !should_shrink;
            if !should_shrink {
                self.violations[i] = yi_gi.abs();
                num_active += 1;
            } else {
                self.violations[i] = 0.0;
            }
        }

        self.sorted_indices.clear();
        self.sorted_indices.reserve(num_active);
        
        for i in 0..n {
            if self.active_mask[i] {
                self.sorted_indices.push(i);
            }
        }
        
        self.sorted_indices.sort_unstable_by(|&a, &b| {
            self.violations[b].partial_cmp(&self.violations[a])
                .unwrap_or(Ordering::Equal)
        });
        
        if self.sorted_indices.len() > 1000 {
            self.sorted_indices.truncate(1000);
        }
        
        self.sorted_indices.clone()
    }
    
    pub fn should_reset(&self, active_count: usize, total: usize) -> bool {
        active_count < total / 20 || active_count < 10
    }
}