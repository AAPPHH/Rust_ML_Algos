use crate::svm::cache::KernelCache;
use crate::svm::memory::{AlignedVec, get_pooled_vec};
use faer::MatRef;
use std::cmp::Ordering;
use rayon::prelude::*;

pub struct PartialArgMaxSelector {
    n: usize,
    violations: AlignedVec<f64>,
    indices: Vec<usize>,
    top_k_buffer: Vec<(usize, f64)>,
    grad_cache: AlignedVec<f64>,
    k_size: usize,
    shrink_threshold: f64,
    iteration_count: usize,
}

impl PartialArgMaxSelector {
    pub fn new(n: usize) -> Self {
        let k_size = (n as f64).sqrt() as usize;
        
        let mut grad_cache = AlignedVec::with_capacity(n);
        grad_cache.resize(n, 0.0);
        
        Self {
            n,
            violations: AlignedVec::with_capacity(n),
            indices: Vec::with_capacity(n),
            top_k_buffer: Vec::with_capacity(k_size * 2),
            grad_cache,
            k_size: k_size.max(10).min(100),
            shrink_threshold: 0.1,
            iteration_count: 0,
        }
    }
    
    #[inline]
    pub fn select_working_set_optimized<C: KernelCache>(
        &mut self,
        alphas: &[f64],
        y: &[f64],
        grad: &[f64],
        c: f64,
        kernel_cache: &mut C,
        active_indices: &[usize],
    ) -> Option<((usize, usize), f64)> {
        const TOL: f64 = 1e-3;
        
        self.iteration_count += 1;
        
        for &i in active_indices {
            self.grad_cache[i] = grad[i];
        }
        
        self.violations.clear();
        self.indices.clear();
        
        let mut heap = self.find_top_k_violations(alphas, y, c, active_indices, TOL);
        
        if heap.is_empty() {
            return None;
        }
        
        let mut best_pair = None;
        let mut best_gain = -1.0;
        
        for &(i, i_violation) in heap.iter().take(self.k_size) {
            let gi = self.grad_cache[i];
            let kii = kernel_cache.get_diagonal(i);
            
            if let Some((j, gain)) = self.find_best_partner(
                i, gi, kii, alphas, y, c, kernel_cache, &heap, TOL
            ) {
                if gain > best_gain {
                    best_gain = gain;
                    best_pair = Some((i, j, i_violation));
                }
            }
        }
        
        if self.iteration_count % 50 == 0 {
            self.adapt_parameters(best_gain);
        }
        
        if let Some((i, j, violation)) = best_pair {
            let i_pos = active_indices.iter().position(|&x| x == i)?;
            let j_pos = active_indices.iter().position(|&x| x == j)?;
            Some(((i_pos, j_pos), violation))
        } else {
            None
        }
    }
    
    #[inline]
    fn find_top_k_violations(
        &mut self,
        alphas: &[f64],
        y: &[f64],
        c: f64,
        active_indices: &[usize],
        tol: f64,
    ) -> Vec<(usize, f64)> {
        self.top_k_buffer.clear();
        
        if active_indices.len() > 1000 {
            let violations: Vec<(usize, f64)> = active_indices
                .par_iter()
                .filter_map(|&i| {
                    let ai = unsafe { *alphas.get_unchecked(i) };
                    let yi_gi = unsafe { y.get_unchecked(i) * self.grad_cache.get_unchecked(i) };
                    
                    let at_lower = ai < c - tol;
                    let at_upper = ai > tol;
                    let violates_lower = at_lower && yi_gi < -tol;
                    let violates_upper = at_upper && yi_gi > tol;
                    
                    if violates_lower || violates_upper {
                        Some((i, yi_gi.abs()))
                    } else {
                        None
                    }
                })
                .collect();
            
            let mut sorted = violations;
            let k = self.k_size.min(sorted.len());
            if k > 0 {
                sorted.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                
                sorted.truncate(k);
                sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }
            sorted
        } else {

            let mut violations = Vec::new();
            
            for &i in active_indices {
                let ai = unsafe { *alphas.get_unchecked(i) };
                let yi_gi = unsafe { y.get_unchecked(i) * self.grad_cache.get_unchecked(i) };
                
                let at_lower = ai < c - tol;
                let at_upper = ai > tol;
                let violates_lower = at_lower && yi_gi < -tol;
                let violates_upper = at_upper && yi_gi > tol;
                
                if violates_lower || violates_upper {
                    violations.push((i, yi_gi.abs()));
                }
            }
            
            violations.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            violations.truncate(self.k_size);
            violations
        }
    }
    
    #[inline]
    fn find_best_partner<C: KernelCache>(
        &mut self,
        i: usize,
        gi: f64,
        kii: f64,
        alphas: &[f64],
        y: &[f64],
        c: f64,
        kernel_cache: &mut C,
        candidates: &[(usize, f64)],
        tol: f64,
    ) -> Option<(usize, f64)> {
        let mut best_j = None;
        let mut max_gain = -1.0;
        
        for &(j, _) in candidates.iter().take(self.k_size * 2) {
            if j == i { continue; }
            
            let aj = alphas[j];
            let yj = y[j];
            
            let at_lower = aj < c - tol;
            let at_upper = aj > tol;
            let yj_gj = yj * self.grad_cache[j];
            
            if !((at_lower && yj_gj < -tol) || (at_upper && yj_gj > tol)) {
                continue;
            }
            
            let gj = self.grad_cache[j];
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
                    max_gain = gain;
                    best_j = Some(j);
                }
            }
        }
        
        best_j.map(|j| (j, max_gain))
    }
    
    fn adapt_parameters(&mut self, current_gain: f64) {
        if current_gain < self.shrink_threshold {
            self.k_size = (self.k_size * 3 / 2).min(100);
            self.shrink_threshold *= 0.9;
        } else if current_gain > self.shrink_threshold * 10.0 {
            self.k_size = (self.k_size * 2 / 3).max(10);
        }
    }
}

pub struct ShrinkingWorkingSet {
    active_mask: Vec<bool>,
    violations: AlignedVec<f64>,
    sorted_indices: Vec<usize>,
    shrink_counter: usize,
}

impl ShrinkingWorkingSet {
    pub fn new(n: usize) -> Self {
        let mut violations = AlignedVec::with_capacity(n);
        violations.resize(n, 0.0);
        
        Self {
            active_mask: vec![true; n],
            violations,
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
        } else if self.shrink_counter > 50 {
            tol * 5.0
        } else {
            tol * 2.0
        };
        
        if n > 1000 {
            let active_flags: Vec<(usize, bool, f64)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let ai = unsafe { *alphas_mat.get_unchecked(i, 0) };
                    let yi = unsafe { *y_mat.get_unchecked(i, 0) };
                    let gi = unsafe { *grad_mat.get_unchecked(i, 0) };
                    let yi_gi = yi * gi;
                    
                    let at_bound_zero = ai <= 1e-8;
                    let at_bound_c = ai >= c - 1e-8;
                    let should_shrink = (at_bound_zero && yi_gi >= 1.0 - shrink_tol) || 
                                       (at_bound_c && yi_gi <= -1.0 + shrink_tol);
                    
                    (i, !should_shrink, if should_shrink { 0.0 } else { yi_gi.abs() })
                })
                .collect();
            
            self.sorted_indices.clear();
            for (i, active, violation) in active_flags {
                self.active_mask[i] = active;
                self.violations[i] = violation;
                if active {
                    self.sorted_indices.push(i);
                }
            }
        } else {
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
}