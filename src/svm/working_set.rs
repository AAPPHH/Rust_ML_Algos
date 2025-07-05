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
    sorted_indices: Vec<usize>,
    last_violation_sum: f64,
}

impl PartialArgMaxSelector {
    pub fn new(n: usize) -> Self {
        let k_size = ((n as f64).sqrt() as usize).max(20).min(200);
        
        let mut grad_cache = AlignedVec::with_capacity(n);
        grad_cache.resize(n, 0.0);
        
        let mut violations = AlignedVec::with_capacity(n);
        violations.resize(n, 0.0);
        
        Self {
            n,
            violations,
            indices: Vec::with_capacity(n),
            top_k_buffer: Vec::with_capacity(k_size * 2),
            grad_cache,
            k_size,
            shrink_threshold: 0.1,
            iteration_count: 0,
            sorted_indices: (0..n).collect(),
            last_violation_sum: 0.0,
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
        
        let heap = self.find_top_k_violations_fast(alphas, y, c, active_indices, TOL);
        
        if heap.is_empty() {
            return None;
        }
        
        let mut best_pair = None;
        let mut best_gain = -1.0;
        
        let search_limit = (self.k_size as f64 * 1.5) as usize;
        
        for (idx, &(i, i_violation)) in heap.iter().take(search_limit).enumerate() {
            let gi = self.grad_cache[i];
            let kii = kernel_cache.get_diagonal(i);
            
            if idx + 1 < heap.len() {
                let next_i = heap[idx + 1].0;
                kernel_cache.prefetch_row(next_i);
            }
            
            if let Some((j, gain)) = self.find_best_partner_fast(
                i, gi, kii, alphas, y, c, kernel_cache, &heap, TOL
            ) {
                if gain > best_gain {
                    best_gain = gain;
                    best_pair = Some((i, j, i_violation));
                }
            }
        }
        
        if self.iteration_count % 20 == 0 {
            self.adapt_parameters_smart(best_gain, heap.len());
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
    fn find_top_k_violations_fast(
        &mut self,
        alphas: &[f64],
        y: &[f64],
        c: f64,
        active_indices: &[usize],
        tol: f64,
    ) -> Vec<(usize, f64)> {
        self.top_k_buffer.clear();
        
        if active_indices.len() > 500 {
            let violations: Vec<(usize, f64)> = active_indices
                .par_iter()
                .filter_map(|&i| {
                    let ai = unsafe { *alphas.get_unchecked(i) };
                    let yi = unsafe { *y.get_unchecked(i) };
                    let gi = unsafe { *self.grad_cache.get_unchecked(i) };
                    let yi_gi = yi * gi;
                    
                    let violation = if ai < c - tol && yi_gi < -tol {
                        Some(-yi_gi)
                    } else if ai > tol && yi_gi > tol {
                        Some(yi_gi)
                    } else {
                        None
                    };
                    
                    violation.map(|v| (i, v))
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
            
            for &i in &self.sorted_indices {
                if !active_indices.contains(&i) {
                    continue;
                }
                
                let ai = unsafe { *alphas.get_unchecked(i) };
                let yi = unsafe { *y.get_unchecked(i) };
                let gi = unsafe { *self.grad_cache.get_unchecked(i) };
                let yi_gi = yi * gi;
                
                let violation = if ai < c - tol && yi_gi < -tol {
                    Some(-yi_gi)
                } else if ai > tol && yi_gi > tol {
                    Some(yi_gi)
                } else {
                    None
                };
                
                if let Some(v) = violation {
                    violations.push((i, v));
                }
            }
            
            violations.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            violations.truncate(self.k_size);
            violations
        }
    }
    
    #[inline]
    fn find_best_partner_fast<C: KernelCache>(
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
        let mut max_gain = 0.0;
        
        let candidate_limit = (self.k_size * 3).min(candidates.len());
        
        for (idx, &(j, _)) in candidates.iter().take(candidate_limit).enumerate() {
            if j == i { continue; }
            
            let aj = alphas[j];
            let yj = y[j];
            let gj = self.grad_cache[j];
            let yj_gj = yj * gj;
            
            let feasible = (aj < c - tol && yj_gj < -tol) || 
                          (aj > tol && yj_gj > tol);
            
            if !feasible {
                continue;
            }
            
            let diff = gi - gj;
            let diff_sq = diff * diff;
            
            if diff_sq <= max_gain * 1.001 {
                continue;
            }
            
            if idx + 1 < candidate_limit {
                let next_j = candidates[idx + 1].0;
                let (_, set_idx) = kernel_cache.hash_key(i, next_j);
                kernel_cache.prefetch_cache_line(set_idx);
            }
            
            let kjj = kernel_cache.get_diagonal(j);
            let kij = kernel_cache.get(i, j);
            let eta = kii + kjj - 2.0 * kij;
            
            if eta > 1e-12 {
                let gain = diff_sq / eta;
                if gain > max_gain {
                    max_gain = gain;
                    best_j = Some(j);
                }
            }
        }
        
        best_j.map(|j| (j, max_gain))
    }
    
    fn adapt_parameters_smart(&mut self, current_gain: f64, num_violations: usize) {
        let violation_sum = self.violations.as_slice().iter().sum::<f64>();
        let violation_change = (violation_sum - self.last_violation_sum).abs();
        self.last_violation_sum = violation_sum;
        
        if violation_change < 0.01 * violation_sum {
            self.k_size = (self.k_size * 5 / 4).min(200).min(num_violations);
        } else if current_gain < self.shrink_threshold {
            self.k_size = (self.k_size * 3 / 2).min(200);
            self.shrink_threshold *= 0.95;
        } else if current_gain > self.shrink_threshold * 20.0 {
            self.k_size = (self.k_size * 3 / 4).max(20);
        }
        
        if self.iteration_count % 100 == 0 {
            self.update_sorted_indices();
        }
    }
    
    fn update_sorted_indices(&mut self) {
        let mut violation_counts: Vec<(usize, f64)> = self.sorted_indices
            .iter()
            .map(|&i| (i, self.violations[i]))
            .collect();
            
        violation_counts.sort_unstable_by(|a, b| 
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        );
        
        self.sorted_indices = violation_counts.into_iter()
            .map(|(i, _)| i)
            .collect();
    }
}

pub struct ShrinkingWorkingSet {
    active_mask: Vec<bool>,
    violations: AlignedVec<f64>,
    sorted_indices: Vec<usize>,
    shrink_counter: usize,
    last_active_count: usize,
    consecutive_no_change: usize,
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
            last_active_count: n,
            consecutive_no_change: 0,
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
        
        let shrink_tol = self.compute_adaptive_tolerance(tol);
        
        if n > 1000 {
            self.parallel_shrinking_check(alphas_mat, y_mat, grad_mat, c, shrink_tol);
        } else {
            self.sequential_shrinking_check(alphas_mat, y_mat, grad_mat, c, shrink_tol);
        }
        
        self.sorted_indices.clear();
        let mut active_count = 0;
        
        for i in 0..n {
            if self.active_mask[i] {
                self.sorted_indices.push(i);
                active_count += 1;
            }
        }
        
        self.sorted_indices.sort_unstable_by(|&a, &b| {
            self.violations[b].partial_cmp(&self.violations[a])
                .unwrap_or(Ordering::Equal)
        });
        
        let max_active = if self.shrink_counter < 10 {
            n
        } else if self.shrink_counter < 50 {
            (n * 3 / 4).max(1000)
        } else {
            (n / 2).max(500)
        };
        
        if self.sorted_indices.len() > max_active {
            self.sorted_indices.truncate(max_active);
        }
        
        if active_count == self.last_active_count {
            self.consecutive_no_change += 1;
        } else {
            self.consecutive_no_change = 0;
        }
        self.last_active_count = active_count;
        
        // Reset if stuck
        if self.consecutive_no_change > 10 {
            self.reset_active_set(n);
        }
        
        self.sorted_indices.clone()
    }
    
    fn compute_adaptive_tolerance(&self, base_tol: f64) -> f64 {
        if self.shrink_counter > 100 {
            base_tol * 20.0
        } else if self.shrink_counter > 50 {
            base_tol * 10.0
        } else if self.shrink_counter > 20 {
            base_tol * 5.0
        } else {
            base_tol * 2.0
        }
    }
    
    fn parallel_shrinking_check(
        &mut self,
        alphas_mat: MatRef<f64>,
        y_mat: MatRef<f64>,
        grad_mat: MatRef<f64>,
        c: f64,
        shrink_tol: f64,
    ) {
        let n = alphas_mat.nrows();
        
        let results: Vec<(bool, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let ai = unsafe { *alphas_mat.get_unchecked(i, 0) };
                let yi = unsafe { *y_mat.get_unchecked(i, 0) };
                let gi = unsafe { *grad_mat.get_unchecked(i, 0) };
                let yi_gi = yi * gi;
                
                let at_lower = ai <= 1e-8;
                let at_upper = ai >= c - 1e-8;
                
                let should_shrink = (at_lower && yi_gi >= 1.0 - shrink_tol) || 
                                   (at_upper && yi_gi <= -1.0 + shrink_tol);
                
                let violation = if should_shrink { 
                    0.0 
                } else { 
                    yi_gi.abs() 
                };
                
                (!should_shrink, violation)
            })
            .collect();
        
        for (i, (active, violation)) in results.into_iter().enumerate() {
            self.active_mask[i] = active;
            self.violations[i] = violation;
        }
    }
    
    fn sequential_shrinking_check(
        &mut self,
        alphas_mat: MatRef<f64>,
        y_mat: MatRef<f64>,
        grad_mat: MatRef<f64>,
        c: f64,
        shrink_tol: f64,
    ) {
        let n = alphas_mat.nrows();
        
        for i in 0..n {
            let ai = unsafe { *alphas_mat.get_unchecked(i, 0) };
            let yi = unsafe { *y_mat.get_unchecked(i, 0) };
            let gi = unsafe { *grad_mat.get_unchecked(i, 0) };
            let yi_gi = yi * gi;
            
            let at_lower = ai <= 1e-8;
            let at_upper = ai >= c - 1e-8;
            
            let should_shrink = (at_lower && yi_gi >= 1.0 - shrink_tol) || 
                               (at_upper && yi_gi <= -1.0 + shrink_tol);
            
            self.active_mask[i] = !should_shrink;
            self.violations[i] = if should_shrink { 0.0 } else { yi_gi.abs() };
        }
    }
    
    fn reset_active_set(&mut self, n: usize) {
        self.active_mask.fill(true);
        self.consecutive_no_change = 0;
        self.shrink_counter = 0;
    }
}