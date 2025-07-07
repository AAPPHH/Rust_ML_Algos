use crate::svm::cache::KernelCache;
use crate::svm::memory::AlignedVec;
use faer::MatRef;
use std::cmp::Ordering;
use rayon::prelude::*;

pub struct WSS2Selector {
    n: usize,
    minus_y_grad: AlignedVec<f64>,
    sorted_indices: Vec<usize>,
    grad_cache: AlignedVec<f64>,
    shrink_counter: usize,
    active_size: usize,
}

impl WSS2Selector {
    pub fn new(n: usize) -> Self {
        let mut minus_y_grad = AlignedVec::with_capacity(n);
        minus_y_grad.resize(n, 0.0);
        
        let mut grad_cache = AlignedVec::with_capacity(n);
        grad_cache.resize(n, 0.0);
        
        Self {
            n,
            minus_y_grad,
            sorted_indices: (0..n).collect(),
            grad_cache,
            shrink_counter: 0,
            active_size: n,
        }
    }
    
    #[inline]
    pub fn select_working_set_wss2<C: KernelCache>(
        &mut self,
        alphas: &[f64],
        y: &[f64],
        grad: &[f64],
        c: f64,
        kernel_cache: &mut C,
        active_indices: &[usize],
    ) -> Option<((usize, usize), f64)> {
        const EPS: f64 = 1e-3;
        
        for &idx in active_indices {
            self.grad_cache[idx] = grad[idx];
            self.minus_y_grad[idx] = -y[idx] * grad[idx];
        }
        
        let (i_up, i_low) = self.find_working_sets(alphas, y, c, active_indices, EPS);
        
        if i_up.is_empty() || i_low.is_empty() {
            return None;
        }
        
        let i = *i_up.iter()
            .min_by(|&&a, &&b| {
                self.minus_y_grad[a].partial_cmp(&self.minus_y_grad[b])
                    .unwrap_or(Ordering::Equal)
            })?;
        
        let j = self.select_j_wss2(i, &i_low, kernel_cache)?;
        
        let violation = (self.minus_y_grad[j] - self.minus_y_grad[i]).abs();
        
        if violation < EPS {
            return None;
        }
        
        let i_pos = active_indices.iter().position(|&x| x == i)?;
        let j_pos = active_indices.iter().position(|&x| x == j)?;
        
        Some(((i_pos, j_pos), violation))
    }
    
    #[inline]
    fn find_working_sets(
        &self,
        alphas: &[f64],
        y: &[f64],
        c: f64,
        active_indices: &[usize],
        eps: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut i_up = Vec::with_capacity(active_indices.len() / 2);
        let mut i_low = Vec::with_capacity(active_indices.len() / 2);
        
        for &idx in active_indices {
            let alpha = alphas[idx];
            let y_val = y[idx];
            
            if (y_val > 0.0 && alpha < c - eps) || (y_val < 0.0 && alpha > eps) {
                i_up.push(idx);
            }
            
            if (y_val > 0.0 && alpha > eps) || (y_val < 0.0 && alpha < c - eps) {
                i_low.push(idx);
            }
        }
        
        (i_up, i_low)
    }
    
    #[inline]
    fn select_j_wss2<C: KernelCache>(
        &self,
        i: usize,
        i_low: &[usize],
        kernel_cache: &mut C,
    ) -> Option<usize> {
        let gi = self.grad_cache[i];
        let kii = kernel_cache.get_diagonal(i);
        
        let mut best_j = None;
        let mut max_gain = -1.0;
        
        if i_low.len() > 1 {
            for &j in i_low.iter().take(8) {
                let (_, set_idx) = kernel_cache.hash_key(i, j);
                kernel_cache.prefetch_cache_line(set_idx);
            }
        }
        
        for &j in i_low {
            if i == j { continue; }
            
            let gj = self.grad_cache[j];
            let b_ij = gi - gj;

            if b_ij <= 0.0 { continue; }
            
            let kjj = kernel_cache.get_diagonal(j);
            let kij = kernel_cache.get(i, j);
            let mut a_ij = kii + kjj - 2.0 * kij;
            
            if a_ij <= 0.0 {
                a_ij = 1e-12; 
            }
            
            let gain = b_ij * b_ij / a_ij;
            
            if gain > max_gain {
                max_gain = gain;
                best_j = Some(j);
            }
        }
        
        best_j
    }
    
    /// Optimierte WSS2 Variante für große Datensätze
    pub fn select_working_set_wss2_parallel<C: KernelCache>(
        &mut self,
        alphas: &[f64],
        y: &[f64],
        grad: &[f64],
        c: f64,
        kernel_cache: &mut C,
        active_indices: &[usize],
    ) -> Option<((usize, usize), f64)> {
        const EPS: f64 = 1e-3;
        
        if active_indices.len() < 1000 {
            return self.select_working_set_wss2(alphas, y, grad, c, kernel_cache, active_indices);
        }
        
        // Parallele Berechnung von -y*grad mit collect-update Pattern
        let updates: Vec<(usize, f64, f64)> = active_indices.par_iter().map(|&idx| {
            let g = grad[idx];
            let y_val = y[idx];
            let minus_y_grad = -y_val * g;
            (idx, g, minus_y_grad)
        }).collect();
        
        // Sequenzielles Update der Caches
        for (idx, g, minus_y_grad) in updates {
            self.grad_cache[idx] = g;
            self.minus_y_grad[idx] = minus_y_grad;
        }
        
        // Parallele Aufteilung in I_up und I_low
        let (i_up, i_low): (Vec<_>, Vec<_>) = active_indices
            .par_iter()
            .filter_map(|&idx| {
                let alpha = alphas[idx];
                let y_val = y[idx];
                
                let in_up = (y_val > 0.0 && alpha < c - EPS) || (y_val < 0.0 && alpha > EPS);
                let in_low = (y_val > 0.0 && alpha > EPS) || (y_val < 0.0 && alpha < c - EPS);
                
                match (in_up, in_low) {
                    (true, true) => Some((Some(idx), Some(idx))),
                    (true, false) => Some((Some(idx), None)),
                    (false, true) => Some((None, Some(idx))),
                    _ => None,
                }
            })
            .unzip();
        
        let i_up: Vec<usize> = i_up.into_iter().flatten().collect();
        let i_low: Vec<usize> = i_low.into_iter().flatten().collect();
        
        if i_up.is_empty() || i_low.is_empty() {
            return None;
        }
        
        // Finde optimales i
        let i = i_up.into_par_iter()
            .min_by(|&a, &b| {
                self.minus_y_grad[a].partial_cmp(&self.minus_y_grad[b])
                    .unwrap_or(Ordering::Equal)
            })?;
        
        // Finde optimales j mit sequenzieller Suche (da kernel_cache mutable ist)
        let j = self.select_j_wss2(i, &i_low, kernel_cache)?;
        
        let violation = (self.minus_y_grad[j] - self.minus_y_grad[i]).abs();
        
        if violation < EPS {
            return None;
        }
        
        let i_pos = active_indices.iter().position(|&x| x == i)?;
        let j_pos = active_indices.iter().position(|&x| x == j)?;
        
        Some(((i_pos, j_pos), violation))
    }
}

/// Shrinking-Strategie für WSS2
pub struct WSS2Shrinking {
    active_mask: Vec<bool>,
    grad_diff: AlignedVec<f64>,
    shrink_iter: usize,
    min_active: usize,
}

impl WSS2Shrinking {
    pub fn new(n: usize) -> Self {
        let mut grad_diff = AlignedVec::with_capacity(n);
        grad_diff.resize(n, 0.0);
        
        Self {
            active_mask: vec![true; n],
            grad_diff,
            shrink_iter: 0,
            min_active: (n / 10).max(100),
        }
    }
    
    /// WSS2-kompatibles Shrinking
    pub fn update_active_set_wss2(
        &mut self,
        alphas_mat: MatRef<f64>,
        y_mat: MatRef<f64>,
        grad_mat: MatRef<f64>,
        c: f64,
        tol: f64,
    ) -> Vec<usize> {
        let n = alphas_mat.nrows();
        self.shrink_iter += 1;
        
        // Berechne maximale Gradienten-Differenz für Shrinking
        let (m_pos, m_neg) = self.compute_gradient_bounds(
            alphas_mat, y_mat, grad_mat, c, tol
        );
        
        let threshold = (m_pos - m_neg) * 10.0; // LIBSVM nutzt Faktor 10
        
        // Shrinking-Entscheidung
        let mut active_indices = Vec::with_capacity(n);
        
        for i in 0..n {
            let alpha = alphas_mat[(i, 0)];
            let y_val = y_mat[(i, 0)];
            let grad = grad_mat[(i, 0)];
            let minus_y_grad = -y_val * grad;
            
            self.grad_diff[i] = minus_y_grad;
            
            // WSS2 Shrinking-Kriterium
            let should_shrink = if y_val > 0.0 {
                if alpha > tol && alpha < c - tol {
                    false // Nicht am Rand, nie shrinken
                } else if alpha <= tol {
                    minus_y_grad > m_pos + threshold
                } else {
                    minus_y_grad < m_neg - threshold
                }
            } else {
                if alpha > tol && alpha < c - tol {
                    false
                } else if alpha <= tol {
                    minus_y_grad < m_neg - threshold
                } else {
                    minus_y_grad > m_pos + threshold
                }
            };
            
            self.active_mask[i] = !should_shrink;
            
            if !should_shrink {
                active_indices.push(i);
            }
        }
        
        // Mindestgröße sicherstellen
        if active_indices.len() < self.min_active {
            // Sortiere nach |grad_diff| und füge die wichtigsten hinzu
            let mut all_indices: Vec<(usize, f64)> = (0..n)
                .filter(|&i| !self.active_mask[i])
                .map(|i| (i, self.grad_diff[i].abs()))
                .collect();
                
            all_indices.sort_unstable_by(|a, b| 
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            );
            
            for (idx, _) in all_indices.into_iter().take(self.min_active - active_indices.len()) {
                active_indices.push(idx);
                self.active_mask[idx] = true;
            }
        }
        
        active_indices
    }
    
    fn compute_gradient_bounds(
        &self,
        alphas_mat: MatRef<f64>,
        y_mat: MatRef<f64>,
        grad_mat: MatRef<f64>,
        c: f64,
        tol: f64,
    ) -> (f64, f64) {
        let n = alphas_mat.nrows();
        let mut m_pos = f64::NEG_INFINITY;
        let mut m_neg = f64::INFINITY;
        
        for i in 0..n {
            if !self.active_mask[i] { continue; }
            
            let alpha = alphas_mat[(i, 0)];
            let y_val = y_mat[(i, 0)];
            let grad = grad_mat[(i, 0)];
            let minus_y_grad = -y_val * grad;
            
            if y_val > 0.0 {
                if alpha < c - tol {
                    m_pos = m_pos.max(minus_y_grad);
                }
                if alpha > tol {
                    m_neg = m_neg.min(minus_y_grad);
                }
            } else {
                if alpha < c - tol {
                    m_neg = m_neg.min(minus_y_grad);
                }
                if alpha > tol {
                    m_pos = m_pos.max(minus_y_grad);
                }
            }
        }
        
        (m_pos, m_neg)
    }
    
    pub fn reset(&mut self, _n: usize) {
        self.active_mask.fill(true);
        self.shrink_iter = 0;
    }
}