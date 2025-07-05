use crate::svm::dataset::FlatDataset;
use crate::svm::cache::{KernelCache, SetAssociativeCache};
use crate::svm::kernel::KernelType;
use crate::svm::working_set::{PartialArgMaxSelector, ShrinkingWorkingSet};
use crate::svm::memory::{AlignedBuffer, get_pooled_vec};
use faer::{Mat, MatRef, col::ColRef};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct DualSVM {
    pub alphas: Option<Mat<f64>>,
    pub support_vectors: Option<FlatDataset>,
    pub support_labels: Option<Mat<f64>>,
    pub bias: f64,
    pub c: f64,
    pub kernel: KernelType,
}

impl Clone for DualSVM {
    fn clone(&self) -> Self {
        Self {
            alphas: self.alphas.clone(),
            support_vectors: self.support_vectors.clone(),
            support_labels: self.support_labels.clone(),
            bias: self.bias,
            c: self.c,
            kernel: self.kernel.clone(),
        }
    }
}

impl DualSVM {
    pub fn new(kernel: KernelType, c: f64) -> Self {
        Self {
            alphas: None,
            support_vectors: None,
            support_labels: None,
            bias: 0.0,
            c,
            kernel,
        }
    }

    pub fn fit(&mut self, dataset: FlatDataset, y: Vec<f64>, max_iter: usize, tol: f64) {
        let n = dataset.n_samples();
        
        let mut alphas = vec![0.0; n];
        let mut grad = vec![0.0; n];
        
        // Initialize gradient
        for i in 0..n {
            grad[i] = -y[i];
        }
        
        let mut bias = 0.0;
        
        let cache_size = self.compute_optimal_cache_size(n);
        let mut kernel_cache = SetAssociativeCache::new(self.kernel.clone(), dataset.clone(), cache_size);
        
        let mut ws_selector = PartialArgMaxSelector::new(n);
        let mut shrinking_ws = ShrinkingWorkingSet::new(n);
        
        let mut active_set: Vec<usize> = (0..n).collect();
        let mut iter = 0;
        let mut shrink_counter = 0;
        let mut num_changed = 0;
        let mut examine_all = true;
        
        let mut convergence_history = Vec::with_capacity(100);
        let mut last_objective = f64::NEG_INFINITY;
        let mut stuck_counter = 0;
        
        let early_stop = AtomicBool::new(false);
        
        while iter < max_iter && (num_changed > 0 || examine_all) && !early_stop.load(Ordering::Relaxed) {
            num_changed = 0;
            
            if !examine_all && iter > 10 && shrink_counter % 10 == 0 {
                let alphas_mat = Mat::from_fn(n, 1, |i, _| alphas[i]);
                let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
                let grad_mat = Mat::from_fn(n, 1, |i, _| grad[i]);
                
                active_set = shrinking_ws.update_active_set(
                    alphas_mat.as_ref(),
                    y_mat.as_ref(),
                    grad_mat.as_ref(),
                    self.c,
                    tol,
                );
                
                if active_set.len() < n / 20 || stuck_counter > 5 {
                    active_set = (0..n).collect();
                    examine_all = true;
                    stuck_counter = 0;
                    continue;
                }
            }
            
            let indices_to_check = if examine_all {
                (0..n).collect::<Vec<_>>()
            } else {
                active_set.clone()
            };
            
            let max_inner = if examine_all { 
                1 
            } else { 
                (indices_to_check.len() as f64 * 0.1).max(10.0).min(1000.0) as usize
            };
            
            let mut inner_iter = 0;
            let mut local_changes = 0;
            
            while inner_iter < max_inner {
                let ws_result = ws_selector.select_working_set_optimized(
                    &alphas,
                    &y,
                    &grad,
                    self.c,
                    &mut kernel_cache,
                    &indices_to_check,
                );
                
                match ws_result {
                    Some(((ii, jj), violation)) => {
                        if violation < tol {
                            break;
                        }
                        
                        let i = indices_to_check[ii];
                        let j = indices_to_check[jj];
                        
                        if inner_iter + 1 < max_inner {
                            kernel_cache.prefetch_row(i);
                            kernel_cache.prefetch_row(j);
                        }
                        
                        if self.take_step_optimized(
                            i, j, 
                            &mut alphas, 
                            &mut grad, 
                            &mut bias,
                            &y,
                            &mut kernel_cache,
                        ) {
                            local_changes += 1;
                        }
                    }
                    None => break,
                }
                
                inner_iter += 1;
            }
            
            num_changed += local_changes;
            
            if iter % 10 == 0 {
                let objective = self.compute_objective_fast(&alphas, &grad);
                let improvement = objective - last_objective;
                convergence_history.push(improvement);
                
                if improvement < tol * 0.01 {
                    stuck_counter += 1;
                } else {
                    stuck_counter = 0;
                }
                
                last_objective = objective;
                
                if convergence_history.len() > 5 {
                    let recent_avg = convergence_history.iter()
                        .rev()
                        .take(5)
                        .sum::<f64>() / 5.0;
                    
                    if recent_avg.abs() < tol * 0.01 {
                        early_stop.store(true, Ordering::Relaxed);
                    }
                }
            }
            
            if examine_all {
                examine_all = false;
            } else if num_changed == 0 {
                examine_all = true;
            }
            
            iter += 1;
            shrink_counter += 1;
            
            if iter > 50 && num_changed == 0 && !examine_all && stuck_counter > 3 {
                break;
            }
        }
        
        self.extract_support_vectors_parallel(alphas, y, dataset, bias);
    }

    fn compute_optimal_cache_size(&self, n: usize) -> usize {

        let available_mb = 1024;
        
        let entry_size = 16;
        let max_entries = (available_mb * 1024 * 1024) / entry_size;
        
        let baseline = ((n as f64).sqrt() * 20.0) as usize;
        
        baseline.min(max_entries).max(1024)
    }

    #[inline(always)]
    fn take_step_optimized<C: KernelCache>(
        &mut self,
        i: usize,
        j: usize,
        alphas: &mut [f64],
        grad: &mut [f64],
        bias: &mut f64,
        y: &[f64],
        kernel_cache: &mut C,
    ) -> bool {
        if i == j {
            return false;
        }
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(alphas.as_ptr().add(i) as *const i8, _MM_HINT_T0);
            _mm_prefetch(alphas.as_ptr().add(j) as *const i8, _MM_HINT_T0);
            _mm_prefetch(grad.as_ptr().add(i) as *const i8, _MM_HINT_T0);
            _mm_prefetch(grad.as_ptr().add(j) as *const i8, _MM_HINT_T0);
        }
        
        let yi = unsafe { *y.get_unchecked(i) };
        let yj = unsafe { *y.get_unchecked(j) };
        let ai_old = unsafe { *alphas.get_unchecked(i) };
        let aj_old = unsafe { *alphas.get_unchecked(j) };
        let s = yi * yj;
        
        let (l, h) = if s < 0.0 {
            let diff = aj_old - ai_old;
            (diff.max(0.0), (self.c + diff).min(self.c))
        } else {
            let sum = ai_old + aj_old;
            ((sum - self.c).max(0.0), sum.min(self.c))
        };
        
        if (l - h).abs() < 1e-12 {
            return false;
        }
        
        let kii = kernel_cache.get_diagonal(i);
        let kjj = kernel_cache.get_diagonal(j);
        let kij = kernel_cache.get(i, j);
        let eta = kii + kjj - 2.0 * kij;
        
        if eta <= 1e-12 {
            return false;
        }
        
        let ei = unsafe { -*grad.get_unchecked(i) };
        let ej = unsafe { -*grad.get_unchecked(j) };
        let mut aj_new = aj_old + yj * (ei - ej) / eta;
        
        aj_new = l.max(aj_new.min(h));
        
        let delta = aj_new - aj_old;
        if delta.abs() < 1e-6 * (aj_new + aj_old + 1e-6) {
            return false;
        }
        
        let ai_new = ai_old + s * delta;
        let delta_ai = ai_new - ai_old;
        let delta_aj = delta;
        
        let b1 = *bias + ei + yi * delta_ai * kii + yj * delta_aj * kij;
        let b2 = *bias + ej + yi * delta_ai * kij + yj * delta_aj * kjj;
        
        *bias = if ai_new > 1e-8 && ai_new < self.c - 1e-8 {
            b1
        } else if aj_new > 1e-8 && aj_new < self.c - 1e-8 {
            b2
        } else {
            (b1 + b2) * 0.5
        };
        
        unsafe {
            *alphas.get_unchecked_mut(i) = ai_new;
            *alphas.get_unchecked_mut(j) = aj_new;
        }
        
        let yi_delta_ai = yi * delta_ai;
        let yj_delta_aj = yj * delta_aj;
        
        self.update_gradient_vectorized_optimized(
            grad,
            kernel_cache,
            i, j,
            yi_delta_ai,
            yj_delta_aj
        );
        
        true
    }

    #[inline(always)]
    fn update_gradient_vectorized_optimized<C: KernelCache>(
        &self,
        grad: &mut [f64],
        kernel_cache: &mut C,
        i: usize,
        j: usize,
        yi_delta_ai: f64,
        yj_delta_aj: f64,
    ) {
        let n = grad.len();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n > 500 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    self.update_gradient_avx2_fma_optimized(
                        grad,
                        kernel_cache,
                        i, j,
                        yi_delta_ai,
                        yj_delta_aj
                    );
                    return;
                }
            }
        }
        
        const BATCH_SIZE: usize = 512;
        let mut buffer = AlignedBuffer::new(2 * BATCH_SIZE);
        
        for batch_start in (0..n).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(n);
            let batch_len = batch_end - batch_start;
            
            buffer.resize(2 * batch_len);
            let (ki_batch, kj_batch) = buffer.split_at_mut(batch_len);
            
            if batch_len > 64 {
                kernel_cache.prefetch_row(i);
                kernel_cache.get_row_batch(i, batch_start..batch_end, ki_batch);
                kernel_cache.prefetch_row(j);
                kernel_cache.get_row_batch(j, batch_start..batch_end, kj_batch);
            } else {
                kernel_cache.get_row_batch(i, batch_start..batch_end, ki_batch);
                kernel_cache.get_row_batch(j, batch_start..batch_end, kj_batch);
            }
            
            for (idx, k) in (batch_start..batch_end).enumerate() {
                unsafe {
                    let grad_k = grad.get_unchecked_mut(k);
                    *grad_k += yi_delta_ai * ki_batch.get_unchecked(idx) 
                             + yj_delta_aj * kj_batch.get_unchecked(idx);
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn update_gradient_avx2_fma_optimized<C: KernelCache>(
        &self,
        grad: &mut [f64],
        kernel_cache: &mut C,
        i: usize,
        j: usize,
        yi_delta_ai: f64,
        yj_delta_aj: f64,
    ) {
        use std::arch::x86_64::*;
        
        let yi_delta_vec = _mm256_set1_pd(yi_delta_ai);
        let yj_delta_vec = _mm256_set1_pd(yj_delta_aj);
        
        let n = grad.len();
        const BATCH_SIZE: usize = 64;
        let mut buffer = AlignedBuffer::new(2 * BATCH_SIZE);
        
        for batch_start in (0..n).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(n);
            let batch_len = batch_end - batch_start;
            
            if batch_end < n {
                _mm_prefetch(grad.as_ptr().add(batch_end) as *const i8, _MM_HINT_T1);
            }
            
            buffer.resize(2 * batch_len);
            let (ki_vals, kj_vals) = buffer.split_at_mut(batch_len);
            
            kernel_cache.get_row_batch(i, batch_start..batch_end, ki_vals);
            kernel_cache.get_row_batch(j, batch_start..batch_end, kj_vals);
            
            let mut k = 0;
            
            while k + 16 <= batch_len {
                let grad0 = _mm256_loadu_pd(grad.as_ptr().add(batch_start + k));
                let grad1 = _mm256_loadu_pd(grad.as_ptr().add(batch_start + k + 4));
                let grad2 = _mm256_loadu_pd(grad.as_ptr().add(batch_start + k + 8));
                let grad3 = _mm256_loadu_pd(grad.as_ptr().add(batch_start + k + 12));
                
                let ki0 = _mm256_loadu_pd(ki_vals.as_ptr().add(k));
                let ki1 = _mm256_loadu_pd(ki_vals.as_ptr().add(k + 4));
                let ki2 = _mm256_loadu_pd(ki_vals.as_ptr().add(k + 8));
                let ki3 = _mm256_loadu_pd(ki_vals.as_ptr().add(k + 12));
                
                let kj0 = _mm256_loadu_pd(kj_vals.as_ptr().add(k));
                let kj1 = _mm256_loadu_pd(kj_vals.as_ptr().add(k + 4));
                let kj2 = _mm256_loadu_pd(kj_vals.as_ptr().add(k + 8));
                let kj3 = _mm256_loadu_pd(kj_vals.as_ptr().add(k + 12));
                
                let update0 = _mm256_fmadd_pd(yi_delta_vec, ki0, 
                             _mm256_mul_pd(yj_delta_vec, kj0));
                let update1 = _mm256_fmadd_pd(yi_delta_vec, ki1, 
                             _mm256_mul_pd(yj_delta_vec, kj1));
                let update2 = _mm256_fmadd_pd(yi_delta_vec, ki2, 
                             _mm256_mul_pd(yj_delta_vec, kj2));
                let update3 = _mm256_fmadd_pd(yi_delta_vec, ki3, 
                             _mm256_mul_pd(yj_delta_vec, kj3));
                
                let new_grad0 = _mm256_add_pd(grad0, update0);
                let new_grad1 = _mm256_add_pd(grad1, update1);
                let new_grad2 = _mm256_add_pd(grad2, update2);
                let new_grad3 = _mm256_add_pd(grad3, update3);

                _mm256_storeu_pd(grad.as_mut_ptr().add(batch_start + k), new_grad0);
                _mm256_storeu_pd(grad.as_mut_ptr().add(batch_start + k + 4), new_grad1);
                _mm256_storeu_pd(grad.as_mut_ptr().add(batch_start + k + 8), new_grad2);
                _mm256_storeu_pd(grad.as_mut_ptr().add(batch_start + k + 12), new_grad3);
                
                k += 16;
            }
            
            while k < batch_len {
                let grad_k = grad.get_unchecked_mut(batch_start + k);
                *grad_k += yi_delta_ai * ki_vals.get_unchecked(k) 
                         + yj_delta_aj * kj_vals.get_unchecked(k);
                k += 1;
            }
        }
    }
    
    #[inline]
    fn compute_objective_fast(&self, alphas: &[f64], grad: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut c = 0.0;
        
        for i in 0..alphas.len() {
            let y = alphas[i] * grad[i] * 0.5 - alphas[i] - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        sum
    }
    
    fn extract_support_vectors_parallel(
        &mut self,
        alphas: Vec<f64>,
        y: Vec<f64>,
        dataset: FlatDataset,
        bias: f64,
    ) {
        let sv_indices: Vec<usize> = (0..alphas.len())
            .into_par_iter()
            .filter(|&i| alphas[i] > 1e-8)
            .collect();
        
        let n_sv = sv_indices.len();
        if n_sv == 0 {
            return;
        }
        
        let n_features = dataset.n_features();
        
        let mut sv_data = Mat::<f64>::zeros(n_sv, n_features);
        let mut sv_labels = Mat::<f64>::zeros(n_sv, 1);
        let mut sv_alphas = Mat::<f64>::zeros(n_sv, 1);
        
        if n_sv > 100 && n_features > 100 {
            let rows_data: Vec<(usize, Vec<f64>)> = sv_indices
                .par_iter()
                .enumerate()
                .map(|(idx, &i)| {
                    let src_row = dataset.data.row(i);
                    let row_vec: Vec<f64> = (0..n_features).map(|j| src_row[j]).collect();
                    (idx, row_vec)
                })
                .collect();

            for (idx, row_data) in rows_data {
                for (j, &val) in row_data.iter().enumerate() {
                    sv_data[(idx, j)] = val;
                }
            }
            
            for (idx, &i) in sv_indices.iter().enumerate() {
                sv_labels[(idx, 0)] = y[i];
                sv_alphas[(idx, 0)] = alphas[i];
            }
        } else {
            for (idx, &i) in sv_indices.iter().enumerate() {
                let src_row = dataset.data.row(i);
                let mut dst_row = sv_data.row_mut(idx);
                dst_row.copy_from(src_row);
                sv_labels[(idx, 0)] = y[i];
                sv_alphas[(idx, 0)] = alphas[i];
            }
        }

        self.support_vectors = Some(FlatDataset { data: sv_data });
        self.support_labels = Some(sv_labels);
        self.alphas = Some(sv_alphas);
        self.bias = bias;
    }

    pub fn decision_function_batch(&self, dataset: &FlatDataset) -> Vec<f64> {
        let sv = self.support_vectors.as_ref().unwrap();
        let sl = self.support_labels.as_ref().unwrap();
        let al = self.alphas.as_ref().unwrap();
        let bias = self.bias;
        
        match &self.kernel {
            KernelType::Linear => self.decision_function_linear_optimized(dataset, sv, sl, al, bias),
            _ => self.decision_function_nonlinear_optimized(dataset, sv, sl, al, bias),
        }
    }
    
    fn decision_function_linear_optimized(
        &self,
        dataset: &FlatDataset,
        sv: &FlatDataset,
        sl: &Mat<f64>,
        al: &Mat<f64>,
        bias: f64,
    ) -> Vec<f64> {
        let n_features = sv.n_features();
        let n_sv = sv.n_samples();
        let n_samples = dataset.n_samples();

        let mut w = vec![0.0; n_features];
        
        if n_sv > 50 {
            use std::sync::Mutex;
            let w_mutex = Mutex::new(vec![0.0; n_features]);
            
            (0..n_sv).into_par_iter().chunks(128).for_each(|chunk| {
                let mut local_w = vec![0.0; n_features];
                
                for i in chunk {
                    let alpha_y = al[(i, 0)] * sl[(i, 0)];
                    let sv_row = sv.data.row(i);
                    
                    for j in 0..n_features {
                        local_w[j] += alpha_y * sv_row[j];
                    }
                }
                
                let mut w_global = w_mutex.lock().unwrap();
                for j in 0..n_features {
                    w_global[j] += local_w[j];
                }
            });
            
            w = w_mutex.into_inner().unwrap();
        } else {
            for i in 0..n_sv {
                let alpha_y = al[(i, 0)] * sl[(i, 0)];
                let sv_row = sv.data.row(i);
                
                for j in 0..n_features {
                    w[j] += alpha_y * sv_row[j];
                }
            }
        }
        
        if n_samples > 500 {
            (0..n_samples).into_par_iter()
                .map(|i| {
                    let row = dataset.get_row(i);
                    let mut sum = bias;
                    
                    let mut j = 0;
                    while j + 4 <= n_features {
                        sum += row[j] * w[j] + row[j+1] * w[j+1] +
                               row[j+2] * w[j+2] + row[j+3] * w[j+3];
                        j += 4;
                    }
                    while j < n_features {
                        sum += row[j] * w[j];
                        j += 1;
                    }
                    
                    sum
                })
                .collect()
        } else {
            let mut result = vec![0.0; n_samples];
            for i in 0..n_samples {
                let row = dataset.get_row(i);
                let mut sum = bias;
                
                for j in 0..n_features {
                    sum += row[j] * w[j];
                }
                
                result[i] = sum;
            }
            result
        }
    }
    
    fn decision_function_nonlinear_optimized(
        &self,
        dataset: &FlatDataset,
        sv: &FlatDataset,
        sl: &Mat<f64>,
        al: &Mat<f64>,
        bias: f64,
    ) -> Vec<f64> {
        let n = dataset.n_samples();
        let n_sv = sv.n_samples();
        
        let coeffs: Vec<f64> = (0..n_sv)
            .map(|i| al[(i, 0)] * sl[(i, 0)])
            .collect();
        
        if n > 200 {
            let chunk_size = (n / rayon::current_num_threads()).max(16);
            
            (0..n).into_par_iter()
                .chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.into_iter().map(|i| {
                        let xi = dataset.get_row(i);
                        let mut sum = bias;
                        
                        let mut j = 0;
                        while j + 2 <= n_sv {
                            let svj0 = sv.get_row(j);
                            let svj1 = sv.get_row(j + 1);
                            
                            sum += coeffs[j] * self.kernel.compute_pair(&xi, &svj0);
                            sum += coeffs[j + 1] * self.kernel.compute_pair(&xi, &svj1);
                            
                            j += 2;
                        }
                        
                        while j < n_sv {
                            let svj = sv.get_row(j);
                            sum += coeffs[j] * self.kernel.compute_pair(&xi, &svj);
                            j += 1;
                        }
                        
                        sum
                    }).collect::<Vec<_>>()
                })
                .collect()
        } else {
            let mut result = vec![0.0; n];
            
            for i in 0..n {
                let xi = dataset.get_row(i);
                let mut sum = bias;
                
                for j in 0..n_sv {
                    let svj = sv.get_row(j);
                    sum += coeffs[j] * self.kernel.compute_pair(&xi, &svj);
                }
                
                result[i] = sum;
            }
            
            result
        }
    }
}