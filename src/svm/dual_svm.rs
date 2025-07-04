use crate::svm::dataset::FlatDataset;
use crate::svm::cache::{KernelCache, SetAssociativeCache};
use crate::svm::kernel::KernelType;
use crate::svm::working_set::{PartialArgMaxSelector, ShrinkingWorkingSet};
use crate::svm::memory::{AlignedBuffer, get_pooled_vec};
use faer::Mat;
use rayon::prelude::*;


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
        let mut grad = vec![-1.0; n];
        for i in 0..n {
            grad[i] *= y[i];
        }
        let mut bias = 0.0;
        
        let cache_size = ((n as f64).sqrt() as usize * 10).max(100).min(8192);
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
        
        while iter < max_iter && (num_changed > 0 || examine_all) {
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
                
                if active_set.len() < n / 20 {
                    active_set = (0..n).collect();
                    examine_all = true;
                    continue;
                }
            }
            
            let indices_to_check = if examine_all {
                (0..n).collect::<Vec<_>>()
            } else {
                active_set.clone()
            };
            
            let mut inner_iter = 0;
            let max_inner = if examine_all { 1 } else { indices_to_check.len().min(1000) };
            
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
                        
                        if self.take_step_optimized(
                            i, j, 
                            &mut alphas, 
                            &mut grad, 
                            &mut bias,
                            &y,
                            &mut kernel_cache,
                        ) {
                            num_changed += 1;
                        }
                    }
                    None => break,
                }
                
                inner_iter += 1;
            }
            
            if iter % 10 == 0 {
                let objective = self.compute_objective(&alphas, &grad);
                convergence_history.push(objective - last_objective);
                last_objective = objective;
                
                if convergence_history.len() > 5 {
                    let recent_avg = convergence_history.iter().rev().take(5).sum::<f64>() / 5.0;
                    if recent_avg.abs() < tol * 0.1 {
                        break;
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
            
            if iter > 10 && num_changed == 0 && !examine_all {
                break;
            }
        }
        
        self.extract_support_vectors_parallel(alphas, y, dataset, bias);
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
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch(alphas.as_ptr().add(i) as *const i8, 0);
            _mm_prefetch(alphas.as_ptr().add(j) as *const i8, 0);
            _mm_prefetch(grad.as_ptr().add(i) as *const i8, 0);
            _mm_prefetch(grad.as_ptr().add(j) as *const i8, 0);
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
        
        if eta <= 0.0 {
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
        
        self.update_gradient_vectorized(
            grad,
            kernel_cache,
            i, j,
            yi_delta_ai,
            yj_delta_aj
        );
        
        true
    }

    #[inline(always)]
    fn update_gradient_vectorized<C: KernelCache>(
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
            if n > 1000 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    self.update_gradient_avx2_fma(
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
        
        const BATCH_SIZE: usize = 256;
        let mut buffer = AlignedBuffer::new(2 * BATCH_SIZE);
        
        for batch_start in (0..n).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(n);
            let batch_len = batch_end - batch_start;
            
            buffer.resize(2 * batch_len);
            let (ki_batch, kj_batch) = buffer.split_at_mut(batch_len);
            
            kernel_cache.get_row_batch(i, batch_start..batch_end, ki_batch);
            kernel_cache.get_row_batch(j, batch_start..batch_end, kj_batch);
            
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
    unsafe fn update_gradient_avx2_fma<C: KernelCache>(
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
        const BATCH_SIZE: usize = 32;
        let mut buffer = AlignedBuffer::new(2 * BATCH_SIZE);
        
        for batch_start in (0..n).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(n);
            let batch_len = batch_end - batch_start;
            
            if batch_end < n {
                _mm_prefetch(grad.as_ptr().add(batch_end) as *const i8, 1);
            }
            
            buffer.resize(2 * batch_len);
            let (ki_vals, kj_vals) = buffer.split_at_mut(batch_len);
            
            kernel_cache.get_row_batch(i, batch_start..batch_end, ki_vals);
            kernel_cache.get_row_batch(j, batch_start..batch_end, kj_vals);
            
            let mut k = 0;
            while k + 8 <= batch_len {
                let grad1 = _mm256_loadu_pd(grad.as_ptr().add(batch_start + k));
                let grad2 = _mm256_loadu_pd(grad.as_ptr().add(batch_start + k + 4));
                
                let ki1 = _mm256_loadu_pd(ki_vals.as_ptr().add(k));
                let ki2 = _mm256_loadu_pd(ki_vals.as_ptr().add(k + 4));
                let kj1 = _mm256_loadu_pd(kj_vals.as_ptr().add(k));
                let kj2 = _mm256_loadu_pd(kj_vals.as_ptr().add(k + 4));
                
                let update1 = _mm256_fmadd_pd(yi_delta_vec, ki1, 
                             _mm256_mul_pd(yj_delta_vec, kj1));
                let update2 = _mm256_fmadd_pd(yi_delta_vec, ki2, 
                             _mm256_mul_pd(yj_delta_vec, kj2));
                
                let new_grad1 = _mm256_add_pd(grad1, update1);
                let new_grad2 = _mm256_add_pd(grad2, update2);

                _mm256_storeu_pd(grad.as_mut_ptr().add(batch_start + k), new_grad1);
                _mm256_storeu_pd(grad.as_mut_ptr().add(batch_start + k + 4), new_grad2);
                
                k += 8;
            }
            
            while k < batch_len {
                let grad_k = grad.get_unchecked_mut(batch_start + k);
                *grad_k += yi_delta_ai * ki_vals.get_unchecked(k) 
                         + yj_delta_aj * kj_vals.get_unchecked(k);
                k += 1;
            }
        }
    }
    
    fn compute_objective(&self, alphas: &[f64], grad: &[f64]) -> f64 {
        alphas.iter().zip(grad.iter())
            .map(|(a, g)| a * g)
            .sum::<f64>() * 0.5
            - alphas.iter().sum::<f64>()
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
        
        let mut sv_data = Mat::zeros(n_sv, dataset.n_features());
        let mut sv_labels = Mat::zeros(n_sv, 1);
        let mut sv_alphas = Mat::zeros(n_sv, 1);
        
        if n_sv > 100 && dataset.n_features() > 100 {
            use std::sync::atomic::{AtomicPtr, Ordering};
            use std::ptr;
            
            let sv_data_ptr = AtomicPtr::new(sv_data.as_ptr() as *mut f64);
            let data_ptr = AtomicPtr::new(dataset.data.as_ptr() as *mut f64);
            let n_features = dataset.n_features();
            
            sv_indices.par_iter().enumerate().for_each(|(idx, &i)| {
                unsafe {
                    let src_offset = i * n_features;
                    let dst_offset = idx * n_features;
                    let src_ptr = data_ptr.load(Ordering::Relaxed).add(src_offset);
                    let dst_ptr = sv_data_ptr.load(Ordering::Relaxed).add(dst_offset);
                    ptr::copy_nonoverlapping(src_ptr, dst_ptr, n_features);
                }
            });
        } else {
            for (idx, &i) in sv_indices.iter().enumerate() {
                let src_row = dataset.data.row(i);
                let mut dst_row = sv_data.row_mut(idx);
                dst_row.copy_from(src_row);
            }
        }

        for (idx, &i) in sv_indices.iter().enumerate() {
            sv_labels[(idx, 0)] = y[i];
            sv_alphas[(idx, 0)] = alphas[i];
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
            KernelType::Linear => self.decision_function_linear(dataset, sv, sl, al, bias),
            _ => self.decision_function_nonlinear(dataset, sv, sl, al, bias),
        }
    }
    
    fn decision_function_linear(
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

        let w: Vec<f64> = if n_sv > 100 {
            use std::sync::Mutex;
            let w_mutex = Mutex::new(vec![0.0; n_features]);
            
            (0..n_sv).into_par_iter().chunks(64).for_each(|chunk| {
                let mut local_w = get_pooled_vec(n_features);
                local_w.resize(n_features, 0.0);
                
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
            
            w_mutex.into_inner().unwrap()
        } else {
            let mut w = vec![0.0; n_features];
            for i in 0..n_sv {
                let alpha_y = al[(i, 0)] * sl[(i, 0)];
                let sv_row = sv.data.row(i);
                
                for j in 0..n_features {
                    w[j] += alpha_y * sv_row[j];
                }
            }
            w
        };
        
        if n_samples > 1000 {
            (0..n_samples).into_par_iter()
                .map(|i| {
                    let row = dataset.get_row(i);
                    let mut sum = 0.0;
                    
                    for j in 0..n_features {
                        sum += row[j] * w[j];
                    }
                    
                    sum + bias
                })
                .collect()
        } else {
            let mut result = vec![0.0; n_samples];
            for i in 0..n_samples {
                let row = dataset.get_row(i);
                let mut sum = 0.0;
                
                for j in 0..n_features {
                    sum += row[j] * w[j];
                }
                
                result[i] = sum + bias;
            }
            result
        }
    }
    
    fn decision_function_nonlinear(
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
        
        if n > 100 {
            (0..n).into_par_iter()
                .chunks(32)
                .flat_map(|chunk| {
                    chunk.into_iter().map(|i| {
                        let xi = dataset.get_row(i);
                        let mut sum = 0.0;
                        
                        for j in 0..n_sv {
                            let svj = sv.get_row(j);
                            sum += coeffs[j] * self.kernel.compute_pair(&xi, &svj);
                        }
                        
                        sum + bias
                    }).collect::<Vec<_>>()
                })
                .collect()
        } else {
            let mut result = vec![0.0; n];
            
            for i in 0..n {
                let xi = dataset.get_row(i);
                let mut sum = 0.0;
                
                for j in 0..n_sv {
                    let svj = sv.get_row(j);
                    sum += coeffs[j] * self.kernel.compute_pair(&xi, &svj);
                }
                
                result[i] = sum + bias;
            }
            
            result
        }
    }
}