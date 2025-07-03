use crate::svm::flat_dataset::FlatDataset;
use crate::svm::flat_kernel_cache::FlatKernelCache;
use crate::svm::svm_kernel::KernelType;
use crate::svm::working_set::{WorkingSetSelector, ShrinkingWorkingSet};
use faer::Mat;

#[derive(Clone)]
pub struct DualSVM {
    pub alphas: Option<Mat<f64>>,
    pub support_vectors: Option<FlatDataset>,
    pub support_labels: Option<Mat<f64>>,
    pub bias: f64,
    pub c: f64,
    pub kernel: KernelType,
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
        for i in 0..n {
            grad[i] = -y[i];
        }
        let mut bias = 0.0;
        
        let cache_size = (n / 4).max(100).min(1024);
        let mut kernel_cache = FlatKernelCache::new(self.kernel.clone(), dataset.clone(), cache_size);
        
        let mut ws_selector = WorkingSetSelector::new(n);
        let mut shrinking_ws = ShrinkingWorkingSet::new(n);
        
        let mut active_set: Vec<usize> = (0..n).collect();
        let mut iter = 0;
        let mut shrink_counter = 0;
        let mut num_changed = 0;
        let mut examine_all = true;
        
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
                let ws_result = ws_selector.select_working_set(
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
                        
                        if self.take_step(
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
        
        self.extract_support_vectors_optimized(alphas, y, dataset, bias);
    }

    #[inline(always)]
    fn take_step(
        &self,
        i: usize,
        j: usize,
        alphas: &mut [f64],
        grad: &mut [f64],
        bias: &mut f64,
        y: &[f64],
        kernel_cache: &mut FlatKernelCache,
    ) -> bool {
        if i == j {
            return false;
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
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if grad.len() > 1000 && is_x86_feature_detected!("avx2") {
                unsafe {
                    Self::update_gradient_avx2(
                        grad,
                        kernel_cache,
                        i, j,
                        yi_delta_ai,
                        yj_delta_aj
                    );
                }
            } else {
                Self::update_gradient_scalar(
                    grad,
                    kernel_cache,
                    i, j,
                    yi_delta_ai,
                    yj_delta_aj
                );
            }
        }
        
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self::update_gradient_scalar(
                grad,
                kernel_cache,
                i, j,
                yi_delta_ai,
                yj_delta_aj
            );
        }
        
        true
    }

    #[inline(always)]
    fn update_gradient_scalar(
        grad: &mut [f64],
        kernel_cache: &mut FlatKernelCache,
        i: usize,
        j: usize,
        yi_delta_ai: f64,
        yj_delta_aj: f64,
    ) {
        let batch_size = 64;
        let n = grad.len();
        
        for batch_start in (0..n).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n);
            
            // Prefetch nächste Batch
            if batch_end < n {
                std::hint::black_box(&grad[batch_end]);
            }
            
            for k in batch_start..batch_end {
                let ki = kernel_cache.get(i, k);
                let kj = kernel_cache.get(j, k);
                unsafe {
                    let grad_k = grad.get_unchecked_mut(k);
                    *grad_k += yi_delta_ai * ki + yj_delta_aj * kj;
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn update_gradient_avx2(
        grad: &mut [f64],
        kernel_cache: &mut FlatKernelCache,
        i: usize,
        j: usize,
        yi_delta_ai: f64,
        yj_delta_aj: f64,
    ) {
        use std::arch::x86_64::*;
        
        let yi_delta_vec = _mm256_set1_pd(yi_delta_ai);
        let yj_delta_vec = _mm256_set1_pd(yj_delta_aj);
        
        let mut k = 0;
        let n = grad.len();
        
        while k + 4 <= n {
            if k + 16 < n {
                _mm_prefetch(grad.as_ptr().add(k + 16) as *const i8, _MM_HINT_T0);
            }
            
            let grad_vec = _mm256_loadu_pd(grad.as_ptr().add(k));
            
            let ki_vals = [
                kernel_cache.get(i, k),
                kernel_cache.get(i, k + 1),
                kernel_cache.get(i, k + 2),
                kernel_cache.get(i, k + 3),
            ];
            let kj_vals = [
                kernel_cache.get(j, k),
                kernel_cache.get(j, k + 1),
                kernel_cache.get(j, k + 2),
                kernel_cache.get(j, k + 3),
            ];
            
            let ki_vec = _mm256_loadu_pd(ki_vals.as_ptr());
            let kj_vec = _mm256_loadu_pd(kj_vals.as_ptr());
            
            let update1 = _mm256_mul_pd(yi_delta_vec, ki_vec);
            let update2 = _mm256_mul_pd(yj_delta_vec, kj_vec);
            let update = _mm256_add_pd(update1, update2);
            let new_grad = _mm256_add_pd(grad_vec, update);
            
            _mm256_storeu_pd(grad.as_mut_ptr().add(k), new_grad);
            
            k += 4;
        }
        
        while k < n {
            let ki = kernel_cache.get(i, k);
            let kj = kernel_cache.get(j, k);
            let grad_k = grad.get_unchecked_mut(k);
            *grad_k += yi_delta_ai * ki + yj_delta_aj * kj;
            k += 1;
        }
    }
    
    fn extract_support_vectors_optimized(
        &mut self,
        alphas: Vec<f64>,
        y: Vec<f64>,
        dataset: FlatDataset,
        bias: f64,
    ) {
        let sv_indices: Vec<usize> = alphas.iter()
            .enumerate()
            .filter(|(_, &a)| a > 1e-8)
            .map(|(i, _)| i)
            .collect();
        
        let n_sv = sv_indices.len();
        if n_sv == 0 {
            return;
        }
        
        let mut sv_data = Mat::zeros(n_sv, dataset.n_features());
        let mut sv_labels = Mat::zeros(n_sv, 1);
        let mut sv_alphas = Mat::zeros(n_sv, 1);
        
        for (idx, &i) in sv_indices.iter().enumerate() {
            let src_row = dataset.data.row(i);
            let mut dst_row = sv_data.row_mut(idx);
            dst_row.copy_from(src_row);
            
            sv_labels[(idx, 0)] = y[i];
            sv_alphas[(idx, 0)] = alphas[i];
        }
        
        self.support_vectors = Some(FlatDataset { data: sv_data });
        self.support_labels = Some(sv_labels);
        self.alphas = Some(sv_alphas);
        self.bias = bias;
    }

// Ersetze die decision_function_batch in dual_svm.rs mit dieser Version

// Ersetze die decision_function_batch in dual_svm.rs mit dieser Version

pub fn decision_function_batch(&self, dataset: &FlatDataset) -> Vec<f64> {
    let sv = self.support_vectors.as_ref().unwrap();
    let sl = self.support_labels.as_ref().unwrap();
    let al = self.alphas.as_ref().unwrap();
    let bias = self.bias;
    
    match &self.kernel {
        KernelType::Linear => {
            // Optimierte lineare Kernel Implementation
            let n_features = sv.n_features();
            let n_sv = sv.n_samples();
            
            // Berechne w einmalig mit optimierter Loop
            let mut w = vec![0.0; n_features];
            
            // Unroll loop für bessere Performance
            for i in 0..n_sv {
                let alpha_y = al[(i, 0)] * sl[(i, 0)];
                let sv_row = sv.data.row(i);
                
                // Vektorisierte Addition
                let mut j = 0;
                while j + 8 <= n_features {
                    unsafe {
                        *w.get_unchecked_mut(j) += alpha_y * sv_row[j];
                        *w.get_unchecked_mut(j + 1) += alpha_y * sv_row[j + 1];
                        *w.get_unchecked_mut(j + 2) += alpha_y * sv_row[j + 2];
                        *w.get_unchecked_mut(j + 3) += alpha_y * sv_row[j + 3];
                        *w.get_unchecked_mut(j + 4) += alpha_y * sv_row[j + 4];
                        *w.get_unchecked_mut(j + 5) += alpha_y * sv_row[j + 5];
                        *w.get_unchecked_mut(j + 6) += alpha_y * sv_row[j + 6];
                        *w.get_unchecked_mut(j + 7) += alpha_y * sv_row[j + 7];
                    }
                    j += 8;
                }
                
                // Handle remaining
                while j < n_features {
                    unsafe {
                        *w.get_unchecked_mut(j) += alpha_y * sv_row[j];
                    }
                    j += 1;
                }
            }
            
            // Optimierte Matrix-Vektor Multiplikation
            let n_samples = dataset.n_samples();
            let mut result = vec![0.0; n_samples];
            
            // Parallele Verarbeitung für große Datasets
            if n_samples > 1000 {
                use rayon::prelude::*;
                result.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, res)| {
                        let row = dataset.get_row(i);
                        let mut sum = 0.0;
                        
                        // Vektorisierte Dot-Product
                        let mut j = 0;
                        while j + 4 <= n_features {
                            sum += row[j] * w[j] + 
                                   row[j+1] * w[j+1] + 
                                   row[j+2] * w[j+2] + 
                                   row[j+3] * w[j+3];
                            j += 4;
                        }
                        
                        while j < n_features {
                            sum += row[j] * w[j];
                            j += 1;
                        }
                        
                        *res = sum + bias;
                    });
            } else {
                // Sequenzielle Version für kleine Datasets
                for i in 0..n_samples {
                    let row = dataset.get_row(i);
                    let mut sum = 0.0;
                    
                    for j in 0..n_features {
                        sum += row[j] * w[j];
                    }
                    
                    result[i] = sum + bias;
                }
            }
            
            result
        }
        _ => {
            // Optimierte nicht-lineare Kernel Implementation
            let n = dataset.n_samples();
            let n_sv = sv.n_samples();
            
            // Vorberechnung der Koeffizienten
            let mut coeffs = vec![0.0; n_sv];
            for i in 0..n_sv {
                coeffs[i] = al[(i, 0)] * sl[(i, 0)];
            }
            
            // Parallele Verarbeitung für große Datasets
            if n > 100 {
                use rayon::prelude::*;
                
                (0..n).into_par_iter()
                    .map(|i| {
                        let xi = dataset.get_row(i);
                        let mut sum = 0.0;
                        
                        // Cache-freundliche Iteration
                        for j in 0..n_sv {
                            let svj = sv.get_row(j);
                            sum += coeffs[j] * self.kernel.compute_pair_row(&xi, &svj);
                        }
                        
                        sum + bias
                    })
                    .collect()
            } else {
                // Sequenzielle Version mit Batch-Processing
                let mut result = vec![0.0; n];
                let batch_size = 32;
                
                for batch_start in (0..n).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(n);
                    
                    for i in batch_start..batch_end {
                        let xi = dataset.get_row(i);
                        let mut sum = 0.0;
                        
                        // Unroll inner loop für bessere Performance
                        let mut j = 0;
                        while j + 4 <= n_sv {
                            let sv0 = sv.get_row(j);
                            let sv1 = sv.get_row(j + 1);
                            let sv2 = sv.get_row(j + 2);
                            let sv3 = sv.get_row(j + 3);
                            
                            sum += coeffs[j] * self.kernel.compute_pair_row(&xi, &sv0);
                            sum += coeffs[j + 1] * self.kernel.compute_pair_row(&xi, &sv1);
                            sum += coeffs[j + 2] * self.kernel.compute_pair_row(&xi, &sv2);
                            sum += coeffs[j + 3] * self.kernel.compute_pair_row(&xi, &sv3);
                            
                            j += 4;
                        }
                        
                        while j < n_sv {
                            let svj = sv.get_row(j);
                            sum += coeffs[j] * self.kernel.compute_pair_row(&xi, &svj);
                            j += 1;
                        }
                        
                        result[i] = sum + bias;
                    }
                }
                
                result
            }
        }
    }
}
}