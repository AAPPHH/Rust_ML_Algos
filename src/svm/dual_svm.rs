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
        
        let yi = y[i];
        let yj = y[j];
        let ai_old = alphas[i];
        let aj_old = alphas[j];
        let s = yi * yj;
        
        let (l, h) = if s < 0.0 {
            let diff = aj_old - ai_old;
            (0f64.max(diff), self.c.min(self.c + diff))
        } else {
            let sum = ai_old + aj_old;
            (0f64.max(sum - self.c), self.c.min(sum))
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
        
        let ei = -grad[i];
        let ej = -grad[j];
        let mut aj_new = aj_old + yj * (ei - ej) / eta;
        
        aj_new = if aj_new < l { l } else if aj_new > h { h } else { aj_new };
        
        if (aj_new - aj_old).abs() < 1e-6 * (aj_new + aj_old + 1e-6) {
            return false;
        }
        
        let ai_new = ai_old + s * (aj_old - aj_new);
        
        let delta_ai = ai_new - ai_old;
        let delta_aj = aj_new - aj_old;
        
        let b1 = *bias + ei + yi * delta_ai * kii + yj * delta_aj * kij;
        let b2 = *bias + ej + yi * delta_ai * kij + yj * delta_aj * kjj;
        
        *bias = if ai_new > 1e-8 && ai_new < self.c - 1e-8 {
            b1
        } else if aj_new > 1e-8 && aj_new < self.c - 1e-8 {
            b2
        } else {
            (b1 + b2) / 2.0
        };
        
        alphas[i] = ai_new;
        alphas[j] = aj_new;
        
        let yi_delta_ai = yi * delta_ai;
        let yj_delta_aj = yj * delta_aj;
        
        if grad.len() > 1000 {
            let batch_size = 64;
            for batch_start in (0..grad.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(grad.len());
                for k in batch_start..batch_end {
                    let ki = kernel_cache.get(i, k);
                    let kj = kernel_cache.get(j, k);
                    grad[k] += yi_delta_ai * ki + yj_delta_aj * kj;
                }
            }
        } else {
            for k in 0..grad.len() {
                let ki = kernel_cache.get(i, k);
                let kj = kernel_cache.get(j, k);
                grad[k] += yi_delta_ai * ki + yj_delta_aj * kj;
            }
        }
        
        true
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

    pub fn decision_function_batch(&self, dataset: &FlatDataset) -> Vec<f64> {
        let sv = self.support_vectors.as_ref().unwrap();
        let sl = self.support_labels.as_ref().unwrap();
        let al = self.alphas.as_ref().unwrap();
        let bias = self.bias;
        
        match &self.kernel {
            KernelType::Linear => {
                let mut w = Mat::<f64>::zeros(sv.n_features(), 1);

                for i in 0..sv.n_samples() {
                    let alpha_y = al[(i, 0)] * sl[(i, 0)];
                    for j in 0..sv.n_features() {
                        w[(j, 0)] += alpha_y * sv.data[(i, j)];
                    }
                }
                
                let result_mat = dataset.as_ref() * &w;
                (0..result_mat.nrows())
                    .map(|i| result_mat[(i, 0)] + bias)
                    .collect()
            }
            _ => {
                let n = dataset.n_samples();
                let n_sv = sv.n_samples();
                let mut result = vec![0.0; n];
                
                let coeffs: Vec<f64> = (0..n_sv)
                    .map(|i| al[(i, 0)] * sl[(i, 0)])
                    .collect();
                
                let batch_size = 32;
                for batch_start in (0..n).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(n);
                    
                    for i in batch_start..batch_end {
                        let xi = dataset.get_row(i);
                        let mut sum = 0.0;
                        
                        for j in 0..n_sv {
                            let svj = sv.get_row(j);
                            sum += coeffs[j] * self.kernel.compute_pair_row(&xi, &svj);
                        }
                        
                        result[i] = sum + bias;
                    }
                }
                
                result
            }
        }
    }
}