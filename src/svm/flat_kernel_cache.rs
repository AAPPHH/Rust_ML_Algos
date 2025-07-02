use crate::svm::flat_dataset::FlatDataset;
use crate::svm::svm_kernel::KernelType;
use faer::Mat;
use std::collections::HashMap;

#[derive(Clone)]
pub struct FlatKernelCache {
    kernel: KernelType,
    dataset: FlatDataset,
    cache: HashMap<u32, f64>,
    kernel_diag: Vec<f64>,
    hits: usize,
    misses: usize,
}

impl FlatKernelCache {
    pub fn new(kernel: KernelType, dataset: FlatDataset, size: usize) -> Self {
        let n = dataset.n_samples();
        
        let mut kernel_diag = vec![0.0; n];
        for i in 0..n {
            let row = dataset.get_row(i);
            kernel_diag[i] = kernel.compute_pair_row(&row, &row);
        }
        
        FlatKernelCache { 
            kernel, 
            dataset, 
            cache: HashMap::with_capacity(size),
            kernel_diag,
            hits: 0,
            misses: 0,
        }
    }

    #[inline(always)]
    pub fn get(&mut self, i: usize, j: usize) -> f64 {
        if i == j {
            self.hits += 1;
            return unsafe { *self.kernel_diag.get_unchecked(i) };
        }
        
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let key = ((i_min as u32) << 16) | (j_max as u32);
        
        if let Some(&val) = self.cache.get(&key) {
            self.hits += 1;
            return val;
        }
        
        self.misses += 1;
        
        let val = if i_min < self.dataset.n_samples() && j_max < self.dataset.n_samples() {
            let xi = self.dataset.get_row(i_min);
            let xj = self.dataset.get_row(j_max);
            self.kernel.compute_pair_row(&xi, &xj)
        } else {
            0.0
        };
        
        if self.cache.len() < self.cache.capacity() {
            self.cache.insert(key, val);
        }
        
        val
    }
    
    #[inline(always)]
    pub fn get_diagonal(&self, i: usize) -> f64 {
        unsafe { *self.kernel_diag.get_unchecked(i) }
    }
    
    pub fn compute_kernel_rows(&mut self, indices: &[usize], target: usize) -> Mat<f64> {
        let mut result = Mat::zeros(indices.len(), 1);
        
        let mut idx = 0;
        while idx + 4 <= indices.len() {
            result[(idx, 0)] = self.get(indices[idx], target);
            result[(idx + 1, 0)] = self.get(indices[idx + 1], target);
            result[(idx + 2, 0)] = self.get(indices[idx + 2], target);
            result[(idx + 3, 0)] = self.get(indices[idx + 3], target);
            idx += 4;
        }
        
        while idx < indices.len() {
            result[(idx, 0)] = self.get(indices[idx], target);
            idx += 1;
        }
        
        result
    }
    
    pub fn get_stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }
}