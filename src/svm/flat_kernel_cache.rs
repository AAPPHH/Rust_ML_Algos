use crate::svm::flat_dataset::FlatDataset;
use crate::svm::svm_kernel::KernelType;
use faer::Mat;
use lru::LruCache;
use std::num::NonZeroUsize;

#[derive(Clone)]
pub struct FlatKernelCache {
    kernel: KernelType,
    dataset: FlatDataset,
    cache: LruCache<u64, f64>,
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
            cache: LruCache::new(NonZeroUsize::new(size.max(1)).unwrap()),
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
        let key = ((i_min as u64) << 32) | (j_max as u64);
        
        if let Some(&val) = self.cache.get(&key) {
            self.hits += 1;
            return val;
        }
        
        self.misses += 1;
        
        let val = {
            let xi = self.dataset.get_row(i_min);
            let xj = self.dataset.get_row(j_max);
            self.kernel.compute_pair_row(&xi, &xj)
        };
        
        self.cache.put(key, val);
        
        val
    }
    
    #[inline(always)]
    pub fn get_diagonal(&self, i: usize) -> f64 {
        unsafe { *self.kernel_diag.get_unchecked(i) }
    }
    
    pub fn compute_kernel_rows(&mut self, indices: &[usize], target: usize) -> Mat<f64> {
        let mut result = Mat::zeros(indices.len(), 1);
        
        let mut idx = 0;
        while idx + 8 <= indices.len() {
            if idx + 8 < indices.len() {
                std::hint::black_box(&indices[idx + 8]);
            }
            
            result[(idx, 0)] = self.get(indices[idx], target);
            result[(idx + 1, 0)] = self.get(indices[idx + 1], target);
            result[(idx + 2, 0)] = self.get(indices[idx + 2], target);
            result[(idx + 3, 0)] = self.get(indices[idx + 3], target);
            result[(idx + 4, 0)] = self.get(indices[idx + 4], target);
            result[(idx + 5, 0)] = self.get(indices[idx + 5], target);
            result[(idx + 6, 0)] = self.get(indices[idx + 6], target);
            result[(idx + 7, 0)] = self.get(indices[idx + 7], target);
            idx += 8;
        }
        
        // Rest
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