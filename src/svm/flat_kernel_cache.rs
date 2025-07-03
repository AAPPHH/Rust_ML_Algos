use crate::svm::flat_dataset::FlatDataset;
use crate::svm::svm_kernel::KernelType;
use faer::Mat;
use lru::LruCache;
use std::num::NonZeroUsize;
use rayon::prelude::*;

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
        let available_bytes = 8 * 1024 * 1024 * 1024_usize;
        let bytes_per_entry = 16;
        let max_entries = available_bytes / bytes_per_entry;
        let optimal_size = (n * n / 4).min(max_entries).max(size);
        
        let mut kernel_diag = vec![0.0; n];
        kernel_diag.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * 256;
            for (i, val) in chunk.iter_mut().enumerate() {
                let idx = start + i;
                if idx < n {
                    let row = dataset.get_row(idx);
                    *val = kernel.compute_pair_row(&row, &row);
                }
            }
        });
        
        FlatKernelCache { 
            kernel, 
            dataset, 
            cache: LruCache::new(NonZeroUsize::new(optimal_size).unwrap()),
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
        
        const PREFETCH_DISTANCE: usize = 8;
        
        let mut idx = 0;
        while idx < indices.len() {
            if idx + PREFETCH_DISTANCE < indices.len() {
                for k in 0..PREFETCH_DISTANCE.min(indices.len() - idx - PREFETCH_DISTANCE) {
                    std::hint::black_box(&indices[idx + PREFETCH_DISTANCE + k]);
                }
            }
            
            let batch_end = (idx + 8).min(indices.len());
            for i in idx..batch_end {
                result[(i, 0)] = self.get(indices[i], target);
            }
            
            idx = batch_end;
        }
        
        result
    }
    
    pub fn get_stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }
}