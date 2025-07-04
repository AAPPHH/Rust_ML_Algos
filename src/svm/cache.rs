use crate::svm::dataset::FlatDataset;
use crate::svm::kernel::KernelType;
use crate::svm::memory::AlignedVec;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use rayon::prelude::*;

const CACHE_LINE_SIZE: usize = 64;
const SET_SIZE: usize = 8;

pub trait KernelCache: Send + Sync {
    fn get(&mut self, i: usize, j: usize) -> f64;
    fn get_diagonal(&self, i: usize) -> f64;
    fn get_row_batch(&mut self, i: usize, indices: std::ops::Range<usize>, output: &mut [f64]);
    fn get_stats(&self) -> (usize, usize);
}

#[repr(align(64))]
struct CacheLine {
    tag: AtomicU64,
    value: f64,
    access_count: AtomicUsize,
}

impl CacheLine {
    fn new() -> Self {
        Self {
            tag: AtomicU64::new(u64::MAX),
            value: 0.0,
            access_count: AtomicUsize::new(0),
        }
    }
}

pub struct SetAssociativeCache {
    kernel: KernelType,
    dataset: FlatDataset,
    sets: Vec<Vec<CacheLine>>,
    kernel_diag: AlignedVec<f64>,
    n_sets: usize,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl SetAssociativeCache {
    pub fn new(kernel: KernelType, dataset: FlatDataset, size: usize) -> Self {
        let n = dataset.n_samples();
        
        let cache_entries = size.max(n * n / 8).min(8 * 1024 * 1024);
        let n_sets = (cache_entries / SET_SIZE).max(1024);
        
        let sets: Vec<Vec<CacheLine>> = (0..n_sets)
            .map(|_| (0..SET_SIZE).map(|_| CacheLine::new()).collect())
            .collect();
        
        let mut kernel_diag = AlignedVec::with_capacity(n);
        kernel_diag.resize(n, 0.0);
        
        kernel_diag.as_mut_slice()
            .par_chunks_mut(256)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * 256;
                for (i, val) in chunk.iter_mut().enumerate() {
                    let idx = start + i;
                    if idx < n {
                        let row = dataset.get_row(idx);
                        *val = kernel.compute_self(&row);
                    }
                }
            });
        
        Self {
            kernel,
            dataset,
            sets,
            kernel_diag,
            n_sets,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }
    
    #[inline(always)]
    fn hash_key(&self, i: usize, j: usize) -> (u64, usize) {
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let key = ((i_min as u64) << 32) | (j_max as u64);
        
        let hash = key.wrapping_mul(0x9e3779b97f4a7c15);
        let set_idx = ((hash >> 32) as usize) % self.n_sets;
        
        (key, set_idx)
    }
}

impl KernelCache for SetAssociativeCache {
    #[inline(always)]
    fn get(&mut self, i: usize, j: usize) -> f64 {
        if i == j {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return unsafe { *self.kernel_diag.get_unchecked(i) };
        }
        
        let (key, set_idx) = self.hash_key(i, j);
        let set = unsafe { self.sets.get_unchecked(set_idx) };
        
        let mut min_access = usize::MAX;
        let mut victim_idx = 0;
        
        for (idx, line) in set.iter().enumerate() {
            let tag = line.tag.load(Ordering::Acquire);
            if tag == key {
                line.access_count.fetch_add(1, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return line.value;
            }
            
            let access = line.access_count.load(Ordering::Relaxed);
            if access < min_access {
                min_access = access;
                victim_idx = idx;
            }
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let xi = self.dataset.get_row(i_min);
        let xj = self.dataset.get_row(j_max);
        let value = self.kernel.compute_pair(&xi, &xj);
        
        let victim = unsafe { self.sets.get_unchecked_mut(set_idx).get_unchecked_mut(victim_idx) };
        victim.tag.store(key, Ordering::Release);
        victim.value = value;
        victim.access_count.store(1, Ordering::Relaxed);
        
        value
    }
    
    #[inline(always)]
    fn get_diagonal(&self, i: usize) -> f64 {
        unsafe { *self.kernel_diag.get_unchecked(i) }
    }
    
    fn get_row_batch(&mut self, i: usize, indices: std::ops::Range<usize>, output: &mut [f64]) {
        let batch_size = indices.end - indices.start;
        debug_assert_eq!(output.len(), batch_size);
        
        if batch_size <= 32 {
            for (idx, j) in indices.enumerate() {
                output[idx] = self.get(i, j);
            }
            return;
        }
        
        const CHUNK_SIZE: usize = 16;
        let xi = self.dataset.get_row(i);
        
        for (chunk_idx, chunk_start) in indices.clone().step_by(CHUNK_SIZE).enumerate() {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(indices.end);
            let chunk_output = &mut output[chunk_idx * CHUNK_SIZE..(chunk_idx * CHUNK_SIZE + (chunk_end - chunk_start))];
            
            if chunk_end < indices.end {
                for j in chunk_end..(chunk_end + CHUNK_SIZE).min(indices.end) {
                    let (_, set_idx) = self.hash_key(i, j);
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    unsafe {
                        use std::arch::x86_64::_mm_prefetch;
                        let ptr = self.sets.get_unchecked(set_idx).as_ptr() as *const i8;
                        _mm_prefetch(ptr, 1);
                    }
                }
            }
            
            for (local_idx, j) in (chunk_start..chunk_end).enumerate() {
                if i == j {
                    chunk_output[local_idx] = self.kernel_diag[i];
                    self.hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    let (key, set_idx) = self.hash_key(i, j);
                    let set = unsafe { self.sets.get_unchecked(set_idx) };
                    
                    let mut found = false;
                    for line in set.iter() {
                        if line.tag.load(Ordering::Acquire) == key {
                            chunk_output[local_idx] = line.value;
                            line.access_count.fetch_add(1, Ordering::Relaxed);
                            self.hits.fetch_add(1, Ordering::Relaxed);
                            found = true;
                            break;
                        }
                    }
                    
                    if !found {
                        self.misses.fetch_add(1, Ordering::Relaxed);
                        let xj = self.dataset.get_row(j);
                        chunk_output[local_idx] = self.kernel.compute_pair(&xi, &xj);
                        
                        let mut min_access = usize::MAX;
                        let mut victim_idx = 0;
                        
                        for (idx, line) in set.iter().enumerate() {
                            let access = line.access_count.load(Ordering::Relaxed);
                            if access < min_access {
                                min_access = access;
                                victim_idx = idx;
                            }
                        }
                        
                        let victim = unsafe { self.sets.get_unchecked_mut(set_idx).get_unchecked_mut(victim_idx) };
                        victim.tag.store(key, Ordering::Release);
                        victim.value = chunk_output[local_idx];
                        victim.access_count.store(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }
    
    fn get_stats(&self) -> (usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed)
        )
    }
}

unsafe impl Send for SetAssociativeCache {}
unsafe impl Sync for SetAssociativeCache {}