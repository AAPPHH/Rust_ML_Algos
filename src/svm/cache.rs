use crate::svm::dataset::FlatDataset;
use crate::svm::kernel::KernelType;
use crate::svm::memory::AlignedVec;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use rayon::prelude::*;
use std::hint::black_box;

const CACHE_LINE_SIZE: usize = 64;
const SET_SIZE: usize = 16; // Increased for better hit rate
const PREFETCH_DISTANCE: usize = 4;

pub trait KernelCache: Send + Sync {
    fn get(&mut self, i: usize, j: usize) -> f64;
    fn get_diagonal(&self, i: usize) -> f64;
    fn get_row_batch(&mut self, i: usize, indices: std::ops::Range<usize>, output: &mut [f64]);
    fn get_stats(&self) -> (usize, usize);
    fn prefetch_row(&mut self, i: usize);
}

#[repr(align(64))]
struct CacheLine {
    tag: AtomicU64,
    value: f64,
    _padding: [u8; 48], // Ensure each line is exactly 64 bytes
}

impl CacheLine {
    #[inline(always)]
    fn new() -> Self {
        Self {
            tag: AtomicU64::new(u64::MAX),
            value: 0.0,
            _padding: [0; 48],
        }
    }
}

pub struct SetAssociativeCache {
    kernel: KernelType,
    dataset: FlatDataset,
    sets: Vec<Vec<CacheLine>>,
    kernel_diag: AlignedVec<f64>,
    n_sets: usize,
    n_samples: usize,
    hits: AtomicUsize,
    misses: AtomicUsize,
    row_cache: Vec<AlignedVec<f64>>,
    row_cache_tags: Vec<AtomicUsize>,
}

impl SetAssociativeCache {
    pub fn new(kernel: KernelType, dataset: FlatDataset, size: usize) -> Self {
        let n = dataset.n_samples();
        
        let cache_entries = (size * 1024).max(n * 32).min(128 * 1024 * 1024 / 8);
        let n_sets = (cache_entries / SET_SIZE).next_power_of_two().max(2048);
        
        let sets: Vec<Vec<CacheLine>> = (0..n_sets)
            .into_par_iter()
            .map(|_| (0..SET_SIZE).map(|_| CacheLine::new()).collect())
            .collect();
        
        let mut kernel_diag = AlignedVec::with_capacity(n);
        kernel_diag.resize(n, 0.0);
        
        kernel_diag.as_mut_slice()
            .par_chunks_mut(64)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * 64;
                for (i, val) in chunk.iter_mut().enumerate() {
                    let idx = start + i;
                    if idx < n {
                        let row = dataset.get_row(idx);
                        *val = kernel.compute_self(&row);
                    }
                }
            });

        const ROW_CACHE_SIZE: usize = 32;
        let row_cache = (0..ROW_CACHE_SIZE)
            .map(|_| {
                let mut v = AlignedVec::with_capacity(n);
                v.resize(n, 0.0);
                v
            })
            .collect();
        
        let row_cache_tags = (0..ROW_CACHE_SIZE)
            .map(|_| AtomicUsize::new(usize::MAX))
            .collect();
        
        Self {
            kernel,
            dataset,
            sets,
            kernel_diag,
            n_sets,
            n_samples: n,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            row_cache,
            row_cache_tags,
        }
    }
    
    #[inline(always)]
    fn hash_key(&self, i: usize, j: usize) -> (u64, usize) {
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let key = ((i_min as u64) << 32) | (j_max as u64);
        
        let hash = key.wrapping_mul(0x517cc1b727220a95);
        let set_idx = ((hash >> 32) as usize) & (self.n_sets - 1);
        
        (key, set_idx)
    }
    
    #[inline(always)]
    fn prefetch_cache_line(&self, set_idx: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            let ptr = self.sets.get_unchecked(set_idx).as_ptr() as *const i8;
            _mm_prefetch(ptr, 1);
        }
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
        
        if j + 1 < self.n_samples {
            let (_, next_set) = self.hash_key(i, j + 1);
            self.prefetch_cache_line(next_set);
        }
        
        let set = unsafe { self.sets.get_unchecked(set_idx) };
        
        let tag0 = set[0].tag.load(Ordering::Acquire);
        if tag0 == key {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return set[0].value;
        }
        
        let tag1 = set[1].tag.load(Ordering::Acquire);
        if tag1 == key {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return set[1].value;
        }
        
        for idx in 2..SET_SIZE {
            let tag = unsafe { set.get_unchecked(idx).tag.load(Ordering::Acquire) };
            if tag == key {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return unsafe { set.get_unchecked(idx).value };
            }
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let xi = self.dataset.get_row(i_min);
        let xj = self.dataset.get_row(j_max);
        let value = self.kernel.compute_pair(&xi, &xj);
        
        let victim_idx = (self.misses.load(Ordering::Relaxed) * 0x9e3779b97f4a7c15) as usize % SET_SIZE;
        let victim = unsafe { self.sets.get_unchecked_mut(set_idx).get_unchecked_mut(victim_idx) };
        victim.tag.store(key, Ordering::Release);
        victim.value = value;
        
        value
    }
    
    #[inline(always)]
    fn get_diagonal(&self, i: usize) -> f64 {
        unsafe { *self.kernel_diag.get_unchecked(i) }
    }
    
    fn get_row_batch(&mut self, i: usize, indices: std::ops::Range<usize>, output: &mut [f64]) {
        let batch_size = indices.end - indices.start;
        debug_assert_eq!(output.len(), batch_size);
        
        let cache_idx = i % self.row_cache.len();
        let cached_row_idx = self.row_cache_tags[cache_idx].load(Ordering::Acquire);
        
        if cached_row_idx == i {
            for (idx, j) in indices.enumerate() {
                output[idx] = if i == j {
                    self.kernel_diag[i]
                } else {
                    unsafe { *self.row_cache[cache_idx].get_unchecked(j) }
                };
            }
            self.hits.fetch_add(batch_size, Ordering::Relaxed);
            return;
        }
        
        if batch_size > 64 {
            let xi = self.dataset.get_row(i);
            let row_cache = &mut self.row_cache[cache_idx];
            
            row_cache.as_mut_slice()
                .par_chunks_mut(256)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start = chunk_idx * 256;
                    for (local_idx, val) in chunk.iter_mut().enumerate() {
                        let j = start + local_idx;
                        if j < self.n_samples {
                            if i == j {
                                *val = self.kernel_diag[i];
                            } else {
                                let xj = self.dataset.get_row(j);
                                *val = self.kernel.compute_pair(&xi, &xj);
                            }
                        }
                    }
                });
            
            self.row_cache_tags[cache_idx].store(i, Ordering::Release);
            
            for (idx, j) in indices.enumerate() {
                output[idx] = unsafe { *row_cache.get_unchecked(j) };
            }
            return;
        }
        
        const CHUNK_SIZE: usize = 8;
        let xi = self.dataset.get_row(i);
        
        for (chunk_idx, chunk_start) in indices.clone().step_by(CHUNK_SIZE).enumerate() {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(indices.end);
            let chunk_output = &mut output[chunk_idx * CHUNK_SIZE..(chunk_idx * CHUNK_SIZE + (chunk_end - chunk_start))];
            
            if chunk_end + CHUNK_SIZE < indices.end {
                for j in chunk_end..(chunk_end + CHUNK_SIZE).min(indices.end) {
                    let (_, set_idx) = self.hash_key(i, j);
                    self.prefetch_cache_line(set_idx);
                }
            }
            
            for (local_idx, j) in (chunk_start..chunk_end).enumerate() {
                if i == j {
                    chunk_output[local_idx] = self.kernel_diag[i];
                    self.hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    chunk_output[local_idx] = self.get(i, j);
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
    
    fn prefetch_row(&mut self, i: usize) {
        for j in 0..self.n_samples.min(PREFETCH_DISTANCE) {
            if i != j {
                let (_, set_idx) = self.hash_key(i, j);
                self.prefetch_cache_line(set_idx);
            }
        }
    }
}

unsafe impl Send for SetAssociativeCache {}
unsafe impl Sync for SetAssociativeCache {}