use crate::svm::dataset::FlatDataset;
use crate::svm::kernel::KernelType;
use crate::svm::memory::AlignedVec;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use rayon::prelude::*;

const CACHE_LINE_SIZE: usize = 64;
const SET_SIZE: usize = 8;
const PREFETCH_DISTANCE: usize = 8;
const ROW_CACHE_SIZE: usize = 64;

pub trait KernelCache: Send + Sync {
    fn get(&mut self, i: usize, j: usize) -> f64;
    fn get_diagonal(&self, i: usize) -> f64;
    fn get_row_batch(&mut self, i: usize, indices: std::ops::Range<usize>, output: &mut [f64]);
    fn get_stats(&self) -> (usize, usize);
    fn prefetch_row(&mut self, i: usize);
    fn hash_key(&self, i: usize, j: usize) -> (u64, usize);
    fn prefetch_cache_line(&self, set_idx: usize);
}

#[repr(align(64))]
struct CacheLine {
    tag: u64,
    value: f64,
    lru_counter: u8,
    _padding: [u8; 47],
}

impl CacheLine {
    #[inline(always)]
    fn new() -> Self {
        Self {
            tag: u64::MAX,
            value: 0.0,
            lru_counter: 0,
            _padding: [0; 47],
        }
    }
}

pub struct SetAssociativeCache<'a> {
    kernel: KernelType,
    dataset: FlatDataset<'a>,
    sets: Vec<Vec<CacheLine>>,
    kernel_diag: AlignedVec<f64>,
    n_sets: usize,
    n_samples: usize,
    hits: AtomicUsize,
    misses: AtomicUsize,
    row_cache: Vec<AlignedVec<f64>>,
    row_cache_tags: Vec<AtomicUsize>,
    row_cache_counters: Vec<AtomicUsize>,
    lru_counter: AtomicU64,
}

impl<'a> SetAssociativeCache<'a> {
    pub fn new(kernel: KernelType, dataset: FlatDataset<'a>, size: usize) -> Self {
        let n = dataset.n_samples();
        
        let cache_entries = (size * 1024 * 1024 / 8).max(n * 16).min(256 * 1024 * 1024 / 8);
        let n_sets = ((cache_entries / SET_SIZE) as u64).next_power_of_two() as usize;
        
        let sets: Vec<Vec<CacheLine>> = (0..n_sets)
            .into_par_iter()
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
            
        let row_cache_counters = (0..ROW_CACHE_SIZE)
            .map(|_| AtomicUsize::new(0))
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
            row_cache_counters,
            lru_counter: AtomicU64::new(0),
        }
    }
    
    #[inline(always)]
    fn hash_key(&self, i: usize, j: usize) -> (u64, usize) {
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let key = ((i_min as u64) << 32) | (j_max as u64);
        
        let mut hash = 0xcbf29ce484222325u64;
        hash ^= key;
        hash = hash.wrapping_mul(0x100000001b3);
        
        let set_idx = (hash as usize) & (self.n_sets - 1);
        (key, set_idx)
    }
    
    #[inline(always)]
    fn prefetch_cache_line(&self, set_idx: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let ptr = self.sets.get_unchecked(set_idx).as_ptr() as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }
    
    #[inline]
    fn should_cache_row(&self, i: usize) -> bool {
        let access_count = self.lru_counter.fetch_add(1, Ordering::Relaxed);
        (access_count % 10) == 0 || i < 100
    }
}

impl<'a> KernelCache for SetAssociativeCache<'a> {
    #[inline(always)]
    fn get(&mut self, i: usize, j: usize) -> f64 {
        if i == j {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return unsafe { *self.kernel_diag.get_unchecked(i) };
        }
        
        let (key, set_idx) = self.hash_key(i, j);
        
        for k in 1..=PREFETCH_DISTANCE {
            if j + k < self.n_samples {
                let (_, next_set) = self.hash_key(i, j + k);
                self.prefetch_cache_line(next_set);
            }
        }
        
        let set = unsafe { self.sets.get_unchecked_mut(set_idx) };
        
        for line in set.iter_mut() {
            if line.tag == key {
                self.hits.fetch_add(1, Ordering::Relaxed);
                line.lru_counter = 255;
                return line.value;
            }
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let xi = self.dataset.get_row(i_min);
        let xj = self.dataset.get_row(j_max);
        let value = self.kernel.compute_pair(&xi, &xj);
        
        let mut min_lru = 255u8;
        let mut victim_idx = 0;
        for (idx, line) in set.iter_mut().enumerate() {
            if line.lru_counter < min_lru {
                min_lru = line.lru_counter;
                victim_idx = idx;
            }
            line.lru_counter = line.lru_counter.saturating_sub(1);
        }
        
        let victim = unsafe { set.get_unchecked_mut(victim_idx) };
        victim.tag = key;
        victim.value = value;
        victim.lru_counter = 255;
        
        value
    }
    
    #[inline(always)]
    fn get_diagonal(&self, i: usize) -> f64 {
        unsafe { *self.kernel_diag.get_unchecked(i) }
    }
    
    fn get_row_batch(&mut self, i: usize, indices: std::ops::Range<usize>, output: &mut [f64]) {
        let batch_size = indices.end - indices.start;
        debug_assert_eq!(output.len(), batch_size);
        
        let mut cache_hit = false;
        let mut cache_idx = 0;
        
        for idx in 0..ROW_CACHE_SIZE {
            if self.row_cache_tags[idx].load(Ordering::Acquire) == i {
                cache_hit = true;
                cache_idx = idx;
                self.row_cache_counters[idx].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
        
        if cache_hit {
            let row_data = &self.row_cache[cache_idx];
            for (idx, j) in indices.enumerate() {
                output[idx] = if i == j {
                    self.kernel_diag[i]
                } else {
                    unsafe { *row_data.get_unchecked(j) }
                };
            }
            self.hits.fetch_add(batch_size, Ordering::Relaxed);
            return;
        }
        
        if batch_size > 32 || self.should_cache_row(i) {
            let xi = self.dataset.get_row(i);
            
            let mut min_count = usize::MAX;
            let mut victim_idx = 0;
            
            for idx in 0..ROW_CACHE_SIZE {
                let count = self.row_cache_counters[idx].load(Ordering::Relaxed);
                if count < min_count {
                    min_count = count;
                    victim_idx = idx;
                }
            }
            
            let row_cache = &mut self.row_cache[victim_idx];
            
            if self.n_samples > 1000 {
                row_cache.as_mut_slice()
                    .par_chunks_mut(512)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let start = chunk_idx * 512;
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
            } else {
                for j in 0..self.n_samples {
                    row_cache[j] = if i == j {
                        self.kernel_diag[i]
                    } else {
                        let xj = self.dataset.get_row(j);
                        self.kernel.compute_pair(&xi, &xj)
                    };
                }
            }
            
            self.row_cache_tags[victim_idx].store(i, Ordering::Release);
            self.row_cache_counters[victim_idx].store(1, Ordering::Release);
            
            for (idx, j) in indices.enumerate() {
                output[idx] = unsafe { *row_cache.get_unchecked(j) };
            }
            return;
        }
        
        const UNROLL: usize = 4;
        let mut idx = 0;
        
        while idx + UNROLL <= batch_size {
            let j0 = indices.start + idx;
            let j1 = indices.start + idx + 1;
            let j2 = indices.start + idx + 2;
            let j3 = indices.start + idx + 3;
            
            output[idx] = if i == j0 { self.kernel_diag[i] } else { self.get(i, j0) };
            output[idx + 1] = if i == j1 { self.kernel_diag[i] } else { self.get(i, j1) };
            output[idx + 2] = if i == j2 { self.kernel_diag[i] } else { self.get(i, j2) };
            output[idx + 3] = if i == j3 { self.kernel_diag[i] } else { self.get(i, j3) };
            
            idx += UNROLL;
        }
        
        while idx < batch_size {
            let j = indices.start + idx;
            output[idx] = if i == j {
                self.kernel_diag[i]
            } else {
                self.get(i, j)
            };
            idx += 1;
        }
    }
    
    fn get_stats(&self) -> (usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed)
        )
    }
    
    fn prefetch_row(&mut self, i: usize) {
        for j in 0..self.n_samples.min(PREFETCH_DISTANCE * 2) {
            if i != j {
                let (_, set_idx) = self.hash_key(i, j);
                self.prefetch_cache_line(set_idx);
            }
        }
    }
    
    #[inline(always)]
    fn hash_key(&self, i: usize, j: usize) -> (u64, usize) {
        let (i_min, j_max) = if i < j { (i, j) } else { (j, i) };
        let key = ((i_min as u64) << 32) | (j_max as u64);
        
        let mut hash = 0xcbf29ce484222325u64;
        hash ^= key;
        hash = hash.wrapping_mul(0x100000001b3);
        
        let set_idx = (hash as usize) & (self.n_sets - 1);
        (key, set_idx)
    }
    
    #[inline(always)]
    fn prefetch_cache_line(&self, set_idx: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let ptr = self.sets.get_unchecked(set_idx).as_ptr() as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }
}

unsafe impl<'a> Send for SetAssociativeCache<'a> {}
unsafe impl<'a> Sync for SetAssociativeCache<'a> {}