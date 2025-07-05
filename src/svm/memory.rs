use std::cell::RefCell;
use std::mem;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

const CACHE_LINE_SIZE: usize = 64;

#[repr(align(64))]
pub struct AlignedVec<T> {
    data: Vec<T>,
}

impl<T> AlignedVec<T> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self { 
            data: Vec::with_capacity(capacity) 
        }
    }
    
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) 
    where T: Clone {
        self.data.resize(new_len, value);
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    #[inline]
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }
    
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[repr(align(64))]
pub struct AlignedBuffer {
    ptr: *mut f64,
    capacity: usize,
    len: usize,
}

impl AlignedBuffer {
    pub fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(
            capacity * mem::size_of::<f64>(),
            CACHE_LINE_SIZE
        ).unwrap();
        
        let ptr = unsafe { alloc(layout) as *mut f64 };
        
        Self {
            ptr,
            capacity,
            len: 0,
        }
    }
    
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        assert!(new_len <= self.capacity, "Buffer capacity exceeded");
        self.len = new_len;

        if new_len > self.len {
            unsafe {
                ptr::write_bytes(
                    self.ptr.add(self.len),
                    0,
                    (new_len - self.len) * mem::size_of::<f64>()
                );
            }
        }
    }
    
    #[inline]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [f64], &mut [f64]) {
        let slice = self.as_mut_slice();
        slice.split_at_mut(mid)
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(
            self.capacity * mem::size_of::<f64>(),
            CACHE_LINE_SIZE
        ).unwrap();
        
        unsafe {
            dealloc(self.ptr as *mut u8, layout);
        }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut new_buffer = Self::new(self.capacity);
        new_buffer.resize(self.len);
        unsafe {
            ptr::copy_nonoverlapping(self.ptr, new_buffer.ptr, self.len);
        }
        new_buffer
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

pub struct MemoryPool {
    pools: RefCell<Vec<PoolBucket>>,
}

struct PoolBucket {
    size: usize,
    free_list: Vec<Vec<f64>>,
}

impl MemoryPool {
    pub fn new() -> Self {
        let bucket_sizes = vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let pools = bucket_sizes.into_iter()
            .map(|size| PoolBucket {
                size,
                free_list: Vec::with_capacity(16),
            })
            .collect();
            
        Self {
            pools: RefCell::new(pools),
        }
    }
    
    pub fn get_vec(&self, capacity: usize) -> PooledVec {
        let mut pools = self.pools.borrow_mut();
        
        let bucket_idx = pools.iter()
            .position(|b| b.size >= capacity)
            .unwrap_or(pools.len() - 1);
        
        let vec = if let Some(bucket) = pools.get_mut(bucket_idx) {
            if let Some(mut vec) = bucket.free_list.pop() {
                vec.clear();
                vec.reserve(capacity);
                vec
            } else {
                Vec::with_capacity(bucket.size.max(capacity))
            }
        } else {
            Vec::with_capacity(capacity)
        };
        
        PooledVec {
            vec,
            pool: self as *const MemoryPool,
            bucket_idx,
        }
    }
}

pub struct PooledVec {
    vec: Vec<f64>,
    pool: *const MemoryPool,
    bucket_idx: usize,
}

impl PooledVec {
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: f64) {
        self.vec.resize(new_len, value);
    }
    
    #[inline]
    pub fn len(&self) -> usize {
        self.vec.len()
    }
}

impl std::ops::Deref for PooledVec {
    type Target = [f64];
    
    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl std::ops::DerefMut for PooledVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl Drop for PooledVec {
    fn drop(&mut self) {
        unsafe {
            let pool = &*self.pool;
            let mut pools = pool.pools.borrow_mut();
            
            if let Some(bucket) = pools.get_mut(self.bucket_idx) {
                if bucket.free_list.len() < 32 {
                    let vec = std::mem::take(&mut self.vec);
                    bucket.free_list.push(vec);
                }
            }
        }
    }
}

thread_local! {
    pub static THREAD_POOL: MemoryPool = MemoryPool::new();
}

#[inline]
pub fn get_pooled_vec(capacity: usize) -> PooledVec {
    THREAD_POOL.with(|pool| pool.get_vec(capacity))
}