use crate::svm::dataset::DatasetRowRef;

#[derive(Clone)]
pub enum KernelType {
    Linear,
    RBF { gamma: f64, neg_gamma: f64 },
    Poly { degree: u32, coef0: f64, gamma: f64 },
}

impl KernelType {
    pub fn new_rbf(gamma: f64) -> Self {
        KernelType::RBF { gamma, neg_gamma: -gamma }
    }
    
    #[inline(always)]
    pub fn compute_pair(&self, x: &DatasetRowRef<'_>, y: &DatasetRowRef<'_>) -> f64 {
        match self {
            KernelType::Linear => self.compute_linear_zero_copy(x, y),
            KernelType::RBF { neg_gamma, .. } => {
                let dist_sq = self.squared_distance_zero_copy(x, y);
                (*neg_gamma * dist_sq).exp()
            }
            KernelType::Poly { degree, coef0, gamma } => {
                let dot = self.compute_linear_zero_copy(x, y);
                (gamma * dot + *coef0).powi(*degree as i32)
            }
        }
    }
    
    #[inline(always)]
    pub fn compute_self(&self, x: &DatasetRowRef<'_>) -> f64 {
        match self {
            KernelType::Linear => self.compute_linear_zero_copy(x, x),
            KernelType::RBF { .. } => 1.0, // RBF(x,x) = 1
            KernelType::Poly { degree, coef0, gamma } => {
                let dot = self.compute_linear_zero_copy(x, x);
                (gamma * dot + *coef0).powi(*degree as i32)
            }
        }
    }
    
    #[inline(always)]
    fn compute_linear_zero_copy(&self, x: &DatasetRowRef<'_>, y: &DatasetRowRef<'_>) -> f64 {
        let n = x.ncols();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n >= 32 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.dot_product_avx2_fma_zero_copy(x, y) };
            }
        }
        
        let mut sum = 0.0;
        let mut i = 0;
        
        // Unrolled loop für bessere Performance
        while i + 8 <= n {
            sum += x[i] * y[i] + x[i+1] * y[i+1] + 
                   x[i+2] * y[i+2] + x[i+3] * y[i+3] +
                   x[i+4] * y[i+4] + x[i+5] * y[i+5] +
                   x[i+6] * y[i+6] + x[i+7] * y[i+7];
            i += 8;
        }
        
        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }
        
        sum
    }
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_avx2_fma_zero_copy(&self, x: &DatasetRowRef<'_>, y: &DatasetRowRef<'_>) -> f64 {
        use std::arch::x86_64::*;
        
        let n = x.ncols();
        
        // Für Zero-Copy müssen wir mit den Strides arbeiten
        match (x, y) {
            (DatasetRowRef::Raw { ptr: x_ptr, stride: x_stride, .. }, 
             DatasetRowRef::Raw { ptr: y_ptr, stride: y_stride, .. }) => {
                
                if *x_stride == 1 && *y_stride == 1 {
                    // Kontinuierliche Daten - verwende optimierten Pfad
                    self.dot_product_avx2_continuous(*x_ptr, *y_ptr, n)
                } else {
                    // Nicht-kontinuierlich - Fallback
                    self.dot_product_strided(*x_ptr, *x_stride, *y_ptr, *y_stride, n)
                }
            },
            _ => {
                // Fallback für andere Kombinationen
                let mut sum = 0.0;
                for i in 0..n {
                    sum += x[i] * y[i];
                }
                sum
            }
        }
    }
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_avx2_continuous(&self, x_ptr: *const f64, y_ptr: *const f64, n: usize) -> f64 {
        use std::arch::x86_64::*;
        
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();
        let mut i = 0;
        
        while i + 16 <= n {
            let x0 = _mm256_loadu_pd(x_ptr.add(i));
            let y0 = _mm256_loadu_pd(y_ptr.add(i));
            let x1 = _mm256_loadu_pd(x_ptr.add(i + 4));
            let y1 = _mm256_loadu_pd(y_ptr.add(i + 4));
            let x2 = _mm256_loadu_pd(x_ptr.add(i + 8));
            let y2 = _mm256_loadu_pd(y_ptr.add(i + 8));
            let x3 = _mm256_loadu_pd(x_ptr.add(i + 12));
            let y3 = _mm256_loadu_pd(y_ptr.add(i + 12));
            
            sum0 = _mm256_fmadd_pd(x0, y0, sum0);
            sum1 = _mm256_fmadd_pd(x1, y1, sum1);
            sum2 = _mm256_fmadd_pd(x2, y2, sum2);
            sum3 = _mm256_fmadd_pd(x3, y3, sum3);
            
            i += 16;
        }
        
        sum0 = _mm256_add_pd(sum0, sum1);
        sum2 = _mm256_add_pd(sum2, sum3);
        sum0 = _mm256_add_pd(sum0, sum2);
        
        let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum0);
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        while i < n {
            result += *x_ptr.add(i) * *y_ptr.add(i);
            i += 1;
        }
        
        result
    }
    
    #[inline]
    unsafe fn dot_product_strided(&self, x_ptr: *const f64, x_stride: isize, 
                                   y_ptr: *const f64, y_stride: isize, n: usize) -> f64 {
        let mut sum = 0.0;
        for i in 0..n {
            sum += *x_ptr.offset(i as isize * x_stride) * *y_ptr.offset(i as isize * y_stride);
        }
        sum
    }
    
    #[inline(always)]
    fn squared_distance_zero_copy(&self, x: &DatasetRowRef<'_>, y: &DatasetRowRef<'_>) -> f64 {
        let n = x.ncols();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n >= 32 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.squared_distance_avx2_zero_copy(x, y) };
            }
        }
        
        let mut sum = 0.0;
        let mut i = 0;
        
        while i + 8 <= n {
            let d0 = x[i] - y[i];
            let d1 = x[i+1] - y[i+1];
            let d2 = x[i+2] - y[i+2];
            let d3 = x[i+3] - y[i+3];
            let d4 = x[i+4] - y[i+4];
            let d5 = x[i+5] - y[i+5];
            let d6 = x[i+6] - y[i+6];
            let d7 = x[i+7] - y[i+7];
            
            sum += (d0*d0 + d1*d1) + (d2*d2 + d3*d3) + 
                   (d4*d4 + d5*d5) + (d6*d6 + d7*d7);
            i += 8;
        }
        
        while i < n {
            let d = x[i] - y[i];
            sum += d * d;
            i += 1;
        }
        
        sum
    }
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn squared_distance_avx2_zero_copy(&self, x: &DatasetRowRef<'_>, y: &DatasetRowRef<'_>) -> f64 {
        use std::arch::x86_64::*;
        
        let n = x.ncols();
        
        match (x, y) {
            (DatasetRowRef::Raw { ptr: x_ptr, stride: x_stride, .. }, 
             DatasetRowRef::Raw { ptr: y_ptr, stride: y_stride, .. }) => {
                
                if *x_stride == 1 && *y_stride == 1 {
                    // Kontinuierliche Daten
                    let mut sum0 = _mm256_setzero_pd();
                    let mut sum1 = _mm256_setzero_pd();
                    let mut i = 0;
                    
                    while i + 8 <= n {
                        let x0 = _mm256_loadu_pd(x_ptr.add(i));
                        let y0 = _mm256_loadu_pd(y_ptr.add(i));
                        let x1 = _mm256_loadu_pd(x_ptr.add(i + 4));
                        let y1 = _mm256_loadu_pd(y_ptr.add(i + 4));
                        
                        let diff0 = _mm256_sub_pd(x0, y0);
                        let diff1 = _mm256_sub_pd(x1, y1);
                        
                        sum0 = _mm256_fmadd_pd(diff0, diff0, sum0);
                        sum1 = _mm256_fmadd_pd(diff1, diff1, sum1);
                        
                        i += 8;
                    }
                    
                    sum0 = _mm256_add_pd(sum0, sum1);
                    let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum0);
                    let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
                    
                    while i < n {
                        let d = *x_ptr.add(i) - *y_ptr.add(i);
                        result += d * d;
                        i += 1;
                    }
                    
                    result
                } else {
                    // Nicht-kontinuierlich
                    let mut sum = 0.0;
                    for i in 0..n {
                        let d = *x_ptr.offset(i as isize * x_stride) - 
                                *y_ptr.offset(i as isize * y_stride);
                        sum += d * d;
                    }
                    sum
                }
            },
            _ => {
                // Fallback
                let mut sum = 0.0;
                for i in 0..n {
                    let d = x[i] - y[i];
                    sum += d * d;
                }
                sum
            }
        }
    }
}