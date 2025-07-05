use faer::RowRef;

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
    pub fn compute_pair(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        match self {
            KernelType::Linear => self.compute_linear_faer(x, y),
            KernelType::RBF { neg_gamma, .. } => {
                let dist_sq = self.squared_distance_faer(x, y);
                (*neg_gamma * dist_sq).exp()
            }
            KernelType::Poly { degree, coef0, gamma } => {
                let dot = self.compute_linear_faer(x, y);
                (gamma * dot + *coef0).powi(*degree as i32)
            }
        }
    }
    
    #[inline(always)]
    pub fn compute_self(&self, x: &RowRef<'_, f64>) -> f64 {
        match self {
            KernelType::Linear => self.compute_linear_faer(x, x),
            KernelType::RBF { .. } => 1.0, // RBF(x,x) = 1
            KernelType::Poly { degree, coef0, gamma } => {
                let dot = self.compute_linear_faer(x, x);
                (gamma * dot + *coef0).powi(*degree as i32)
            }
        }
    }
    
    #[inline(always)]
    fn compute_linear_faer(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        let n = x.ncols();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n >= 32 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.dot_product_avx2_fma_unrolled(x, y) };
            }
        }
        
        let mut sum = 0.0;
        let mut i = 0;
        
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
    
    #[inline(always)]
    fn dot_product_unrolled(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        let mut i = 0;
        
        while i + 16 <= n {
            let sum0 = x[i] * y[i] + x[i+1] * y[i+1];
            let sum1 = x[i+2] * y[i+2] + x[i+3] * y[i+3];
            let sum2 = x[i+4] * y[i+4] + x[i+5] * y[i+5];
            let sum3 = x[i+6] * y[i+6] + x[i+7] * y[i+7];
            let sum4 = x[i+8] * y[i+8] + x[i+9] * y[i+9];
            let sum5 = x[i+10] * y[i+10] + x[i+11] * y[i+11];
            let sum6 = x[i+12] * y[i+12] + x[i+13] * y[i+13];
            let sum7 = x[i+14] * y[i+14] + x[i+15] * y[i+15];
            
            sum += (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
            i += 16;
        }
        
        while i + 4 <= n {
            sum += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] + x[i+3] * y[i+3];
            i += 4;
        }
        
        while i < n {
            sum += x[i] * y[i];
            i += 1;
        }
        
        sum
    }
    
    #[inline(always)]
    fn squared_distance_unrolled(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
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
    
    #[inline(always)]
    fn compute_linear_faer_fallback(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        let mut sum = 0.0;
        let n = x.ncols();
        for i in 0..n {
            sum += x[i] * y[i];
        }
        sum
    }
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_avx2_fma_unrolled(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        use std::arch::x86_64::*;
        
        let n = x.ncols();
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();
        let mut i = 0;
        
        while i + 16 <= n {
            let x0 = _mm256_loadu_pd(x.as_ptr().add(i));
            let y0 = _mm256_loadu_pd(y.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(x.as_ptr().add(i + 4));
            let y1 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
            let x2 = _mm256_loadu_pd(x.as_ptr().add(i + 8));
            let y2 = _mm256_loadu_pd(y.as_ptr().add(i + 8));
            let x3 = _mm256_loadu_pd(x.as_ptr().add(i + 12));
            let y3 = _mm256_loadu_pd(y.as_ptr().add(i + 12));
            
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
            result += x[i] * y[i];
            i += 1;
        }
        
        result
    }
    
    #[inline(always)]
    fn squared_distance_faer(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        let n = x.ncols();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n >= 32 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.squared_distance_avx2_fma(x, y) };
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
    unsafe fn squared_distance_avx2_fma(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        use std::arch::x86_64::*;
        
        let n = x.ncols();
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut i = 0;
        
        while i + 8 <= n {
            let x0 = _mm256_loadu_pd(x.as_ptr().add(i));
            let y0 = _mm256_loadu_pd(y.as_ptr().add(i));
            let x1 = _mm256_loadu_pd(x.as_ptr().add(i + 4));
            let y1 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
            
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
            let d = x[i] - y[i];
            result += d * d;
            i += 1;
        }
        
        result
    }
}