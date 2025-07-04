use faer::RowRef;

#[derive(Clone)]
pub enum KernelType {
    Linear,
    RBF { gamma: f64 },
    Poly { degree: u32, coef0: f64, gamma: f64 },
}

impl KernelType {
    #[inline(always)]
    pub fn compute_pair(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        match self {
            KernelType::Linear => self.compute_linear(x, y),
            KernelType::RBF { gamma } => self.compute_rbf(x, y, *gamma),
            KernelType::Poly { degree, coef0, gamma } => {
                self.compute_poly(x, y, *degree, *coef0, *gamma)
            }
        }
    }
    
    #[inline(always)]
    pub fn compute_self(&self, x: &RowRef<'_, f64>) -> f64 {
        match self {
            KernelType::Linear => self.compute_linear(x, x),
            KernelType::RBF { .. } => 1.0, // RBF(x,x) = 1
            KernelType::Poly { degree, coef0, gamma } => {
                let dot = self.compute_linear(x, x);
                (gamma * dot + *coef0).powi(*degree as i32)
            }
        }
    }
    
    #[inline(always)]
    fn compute_linear(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        let n = x.ncols();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n >= 32 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.dot_product_avx2_fma(x, y) };
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
    fn compute_rbf(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>, gamma: f64) -> f64 {
        let dist_sq = self.squared_distance(x, y);
        (-gamma * dist_sq).exp()
    }
    
    #[inline(always)]
    fn compute_poly(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>, degree: u32, coef0: f64, gamma: f64) -> f64 {
        let dot = self.compute_linear(x, y);
        (gamma * dot + coef0).powi(degree as i32)
    }
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_avx2_fma(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        use std::arch::x86_64::*;
        
        let n = x.ncols();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut i = 0;
        
        while i + 8 <= n {
            let x1 = _mm256_loadu_pd(x.as_ptr().add(i));
            let y1 = _mm256_loadu_pd(y.as_ptr().add(i));
            let x2 = _mm256_loadu_pd(x.as_ptr().add(i + 4));
            let y2 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
            
            sum1 = _mm256_fmadd_pd(x1, y1, sum1);
            sum2 = _mm256_fmadd_pd(x2, y2, sum2);
            
            i += 8;
        }
        
        sum1 = _mm256_add_pd(sum1, sum2);
        
        let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum1);
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        while i < n {
            result += x[i] * y[i];
            i += 1;
        }
        
        result
    }
    
    #[inline(always)]
    fn squared_distance(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        let n = x.ncols();
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n >= 32 && is_x86_feature_detected!("avx2") {
                return unsafe { self.squared_distance_avx2(x, y) };
            }
        }
        
        let mut sum = 0.0;
        let mut i = 0;
        
        while i + 4 <= n {
            let d0 = x[i] - y[i];
            let d1 = x[i+1] - y[i+1];
            let d2 = x[i+2] - y[i+2];
            let d3 = x[i+3] - y[i+3];
            
            sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
            i += 4;
        }
        
        while i < n {
            let d = x[i] - y[i];
            sum += d * d;
            i += 1;
        }
        
        sum
    }
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn squared_distance_avx2(&self, x: &RowRef<'_, f64>, y: &RowRef<'_, f64>) -> f64 {
        use std::arch::x86_64::*;
        
        let n = x.ncols();
        let mut sum = _mm256_setzero_pd();
        let mut i = 0;
        
        while i + 4 <= n {
            let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));
            let y_vec = _mm256_loadu_pd(y.as_ptr().add(i));
            let diff = _mm256_sub_pd(x_vec, y_vec);
            let sq = _mm256_mul_pd(diff, diff);
            sum = _mm256_add_pd(sum, sq);
            i += 4;
        }
        
        let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum);
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        while i < n {
            let d = x[i] - y[i];
            result += d * d;
            i += 1;
        }
        
        result
    }
}