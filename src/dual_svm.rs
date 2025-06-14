use crate::flat_dataset::FlatDataset;
use crate::flat_kernel_cache::FlatKernelCache;
use crate::svm_kernel::KernelType;
use crate::working_set::select_working_set_wss2_flat_cache;
use faer::Mat;

#[derive(Clone)]
pub struct DualSVM {
    pub alphas: Option<Mat<f64>>,
    pub support_vectors: Option<FlatDataset>,
    pub support_labels: Option<Mat<f64>>,
    pub bias: f64,
    pub c: f64,
    pub kernel: KernelType,
}

impl DualSVM {
    pub fn new(kernel: KernelType, c: f64) -> Self {
        Self {
            alphas: None,
            support_vectors: None,
            support_labels: None,
            bias: 0.0,
            c,
            kernel,
        }
    }

    pub fn fit(&mut self, dataset: FlatDataset, y: Vec<f64>, max_iter: usize, tol: f64) {
        let n = dataset.n_samples();
        let mut alphas = Mat::<f64>::zeros(n, 1);
        let mut bias = 0.0;
        let c = self.c;
        let mut passes = 0;
        let max_passes = 5;
        let mut iter = 0;
        let min_iter = 100;
        let mut grad = Mat::<f64>::from_fn(n, 1, |i, _| -y[i]);
        let mut kernel_cache = FlatKernelCache::new(self.kernel.clone(), dataset.clone(), 100);
        let active_set: Vec<usize> = (0..n).collect();

        while (passes < max_passes || iter < min_iter) && iter < max_iter {
            let mut maximal_violation: f64 = 0.0;


            while let Some(((ii, jj), violation)) = select_working_set_wss2_flat_cache(
                alphas.col_as_slice(0),
                &y,
                grad.col_as_slice(0),
                c,
                &mut kernel_cache,
                &active_set,
            ) {
                let i = active_set[ii];
                let j = active_set[jj];
                maximal_violation = maximal_violation.max(violation);

                let (yi, yj) = (y[i], y[j]);
                let (ai_old, aj_old) = (alphas.read(i, 0), alphas.read(j, 0));

                let (l, h) = if yi != yj {
                    (f64::max(0.0, aj_old - ai_old), f64::min(c, c + aj_old - ai_old))
                } else {
                    (f64::max(0.0, ai_old + aj_old - c), f64::min(c, ai_old + aj_old))
                };
                if (l - h).abs() < 1e-12 { break; }

                let kii = kernel_cache.get(i, i);
                let kjj = kernel_cache.get(j, j);
                let kij = kernel_cache.get(i, j);
                let eta = 2.0 * kij - kii - kjj;
                if eta >= 0.0 { break; }

                let (gi, gj) = (grad.read(i, 0), grad.read(j, 0));
                let mut aj_new = aj_old - yj * (gi - gj) / eta;
                aj_new = aj_new.clamp(l, h);
                if (aj_new - aj_old).abs() < 1e-6 { break; }
                let ai_new = ai_old + yi * yj * (aj_old - aj_new);

                let b1 = bias - gi - yi * (ai_new - ai_old) * kii - yj * (aj_new - aj_old) * kij;
                let b2 = bias - gj - yi * (ai_new - ai_old) * kij - yj * (aj_new - aj_old) * kjj;
                bias = if ai_new > 0.0 && ai_new < c { b1 }
                    else if aj_new > 0.0 && aj_new < c { b2 }
                    else { (b1 + b2) / 2.0 };

                alphas.write(i, 0, ai_new);
                alphas.write(j, 0, aj_new);

                let delta_ai = ai_new - ai_old;
                let delta_aj = aj_new - aj_old;

                for &k in &active_set {
                    let ki = kernel_cache.get(i, k);
                    let kj = kernel_cache.get(j, k);
                    let update = ki * yi * delta_ai + kj * yj * delta_aj;
                    grad.write(k, 0, grad.read(k, 0) + update);
                }
            }

            if maximal_violation < tol { passes += 1; } else { passes = 0; }
            iter += 1;
        }

        let sv_idx: Vec<usize> = (0..n).filter(|&i| alphas.read(i, 0) > 1e-8).collect();

        let mut sv_mat = Mat::zeros(sv_idx.len(), dataset.n_features());
        for (row_idx, &i) in sv_idx.iter().enumerate() {
            for j in 0..dataset.n_features() {
                sv_mat.write(row_idx, j, dataset.data.read(i, j));
            }
        }

        self.support_vectors = Some(FlatDataset { data: sv_mat });
        self.support_labels = Some(Mat::from_fn(sv_idx.len(), 1, |i, _| y[sv_idx[i]]));
        self.alphas = Some(Mat::from_fn(sv_idx.len(), 1, |i, _| alphas.read(sv_idx[i], 0)));
        self.bias = bias;
    }

    pub fn decision_function_batch(&self, dataset: &FlatDataset) -> Vec<f64> {
        let sv = self.support_vectors.as_ref().unwrap();
        let sl = self.support_labels.as_ref().unwrap();
        let al = self.alphas.as_ref().unwrap();
        let coeff = Mat::from_fn(al.nrows(), 1, |i, _| al.read(i, 0) * sl.read(i, 0));
        let bias = self.bias;

        if let KernelType::Linear = self.kernel {
            let kernel_matrix = dataset.data.as_ref() * sv.data.as_ref().transpose();
            let result_mat = &kernel_matrix * &coeff;
            return (0..result_mat.nrows()).map(|i| result_mat.read(i, 0) + bias).collect();
        }

        let mut result = Vec::with_capacity(dataset.n_samples());
        for i in 0..dataset.n_samples() {
            let xi = dataset.get_row(i);
            let mut sum = 0.0;
            for j in 0..sv.n_samples() {
                let svj = sv.get_row(j);
                sum += coeff.read(j, 0) * self.kernel.compute_pair_flat(xi, svj);
            }
            result.push(sum + bias);
        }
        result
    }
}
