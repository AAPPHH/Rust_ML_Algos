use crate::svm::flat_dataset::FlatDataset;
use crate::svm::svm_kernel::KernelType;
use faer::Mat;

#[derive(Clone)]
pub struct FlatKernelCache {
    kernel: KernelType,
    dataset: FlatDataset,
    cache: Mat<f64>,
}

impl FlatKernelCache {
    pub fn new(kernel: KernelType, dataset: FlatDataset, _size: usize) -> Self {
        let n = dataset.n_samples();
        let cache = Mat::<f64>::from_fn(n, n, |_, _| f64::NAN);
        FlatKernelCache { kernel, dataset, cache }
    }

    pub fn get(&mut self, i: usize, j: usize) -> f64 {
        let cached = self.cache[(i, j)];
        if !cached.is_nan() {
            return cached;
        }

        let ncols   = self.dataset.data.ncols();
        let offset_i = i * ncols;
        let offset_j = j * ncols;
        let base_ptr = self.dataset.data.as_ptr();

        let xi = unsafe { std::slice::from_raw_parts(base_ptr.add(offset_i), ncols) };
        let xj = unsafe { std::slice::from_raw_parts(base_ptr.add(offset_j), ncols) };

        let val = self.kernel.compute_pair_flat(xi, xj);

        self.cache[(i, j)] = val;
        self.cache[(j, i)] = val;
        val
    }
}
