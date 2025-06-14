use crate::flat_dataset::FlatDataset;
use crate::svm_kernel::KernelType;
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
        let val = self.cache.read(i, j);
        if !val.is_nan() {
            return val;
        }
        let xi = self.dataset.get_row(i);
        let xj = self.dataset.get_row(j);
        let val = self.kernel.compute_pair_flat(xi, xj);
        self.cache.write(i, j, val);
        self.cache.write(j, i, val);
        val
    }
}
