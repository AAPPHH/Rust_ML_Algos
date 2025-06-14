use crate::flat_dataset::FlatDataset;
use crate::svm_kernel::KernelType;

#[derive(Clone)]
pub struct FlatKernelCache {
    kernel: KernelType,
    dataset: FlatDataset,
    cache: Vec<Vec<Option<f64>>>,
}

impl FlatKernelCache {
    pub fn new(kernel: KernelType, dataset: FlatDataset, _size: usize) -> Self {
        let n = dataset.n_samples();
        let cache = vec![vec![None; n]; n];
        FlatKernelCache { kernel, dataset, cache }
    }

    pub fn get(&mut self, i: usize, j: usize) -> f64 {
        if let Some(val) = self.cache[i][j] {
            return val;
        }
        let xi = self.dataset.get_row(i);
        let xj = self.dataset.get_row(j);
        let val = self.kernel.compute_pair_flat(xi, xj);
        self.cache[i][j] = Some(val);
        self.cache[j][i] = Some(val);
        val
    }
}
