use crate::svm_kernel::KernelType;

#[derive(Clone)]
pub struct FlatDataset {
    pub data: Vec<f64>,
    pub n_samples: usize,
    pub n_features: usize,
}

impl FlatDataset {
    pub fn from_nested(nested: Vec<Vec<f64>>) -> Self {
        let n_samples = nested.len();
        let n_features = nested[0].len();

        let mut data = Vec::with_capacity(n_samples * n_features);
        for sample in nested {
            assert_eq!(sample.len(), n_features);
            data.extend_from_slice(&sample);
        }

        FlatDataset { data, n_samples, n_features }
    }

    pub fn get_row(&self, i: usize) -> &[f64] {
        let start = i * self.n_features;
        let end = start + self.n_features;
        &self.data[start..end]
    }
}
