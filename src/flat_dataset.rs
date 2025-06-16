use faer::Mat;

#[derive(Clone)]
pub struct FlatDataset {
    pub data: Mat<f64>,
}

impl FlatDataset {
    pub fn from_nested(nested: Vec<Vec<f64>>) -> Self {
        let n_samples = nested.len();
        let n_features = nested[0].len();

        let mut mat = Mat::<f64>::zeros(n_samples, n_features);

        for (i, sample) in nested.into_iter().enumerate() {
            assert_eq!(sample.len(), n_features);
            for (j, &val) in sample.iter().enumerate() {
                mat[(i, j)] = val;
            }
        }

        FlatDataset { data: mat }
    }

    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    pub fn get_row(&self, i: usize) -> Vec<f64> {
        let cols = self.data.ncols();
        (0..cols).map(|j| self.data[(i, j)]).collect()
    }
}
