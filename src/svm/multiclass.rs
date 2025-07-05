use rayon::prelude::*;
use crate::svm::kernel::KernelType;
use crate::svm::dual_svm::DualSVM;
use crate::svm::dataset::FlatDataset;
use faer::Mat;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct SVM {
    pub classifiers: Vec<(f64, f64, DualSVM)>,
    pub classes: Vec<f64>,
    pub c: f64,
    pub kernel_type: String,
    pub degree: u32,
    pub coef0: f64,
    pub gamma: f64,
}

impl SVM {
    pub fn poly(degree: u32, coef0: f64, c: f64, gamma: Option<f64>) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
            c,
            kernel_type: "poly".to_string(),
            degree,
            coef0,
            gamma: gamma.unwrap_or(1.0),
        }
    }

    pub fn rbf(gamma: f64, c: f64) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
            c,
            kernel_type: "rbf".to_string(),
            degree: 0,
            coef0: 0.0,
            gamma,
        }
    }

    pub fn linear(c: f64) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
            c,
            kernel_type: "linear".to_string(),
            degree: 1,
            coef0: 0.0,
            gamma: 0.0,
        }
    }

    pub fn fit(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<(), String> {
        let n_samples = x.len();
        if n_samples == 0 || y.len() != n_samples {
            return Err("Empty data or label size mismatch".to_string());
        }
        let n_features = x[0].len();

        let mut classes: Vec<f64> = y.clone();
        classes.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        self.classes = classes.clone();

        let kernel_def = match self.kernel_type.as_str() {
            "poly" => KernelType::Poly {
                degree: self.degree,
                coef0: self.coef0,
                gamma: self.gamma,
            },
            "rbf" => KernelType::new_rbf(self.gamma),
            "linear" => KernelType::Linear,
            _ => KernelType::Linear,
        };

        let dataset = FlatDataset::from_nested(x);
        let c_val = self.c;

        let pairs: Vec<(f64, f64)> = classes
            .iter()
            .enumerate()
            .flat_map(|(i, &a)| classes.iter().skip(i + 1).map(move |&b| (a, b)))
            .collect();

        let progress = AtomicUsize::new(0);
        let total_pairs = pairs.len();
        
        let classifiers: Vec<(f64, f64, DualSVM)> = pairs
            .par_iter()
            .map(|&(class_a, class_b)| {
                let idx: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter(|(_, &lab)| lab == class_a || lab == class_b)
                    .map(|(i, _)| i)
                    .collect();

                if idx.is_empty() {
                    return (class_a, class_b, DualSVM::new(kernel_def.clone(), c_val));
                }

                let mut x_bin_mat = Mat::<f64>::zeros(idx.len(), n_features);

                if idx.len() > 100 && n_features > 100 {
                    let rows_data: Vec<(usize, Vec<f64>)> = idx
                        .par_iter()
                        .enumerate()
                        .map(|(row_idx, &i)| {
                            let src_row = dataset.data.row(i);
                            let row_vec: Vec<f64> = (0..n_features).map(|j| src_row[j]).collect();
                            (row_idx, row_vec)
                        })
                        .collect();
                    
                    for (row_idx, row_data) in rows_data {
                        for (j, &val) in row_data.iter().enumerate() {
                            x_bin_mat[(row_idx, j)] = val;
                        }
                    }
                } else {
                    for (row_idx, &i) in idx.iter().enumerate() {
                        let src_row = dataset.data.row(i);
                        let mut dst_row = x_bin_mat.row_mut(row_idx);
                        dst_row.copy_from(src_row);
                    }
                }


                let x_bin = FlatDataset { data: x_bin_mat };

                let y_bin: Vec<f64> = idx.iter()
                    .map(|&i| if y[i] == class_a { 1.0 } else { -1.0 })
                    .collect();

                let mut svm = DualSVM::new(kernel_def.clone(), c_val);
                svm.fit(x_bin, y_bin, max_iter, tol);

                let completed = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % 10 == 0 || completed == total_pairs {
                    eprintln!("Training progress: {}/{} classifiers", completed, total_pairs);
                }

                (class_a, class_b, svm)
            })
            .collect();

        self.classifiers = classifiers;
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        let n_samples = x.len();
        if n_samples == 0 {
            return vec![];
        }
        
        let dataset = FlatDataset::from_nested(x);
        let n_classes = self.classes.len();

        if n_samples < 100 {
            return self.predict_sequential(&dataset);
        }


        let votes: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|_| vec![0usize; n_classes])
            .collect();

        self.classifiers.par_iter().for_each(|(class_a, class_b, svm)| {
            let preds = svm.decision_function_batch(&dataset);
            let idx_a = self.classes.iter().position(|c| c == class_a).unwrap();
            let idx_b = self.classes.iter().position(|c| c == class_b).unwrap();
            
            for (i, &score) in preds.iter().enumerate() {
                unsafe {
                    let vote_ptr = if score >= 0.0 {
                        votes[i].as_ptr().add(idx_a) as *mut usize
                    } else {
                        votes[i].as_ptr().add(idx_b) as *mut usize
                    };
                    let current = std::ptr::read_volatile(vote_ptr);
                    std::ptr::write_volatile(vote_ptr, current + 1);
                }
            }
        });

        votes.into_par_iter()
            .map(|row| {
                let (idx, _) = row.iter().enumerate()
                    .max_by_key(|&(_, cnt)| cnt)
                    .unwrap();
                self.classes[idx]
            })
            .collect()
    }

    fn predict_sequential(&self, dataset: &FlatDataset) -> Vec<f64> {
        let n_samples = dataset.n_samples();
        let n_classes = self.classes.len();
        let mut votes = vec![vec![0usize; n_classes]; n_samples];

        for (class_a, class_b, svm) in &self.classifiers {
            let preds = svm.decision_function_batch(dataset);
            let idx_a = self.classes.iter().position(|c| c == class_a).unwrap();
            let idx_b = self.classes.iter().position(|c| c == class_b).unwrap();
            
            for (i, &score) in preds.iter().enumerate() {
                if score >= 0.0 {
                    votes[i][idx_a] += 1;
                } else {
                    votes[i][idx_b] += 1;
                }
            }
        }

        votes.iter()
            .map(|row| {
                let (idx, _) = row.iter().enumerate()
                    .max_by_key(|&(_, cnt)| cnt)
                    .unwrap();
                self.classes[idx]
            })
            .collect()
    }

    pub fn predict_proba(&self, x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n_samples = x.len();
        if n_samples == 0 {
            return vec![];
        }
        
        let dataset = FlatDataset::from_nested(x);
        let n_classes = self.classes.len();

        let mut decision_values = vec![vec![vec![0.0; n_classes]; n_classes]; n_samples];

        for (class_a, class_b, svm) in &self.classifiers {
            let preds = svm.decision_function_batch(&dataset);
            let idx_a = self.classes.iter().position(|c| c == class_a).unwrap();
            let idx_b = self.classes.iter().position(|c| c == class_b).unwrap();
            
            for (i, &score) in preds.iter().enumerate() {
                decision_values[i][idx_a][idx_b] = score;
                decision_values[i][idx_b][idx_a] = -score;
            }
        }

        decision_values.into_par_iter()
            .map(|sample_decisions| {
                let mut probs = vec![0.0; n_classes];
                let mut sum = 0.0;
                
                for i in 0..n_classes {
                    let mut prob = 1.0;
                    for j in 0..n_classes {
                        if i != j {
                            let p = 1.0 / (1.0 + (-sample_decisions[i][j]).exp());
                            prob *= p;
                        }
                    }
                    probs[i] = prob;
                    sum += prob;
                }
                
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }
                
                probs
            })
            .collect()
    }
}