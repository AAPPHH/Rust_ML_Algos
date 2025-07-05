use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

use my_rust_module::svm::{
    SVM, DualSVM, FlatDataset, KernelType,
    cache::{KernelCache, SetAssociativeCache},
    working_set::PartialArgMaxSelector,
};

// Synthetic data generators
fn generate_linear_separable_data(n_samples: usize, n_features: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, noise).unwrap();
    
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);
    
    // Generate hyperplane coefficients
    let w: Vec<f64> = (0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b = rng.gen_range(-1.0..1.0);
    
    for _ in 0..n_samples {
        let mut sample: Vec<f64> = (0..n_features)
            .map(|_| rng.gen_range(-10.0..10.0))
            .collect();
        
        // Calculate true label
        let score: f64 = sample.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f64>() + b;
        let label = if score + normal.sample(&mut rng) > 0.0 { 1.0 } else { -1.0 };
        
        y.push(label);
        x.push(sample);
    }
    
    (x, y)
}

fn generate_clusters_data(n_samples: usize, n_features: usize, n_clusters: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(42);
    let samples_per_cluster = n_samples / n_clusters;
    
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);
    
    // Generate cluster centers
    let centers: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| {
            (0..n_features)
                .map(|_| rng.gen_range(-20.0..20.0))
                .collect()
        })
        .collect();
    
    // Generate samples around centers
    for (cluster_id, center) in centers.iter().enumerate() {
        let normal = Normal::new(0.0, 2.0).unwrap();
        
        for _ in 0..samples_per_cluster {
            let sample: Vec<f64> = center.iter()
                .map(|&c| c + normal.sample(&mut rng))
                .collect();
            
            x.push(sample);
            y.push(cluster_id as f64);
        }
    }
    
    (x, y)
}

// Benchmark individual components
fn benchmark_kernel_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_cache");
    
    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, _) = generate_linear_separable_data(n_samples, 50, 0.1);
        let dataset = FlatDataset::from_nested(x);
        let kernel = KernelType::new_rbf(0.1);
        
        group.bench_with_input(
            BenchmarkId::new("get_operations", n_samples),
            &n_samples,
            |b, &n| {
                let mut cache = SetAssociativeCache::new(kernel.clone(), dataset.clone(), 256);
                let mut idx = 0;
                b.iter(|| {
                    let i = idx % n;
                    let j = (idx + 1) % n;
                    idx += 1;
                    black_box(cache.get(i, j))
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("get_row_batch", n_samples),
            &n_samples,
            |b, &n| {
                let mut cache = SetAssociativeCache::new(kernel.clone(), dataset.clone(), 256);
                let mut output = vec![0.0; 100];
                let mut idx = 0;
                b.iter(|| {
                    let i = idx % n;
                    let range = 0..100.min(n);
                    cache.get_row_batch(i, range, &mut output);
                    idx += 1;
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_working_set_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("working_set_selection");
    
    for &n_samples in &[100, 500, 1000, 5000] {
        let (x, y) = generate_linear_separable_data(n_samples, 50, 0.1);
        let dataset = FlatDataset::from_nested(x);
        let kernel = KernelType::new_rbf(0.1);
        
        group.bench_with_input(
            BenchmarkId::new("select_working_set", n_samples),
            &n_samples,
            |b, &n| {
                let mut cache = SetAssociativeCache::new(kernel.clone(), dataset.clone(), 256);
                let mut selector = PartialArgMaxSelector::new(n);
                let alphas = vec![0.1; n];
                let grad = vec![-1.0; n];
                let active_indices: Vec<usize> = (0..n).collect();
                
                b.iter(|| {
                    selector.select_working_set_optimized(
                        &alphas,
                        &y,
                        &grad,
                        1.0,
                        &mut cache,
                        &active_indices,
                    )
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_kernel_computations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_computations");
    
    for &n_features in &[10, 50, 100, 500] {
        let x = vec![vec![1.0; n_features]; 2];
        let dataset = FlatDataset::from_nested(x);
        let row1 = dataset.get_row(0);
        let row2 = dataset.get_row(1);
        
        group.bench_with_input(
            BenchmarkId::new("linear_kernel", n_features),
            &n_features,
            |b, _| {
                let kernel = KernelType::Linear;
                b.iter(|| kernel.compute_pair(&row1, &row2));
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("rbf_kernel", n_features),
            &n_features,
            |b, _| {
                let kernel = KernelType::new_rbf(0.1);
                b.iter(|| kernel.compute_pair(&row1, &row2));
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("poly_kernel", n_features),
            &n_features,
            |b, _| {
                let kernel = KernelType::Poly { degree: 3, coef0: 1.0, gamma: 0.1 };
                b.iter(|| kernel.compute_pair(&row1, &row2));
            }
        );
    }
    
    group.finish();
}

fn benchmark_full_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_training");
    group.sample_size(10); // Reduce sample size for longer benchmarks
    
    for &n_samples in &[100, 500, 1000] {
        // Binary classification
        let (x, y) = generate_linear_separable_data(n_samples, 20, 0.1);
        
        group.bench_with_input(
            BenchmarkId::new("linear_svm", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut svm = SVM::linear(1.0);
                    svm.fit(x.clone(), y.clone(), 100, 1e-3).unwrap();
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("rbf_svm", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut svm = SVM::rbf(0.1, 1.0);
                    svm.fit(x.clone(), y.clone(), 100, 1e-3).unwrap();
                });
            }
        );
        
        // Multiclass
        let (x_multi, y_multi) = generate_clusters_data(n_samples, 20, 3);
        
        group.bench_with_input(
            BenchmarkId::new("multiclass_linear", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut svm = SVM::linear(1.0);
                    svm.fit(x_multi.clone(), y_multi.clone(), 100, 1e-3).unwrap();
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    
    // Train models once
    let (x_train, y_train) = generate_linear_separable_data(1000, 50, 0.1);
    
    let mut linear_svm = SVM::linear(1.0);
    linear_svm.fit(x_train.clone(), y_train.clone(), 100, 1e-3).unwrap();
    
    let mut rbf_svm = SVM::rbf(0.1, 1.0);
    rbf_svm.fit(x_train.clone(), y_train.clone(), 100, 1e-3).unwrap();
    
    for &n_samples in &[10, 100, 1000] {
        let (x_test, _) = generate_linear_separable_data(n_samples, 50, 0.1);
        
        group.bench_with_input(
            BenchmarkId::new("linear_predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| linear_svm.predict(x_test.clone()));
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("rbf_predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| rbf_svm.predict(x_test.clone()));
            }
        );
    }
    
    group.finish();
}

fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    use my_rust_module::svm::memory::{AlignedBuffer, get_pooled_vec};
    
    for &size in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("aligned_buffer", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let mut buffer = AlignedBuffer::new(s);
                    buffer.resize(s);
                    let slice = buffer.as_mut_slice();
                    for i in 0..s {
                        slice[i] = i as f64;
                    }
                    black_box(slice[s/2])
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("pooled_vec", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let mut vec = get_pooled_vec(s);
                    vec.resize(s, 0.0);
                    for i in 0..s {
                        vec[i] = i as f64;
                    }
                    black_box(vec[s/2])
                });
            }
        );
    }
    
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = 
        benchmark_kernel_cache,
        benchmark_working_set_selection,
        benchmark_kernel_computations,
        benchmark_full_training,
        benchmark_prediction,
        benchmark_memory_operations
}

criterion_main!(benches);