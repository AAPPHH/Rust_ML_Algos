use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use faer::Mat;

use my_rust_module::svm::{
    SVM, FlatDataset, KernelType,
    cache::{KernelCache, SetAssociativeCache},
    working_set::WSS2Selector,
};

fn generate_dataset(n_samples: usize, n_features: usize) -> (FlatDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 0.1).unwrap();
    
    let mut data = Mat::<f64>::zeros(n_samples, n_features);
    let mut y = Vec::with_capacity(n_samples);
    
    // Generate linearly separable data
    let w: Vec<f64> = (0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b = rng.gen_range(-1.0..1.0);
    
    for i in 0..n_samples {
        let mut score = b;
        for j in 0..n_features {
            let val = rng.gen_range(-10.0..10.0);
            data[(i, j)] = val;
            score += val * w[j];
        }
        
        let label = if score + normal.sample(&mut rng) > 0.0 { 1.0 } else { -1.0 };
        y.push(label);
    }
    
    (FlatDataset::new(data), y)
}

fn generate_multiclass_dataset(n_samples: usize, n_features: usize, n_classes: usize) -> (FlatDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(42);
    let samples_per_class = n_samples / n_classes;
    
    let mut data = Mat::<f64>::zeros(n_samples, n_features);
    let mut y = Vec::with_capacity(n_samples);
    
    // Generate cluster centers
    let centers: Vec<Vec<f64>> = (0..n_classes)
        .map(|_| {
            (0..n_features)
                .map(|_| rng.gen_range(-20.0..20.0))
                .collect()
        })
        .collect();
    
    let normal = Normal::new(0.0, 2.0).unwrap();
    
    let mut idx = 0;
    for (class_id, center) in centers.iter().enumerate() {
        for _ in 0..samples_per_class {
            if idx >= n_samples { break; }
            
            for j in 0..n_features {
                data[(idx, j)] = center[j] + normal.sample(&mut rng);
            }
            
            y.push(class_id as f64);
            idx += 1;
        }
    }
    
    (FlatDataset::new(data), y)
}

fn benchmark_kernel_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_cache");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(2));
    
    for &n_samples in &[100, 1000] {
        let (dataset, _) = generate_dataset(n_samples, 50);
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
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));
    
    for &n_samples in &[100, 1000] {
        let (dataset, y) = generate_dataset(n_samples, 50);
        let kernel = KernelType::new_rbf(0.1);
        
        group.bench_with_input(
            BenchmarkId::new("select_working_set", n_samples),
            &n_samples,
            |b, &n| {
                let mut cache = SetAssociativeCache::new(kernel.clone(), dataset.clone(), 256);
                let mut selector = WSS2Selector::new(n);
                let alphas = vec![0.1; n];
                let grad = vec![-1.0; n];
                let active_indices: Vec<usize> = (0..n).collect();
                
                b.iter(|| {
                    selector.select_working_set_wss2(
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
    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(2));
    
    for &n_features in &[10, 100] {
        let mut data = Mat::<f64>::zeros(2, n_features);
        for i in 0..2 {
            for j in 0..n_features {
                data[(i, j)] = 1.0;
            }
        }
        let dataset = FlatDataset::new(data);
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
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));
    group.warm_up_time(std::time::Duration::from_secs(1));
    
    for &n_samples in &[100] {
        let (dataset, y) = generate_dataset(n_samples, 10);
        
        group.bench_with_input(
            BenchmarkId::new("linear_svm", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut svm = SVM::linear(1.0);
                    svm.fit_dataset(dataset.clone(), y.clone(), 10, 1e-1).unwrap();
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("rbf_svm", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut svm = SVM::rbf(0.1, 1.0);
                    svm.fit_dataset(dataset.clone(), y.clone(), 10, 1e-1).unwrap();
                });
            }
        );
        
        let (dataset_multi, y_multi) = generate_multiclass_dataset(n_samples, 10, 3);
        
        group.bench_with_input(
            BenchmarkId::new("multiclass_linear", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut svm = SVM::linear(1.0);
                    svm.fit_dataset(dataset_multi.clone(), y_multi.clone(), 10, 1e-1).unwrap();
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));
    
    let (dataset_train, y_train) = generate_dataset(200, 10);
    
    let mut linear_svm = SVM::linear(1.0);
    println!("Training linear SVM for prediction benchmark...");
    if let Err(e) = linear_svm.fit_dataset(dataset_train.clone(), y_train.clone(), 30, 1e-2) {
        println!("Linear SVM training failed: {:?}", e);
        return;
    }
    
    let mut rbf_svm = SVM::rbf(0.1, 1.0);
    println!("Training RBF SVM for prediction benchmark...");
    if let Err(e) = rbf_svm.fit_dataset(dataset_train.clone(), y_train.clone(), 30, 1e-2) {
        println!("RBF SVM training failed: {:?}", e);
        return;
    }
    
    for &n_samples in &[10, 100] {
        let (dataset_test, _) = generate_dataset(n_samples, 10);
        
        group.bench_with_input(
            BenchmarkId::new("linear_predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| linear_svm.predict_dataset(&dataset_test));
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("rbf_predict", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| rbf_svm.predict_dataset(&dataset_test));
            }
        );
    }
    
    group.finish();
}

fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(3));
    
    use  my_rust_module::svm::memory::{AlignedBuffer, get_pooled_vec};
    
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
        .warm_up_time(std::time::Duration::from_secs(1))
        .significance_level(0.1)
        .confidence_level(0.90)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = 
        benchmark_kernel_cache,
        benchmark_working_set_selection,
        benchmark_kernel_computations,
        benchmark_full_training,
        benchmark_memory_operations
}

criterion_main!(benches);