pub mod dual_svm;
pub mod multiclass;
pub mod kernel;
pub mod cache;
pub mod working_set;
pub mod dataset;
pub mod memory;
pub mod python;

pub use multiclass::SVM;
pub use dual_svm::DualSVM;
pub use dataset::FlatDataset;
pub use kernel::KernelType;