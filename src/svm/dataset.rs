use faer::{col::AsColMut, Mat, RowRef};
use ndarray::ArrayView2;
use rayon::prelude::*;
use std::marker::PhantomData;

pub trait SvmDataset<'a>: Send + Sync {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
    fn get_row(&self, i: usize) -> DatasetRowRef<'_>;
    fn to_owned(&self) -> FlatDataset<'static>;
}

pub enum DatasetStorage<'a> {
    Owned(Mat<f64>),
    Borrowed {
        ptr: *const f64,
        n_rows: usize,
        n_cols: usize,
        row_stride: isize,
        col_stride: isize,
        _phantom: PhantomData<&'a f64>,
    },
}

pub struct FlatDataset<'a> {
    pub storage: DatasetStorage<'a>,
}

unsafe impl<'a> Send for FlatDataset<'a> {}
unsafe impl<'a> Sync for FlatDataset<'a> {}

impl<'a> SvmDataset<'a> for FlatDataset<'a> {
    #[inline]
    fn n_samples(&self) -> usize {
        match &self.storage {
            DatasetStorage::Owned(mat) => mat.nrows(),
            DatasetStorage::Borrowed { n_rows, .. } => *n_rows,
        }
    }

    #[inline]
    fn n_features(&self) -> usize {
        match &self.storage {
            DatasetStorage::Owned(mat) => mat.ncols(),
            DatasetStorage::Borrowed { n_cols, .. } => *n_cols,
        }
    }

    #[inline]
    fn get_row(&self, i: usize) -> DatasetRowRef<'_> {
        match &self.storage {
            DatasetStorage::Owned(mat) => DatasetRowRef::Faer(mat.row(i)),
            DatasetStorage::Borrowed {
                ptr,
                n_cols,
                row_stride,
                col_stride,
                ..
            } => unsafe {
                let row_ptr = ptr.offset(i as isize * *row_stride);
                DatasetRowRef::Raw {
                    ptr: row_ptr,
                    len: *n_cols,
                    stride: *col_stride,
                    _phantom: PhantomData,
                }
            },
        }
    }

    fn to_owned(&self) -> FlatDataset<'static> {
        match self.storage {
            DatasetStorage::Owned(ref mat) => FlatDataset {
                storage: DatasetStorage::Owned(mat.clone()),
            },
            DatasetStorage::Borrowed {
                ptr,
                n_rows,
                n_cols,
                row_stride,
                col_stride,
                ..
            } => {
                let mut mat = Mat::<f64>::zeros(n_rows, n_cols);

                if row_stride == n_cols as isize && col_stride == 1 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(ptr, mat.as_ptr_mut(), n_rows * n_cols);
                    }
                } else {
                    // Your safe, sequential implementation
                    for j in 0..n_cols {
                        for i in 0..n_rows {
                            unsafe {
                                let value =
                                    *ptr.offset(i as isize * row_stride + j as isize * col_stride);
                                mat[(i, j)] = value;
                            }
                        }
                    }
                }

                FlatDataset {
                    storage: DatasetStorage::Owned(mat),
                }
            }
        }
    }
}

impl<'a> FlatDataset<'a> {
    pub fn new(data: Mat<f64>) -> FlatDataset<'static> {
        FlatDataset {
            storage: DatasetStorage::Owned(data),
        }
    }

    pub fn from_numpy_array(array: ArrayView2<'a, f64>) -> Self {
        let shape = array.shape();
        let strides = array.strides();

        FlatDataset {
            storage: DatasetStorage::Borrowed {
                ptr: array.as_ptr(),
                n_rows: shape[0],
                n_cols: shape[1],
                row_stride: strides[0],
                col_stride: strides[1],
                _phantom: PhantomData,
            },
        }
    }

    pub fn data(&self) -> Mat<f64> {
        match &self.storage {
            DatasetStorage::Owned(mat) => mat.clone(),
            DatasetStorage::Borrowed { .. } => SvmDataset::to_owned(self).storage.unwrap_owned(),
        }
    }
}

impl DatasetStorage<'_> {
    fn unwrap_owned(self) -> Mat<f64> {
        match self {
            DatasetStorage::Owned(mat) => mat,
            _ => panic!("Called unwrap_owned on a borrowed DatasetStorage"),
        }
    }
}

pub struct IndexedDataset<'a, D: SvmDataset<'a>> {
    parent: &'a D,
    indices: &'a [usize],
}

impl<'a, D: SvmDataset<'a>> IndexedDataset<'a, D> {
    pub fn new(parent: &'a D, indices: &'a [usize]) -> Self {
        Self { parent, indices }
    }
}

impl<'a, D: SvmDataset<'a>> SvmDataset<'a> for IndexedDataset<'a, D> {
    fn n_samples(&self) -> usize {
        self.indices.len()
    }

    fn n_features(&self) -> usize {
        self.parent.n_features()
    }

    #[inline]
    fn get_row(&self, i: usize) -> DatasetRowRef<'_> {
        let parent_idx = unsafe { *self.indices.get_unchecked(i) };
        self.parent.get_row(parent_idx)
    }

    fn to_owned(&self) -> FlatDataset<'static> {
        let n_samples = self.n_samples();
        let n_features = self.n_features();
        let mut mat = Mat::<f64>::zeros(n_samples, n_features);

        mat.as_mut()
            .par_col_chunks_mut(1)
            .enumerate()
            .for_each(|(j, mut dst_col)| {
                for i in 0..n_samples {
                    let src_row = self.get_row(i);
                    dst_col[(i, 0)] = src_row[j];
                }
            });

        FlatDataset::new(mat)
    }
}

impl Clone for FlatDataset<'static> {
    fn clone(&self) -> Self {
        match &self.storage {
            DatasetStorage::Owned(mat) => FlatDataset {
                storage: DatasetStorage::Owned(mat.clone()),
            },
            DatasetStorage::Borrowed { .. } => {
                panic!("Cannot clone a borrowed dataset that is not 'static")
            }
        }
    }
}

pub enum DatasetRowRef<'a> {
    Faer(RowRef<'a, f64>),
    Raw {
        ptr: *const f64,
        len: usize,
        stride: isize,
        _phantom: PhantomData<&'a ()>,
    },
}

unsafe impl<'a> Send for DatasetRowRef<'a> {}
unsafe impl<'a> Sync for DatasetRowRef<'a> {}

impl<'a> DatasetRowRef<'a> {
    #[inline]
    pub fn ncols(&self) -> usize {
        match self {
            DatasetRowRef::Faer(row) => row.ncols(),
            DatasetRowRef::Raw { len, .. } => *len,
        }
    }
}

impl<'a> std::ops::Index<usize> for DatasetRowRef<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            DatasetRowRef::Faer(row) => &row[index],
            DatasetRowRef::Raw { ptr, stride, .. } => {
                unsafe { &*ptr.offset(index as isize * *stride) }
            }
        }
    }
}