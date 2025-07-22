use faer::{Mat, MatRef, RowRef};
use ndarray::ArrayView2;
use std::marker::PhantomData;

pub enum DatasetStorage<'a> {
    Owned(Mat<f64>),
    Borrowed {
        ptr: *const f64,
        n_rows: usize,
        n_cols: usize,
        row_stride: isize,
        col_stride: isize,
        _phantom: PhantomData<&'a f64>,
    }
}

pub struct FlatDataset<'a> {
    pub storage: DatasetStorage<'a>,
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
            }
        }
    }
    
    pub fn to_owned(&self) -> FlatDataset<'static> {
        match &self.storage {
            DatasetStorage::Owned(mat) => FlatDataset {
                storage: DatasetStorage::Owned(mat.clone()),
            },
            DatasetStorage::Borrowed { ptr, n_rows, n_cols, row_stride, col_stride, .. } => {
                let mut mat = Mat::<f64>::zeros(*n_rows, *n_cols);
                
                unsafe {
                    if *row_stride == *n_cols as isize && *col_stride == 1 {
                        std::ptr::copy_nonoverlapping(
                            *ptr,
                            mat.as_ptr_mut(),
                            n_rows * n_cols
                        );
                    } else {
                        for i in 0..*n_rows {
                            let src_row = ptr.offset(i as isize * row_stride);
                            let dst_row = mat.as_ptr_mut().add(i * n_cols);
                            
                            if *col_stride == 1 {
                                std::ptr::copy_nonoverlapping(src_row, dst_row, *n_cols);
                            } else {
                                for j in 0..*n_cols {
                                    *dst_row.add(j) = *src_row.offset(j as isize * col_stride);
                                }
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

    #[inline]
    pub fn n_samples(&self) -> usize {
        match &self.storage {
            DatasetStorage::Owned(mat) => mat.nrows(),
            DatasetStorage::Borrowed { n_rows, .. } => *n_rows,
        }
    }

    #[inline]
    pub fn n_features(&self) -> usize {
        match &self.storage {
            DatasetStorage::Owned(mat) => mat.ncols(),
            DatasetStorage::Borrowed { n_cols, .. } => *n_cols,
        }
    }

    #[inline]
    pub fn get_row(&self, i: usize) -> DatasetRowRef<'_> {
        match &self.storage {
            DatasetStorage::Owned(mat) => DatasetRowRef::Faer(mat.row(i)),
            DatasetStorage::Borrowed { ptr, n_cols, row_stride, col_stride, .. } => {
                unsafe {
                    let row_ptr = ptr.offset(i as isize * row_stride);
                    DatasetRowRef::Raw {
                        ptr: row_ptr,
                        len: *n_cols,
                        stride: *col_stride,
                    }
                }
            }
        }
    }
    
    pub fn as_ref(&self) -> DatasetMatRef<'_> {
        match &self.storage {
            DatasetStorage::Owned(mat) => DatasetMatRef::Faer(mat.as_ref()),
            DatasetStorage::Borrowed { ptr, n_rows, n_cols, row_stride, col_stride, .. } => {
                DatasetMatRef::Raw {
                    ptr: *ptr,
                    n_rows: *n_rows,
                    n_cols: *n_cols,
                    row_stride: *row_stride,
                    col_stride: *col_stride,
                }
            }
        }
    }
    
    pub fn data(&self) -> Mat<f64> {
        match &self.storage {
            DatasetStorage::Owned(mat) => mat.clone(),
            DatasetStorage::Borrowed { .. } => self.to_owned().data()
        }
    }
}

impl Clone for FlatDataset<'static> {
    fn clone(&self) -> Self {
        match &self.storage {
            DatasetStorage::Owned(mat) => FlatDataset {
                storage: DatasetStorage::Owned(mat.clone()),
            },
            DatasetStorage::Borrowed { .. } => panic!("Cannot clone borrowed dataset"),
        }
    }
}

pub enum DatasetRowRef<'a> {
    Faer(RowRef<'a, f64>),
    Raw {
        ptr: *const f64,
        len: usize,
        stride: isize,
    }
}

impl<'a> DatasetRowRef<'a> {
    #[inline]
    pub fn as_ptr(&self) -> *const f64 {
        match self {
            DatasetRowRef::Faer(row) => row.as_ptr(),
            DatasetRowRef::Raw { ptr, .. } => *ptr,
        }
    }
    
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
            DatasetRowRef::Raw { ptr, stride, .. } => unsafe {
                &*ptr.offset(index as isize * stride)
            }
        }
    }
}

/// Ein Matrix-View
pub enum DatasetMatRef<'a> {
    Faer(MatRef<'a, f64>),
    Raw {
        ptr: *const f64,
        n_rows: usize,
        n_cols: usize,
        row_stride: isize,
        col_stride: isize,
    }
}

impl<'a> DatasetMatRef<'a> {
    pub fn row(&self, i: usize) -> DatasetRowRef<'a> {
        match self {
            DatasetMatRef::Faer(mat) => DatasetRowRef::Faer(mat.row(i)),
            DatasetMatRef::Raw { ptr, n_cols, row_stride, col_stride, .. } => {
                unsafe {
                    DatasetRowRef::Raw {
                        ptr: ptr.offset(i as isize * row_stride),
                        len: *n_cols,
                        stride: *col_stride,
                    }
                }
            }
        }
    }
}