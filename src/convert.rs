// Conversion methods from diffsol matrix to Python 2D array

use diffsol::{DiffSlScalar, Scalar};

use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

// Trait for valid matrix and python scalar types (f32 and f64 currently used)
pub trait PyCompatibleScalar: Scalar + numpy::Element {}
impl<T> PyCompatibleScalar for T where T: Scalar + numpy::Element {}

// Strong binding to fulfil matrix types in diffsol and python.
// Required because Rust does not yet support associated‑type equality constraints involving higher‑ranked lifetimes
// within where clauses, i.e. the PySolve implementation can not yet have a where clause such as
//     for<'b> <<<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner as MatrixToPy<'b>>::T = M::T,
// in order to strongly bind the python type to the matrix type. This is discussed in Rust issue #20041.
// So here instead, MatrixType is an associated type which is directly taken from the python type.
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
pub trait RealF32OrF64: sealed::Sealed + DiffSlScalar + numpy::Element + num_traits::Float + std::ops::AddAssign {
    type MatrixType;
}
impl RealF32OrF64 for f32 {
    type MatrixType = f32;
}
impl RealF32OrF64 for f64 {
    type MatrixType = f64;
}

// 2D matrix to python array conversion
pub trait MatrixToPy<'py, T> {
    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<T>>;
}

impl<'py, T: PyCompatibleScalar> MatrixToPy<'py, T> for nalgebra::DMatrix<T> {
    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<T>> {
        let view = unsafe {
            ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
        };
        view.to_pyarray(py)
    }
}

impl<'py, T: PyCompatibleScalar> MatrixToPy<'py, T> for faer::Mat<T> {
    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<T>> {
        let strides = (self.row_stride() as usize, self.col_stride() as usize);
        let view =
            unsafe { ArrayView2::from_shape_ptr(self.shape().strides(strides), self.as_ptr()) };
        view.to_pyarray(py)
    }
}

// 1D vector to python array conversion
pub trait VectorToPy<'py, T> {
    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<T>>;
}

impl<'py, T: PyCompatibleScalar> VectorToPy<'py, T> for nalgebra::DVector<T> {
    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
        let view = unsafe { ArrayView1::from_shape_ptr(self.len(), self.as_ptr()) };
        view.to_pyarray(py)
    }
}

impl<'py, T: PyCompatibleScalar> VectorToPy<'py, T> for faer::Col<T> {
    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
        let view = unsafe { ArrayView1::from_shape_ptr(self.nrows(), self.as_ptr()) };
        view.to_pyarray(py)
    }
}
