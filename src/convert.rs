// Conversion methods from diffsol matrix to Python 2D array

use diffsol::Scalar;

use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

// Trait for valid matrix and python scalar types (f32 and f64 currently used)
pub trait PyCompatibleScalar: Scalar + numpy::Element {}
impl<T> PyCompatibleScalar for T where T: Scalar + numpy::Element {}

// 2D matrix to python array conversion
pub trait MatrixToPy<'py> {
    type T: PyCompatibleScalar;

    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<Self::T>>;
}

impl<'py, T: PyCompatibleScalar> MatrixToPy<'py> for nalgebra::DMatrix<T> {
    type T = T;

    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<T>> {
        let view = unsafe {
            ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
        };
        view.to_pyarray(py)
    }
}

impl<'py, T: PyCompatibleScalar> MatrixToPy<'py> for faer::Mat<T> {
    type T = T;

    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<T>> {
        let strides = (self.row_stride() as usize, self.col_stride() as usize);
        let view =
            unsafe { ArrayView2::from_shape_ptr(self.shape().strides(strides), self.as_ptr()) };
        view.to_pyarray(py)
    }
}

// 1D vector to python array conversion
pub trait VectorToPy<'py> {
    type T: Scalar + numpy::Element;

    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<Self::T>>;
}

impl<'py, T: PyCompatibleScalar> VectorToPy<'py> for nalgebra::DVector<T> {
    type T = T;

    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
        let view = unsafe { ArrayView1::from_shape_ptr(self.len(), self.as_ptr()) };
        view.to_pyarray(py)
    }
}

impl<'py, T: PyCompatibleScalar> VectorToPy<'py> for faer::Col<T> {
    type T = T;

    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
        let view = unsafe { ArrayView1::from_shape_ptr(self.nrows(), self.as_ptr()) };
        view.to_pyarray(py)
    }
}
