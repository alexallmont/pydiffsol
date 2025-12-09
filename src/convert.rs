// Conversion methods from diffsol matrix to Python 2D array

use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

pub trait MatrixToPy<'py> {
    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>>;
}

impl<'py> MatrixToPy<'py> for nalgebra::DMatrix<f64> {
    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let view = unsafe {
            ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
        };
        view.to_pyarray(py)
    }
}

impl<'py> MatrixToPy<'py> for faer::Mat<f64> {
    fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let strides = (self.row_stride() as usize, self.col_stride() as usize);
        let view =
            unsafe { ArrayView2::from_shape_ptr(self.shape().strides(strides), self.as_ptr()) };
        view.to_pyarray(py)
    }
}

pub trait VectorToPy<'py> {
    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>>;
}

impl<'py> VectorToPy<'py> for nalgebra::DVector<f64> {
    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let view = unsafe { ArrayView1::from_shape_ptr(self.len(), self.as_ptr()) };
        view.to_pyarray(py)
    }
}

impl<'py> VectorToPy<'py> for faer::Col<f64> {
    fn to_pyarray1(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let view = unsafe { ArrayView1::from_shape_ptr(self.nrows(), self.as_ptr()) };
        view.to_pyarray(py)
    }
}
