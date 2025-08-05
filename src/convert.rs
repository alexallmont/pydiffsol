use numpy::{ToPyArray, PyArray2};
use numpy::ndarray::{ArrayView2, ShapeBuilder};
use pyo3::prelude::*;

pub trait MatrixToPy<'py> {
  fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>>;
}

impl<'py> MatrixToPy<'py> for nalgebra::DMatrix<f64> {
  fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
    let view = unsafe {
        ArrayView2::from_shape_ptr(
            self.shape().strides(self.strides()),
            self.as_ptr()
        )
    };
    view.to_pyarray(py).into()
  }
}

impl<'py> MatrixToPy<'py> for faer::Mat<f64> {
  fn to_pyarray2(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
    let strides = (self.row_stride() as usize, self.col_stride() as usize);
    let view = unsafe {
        ArrayView2::from_shape_ptr(
            self.shape().strides(strides),
            self.as_ptr()
        )
    };
    view.to_pyarray(py).into()
  }
}
