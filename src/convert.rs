use nalgebra::DMatrix;
use numpy::{ToPyArray, PyArray2};
use numpy::ndarray::{ArrayView2, ShapeBuilder};
use pyo3::prelude::*;

pub trait MatrixToPy<'py> {
  fn to_pyarray_view(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>>;
}

impl<'py> MatrixToPy<'py> for DMatrix<f64> {
  fn to_pyarray_view(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
      let view = unsafe {
          ArrayView2::from_shape_ptr(
              self.shape().strides(self.strides()),
              self.as_ptr()
          )
      };
      view.to_pyarray(py).into()
  }
}
