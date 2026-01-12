// Conversion methods from diffsol matrix to Python 2D array

use numpy::{
    dtype,
    ndarray::{ArrayView1, ArrayView2, ShapeBuilder},
    Element, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;

use crate::{error::PyDiffsolError, py_types::PyCompatibleScalar};

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

pub(crate) fn to_arrayview2<'py, T>(
    arr: &'py Bound<'py, PyUntypedArray>,
) -> Result<ArrayView2<'py, T>, PyDiffsolError>
where
    T: Element,
{
    if arr.ndim() != 2 {
        return Err(PyDiffsolError::Conversion(format!(
            "Expecting 2D array but got {}D array",
            arr.ndim()
        )));
    }

    let expected = dtype::<T>(arr.py());
    let actual = arr.dtype();
    if !actual.is_equiv_to(&expected) {
        return Err(PyDiffsolError::Conversion(format!(
            "Expected array type of {} but got {}",
            std::any::type_name::<T>(),
            actual
        )));
    }

    let typed: &Bound<'py, PyArray2<T>> = arr
        .cast::<PyArray2<T>>()
        .map_err(|_| PyDiffsolError::Conversion("FIXME".to_string()))?;

    // SAFETY: From above checks at this point we know dtype, shape and lifetimes are correct
    Ok(unsafe { typed.as_array() })
}
