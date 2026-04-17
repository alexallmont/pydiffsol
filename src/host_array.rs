use std::mem::size_of;

use diffsol_c::host_array::HostArray;
use numpy::{
    ndarray::{Array2, ArrayView2},
    PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::{prelude::*, PyAny};

use crate::error::PyDiffsolError;

/// Utility function for calculating byte strides from element strides and size, with error handling for negative strides.
fn byte_strides(strides: &[isize], elem_size: usize) -> Result<Vec<usize>, PyDiffsolError> {
    strides
        .iter()
        .map(|&stride| {
            let stride = usize::try_from(stride).map_err(|_| {
                PyDiffsolError::Conversion(
                    "Negative-stride NumPy arrays are not supported".to_string(),
                )
            })?;
            Ok(stride * elem_size)
        })
        .collect()
}

pub(crate) fn pyarray1_to_host(
    array: PyReadonlyArray1<'_, f64>,
) -> Result<HostArray, PyDiffsolError> {
    let slice = array.as_slice().map_err(|_| {
        PyDiffsolError::Conversion("Expected a contiguous 1D float64 NumPy array".to_string())
    })?;
    Ok(HostArray::new_vector(
        slice.as_ptr() as *mut u8,
        slice.len(),
        diffsol_c::ScalarType::F64,
    ))
}

pub(crate) fn pyarray2_to_host_f32(
    array: PyReadonlyArray2<'_, f32>,
) -> Result<HostArray, PyDiffsolError> {
    let view = array.as_array();
    Ok(HostArray::new(
        view.as_ptr() as *mut u8,
        view.shape().to_vec(),
        byte_strides(view.strides(), size_of::<f32>())?,
        diffsol_c::ScalarType::F32,
    ))
}

pub(crate) fn pyarray2_to_host_f64(
    array: PyReadonlyArray2<'_, f64>,
) -> Result<HostArray, PyDiffsolError> {
    let view = array.as_array();
    Ok(HostArray::new(
        view.as_ptr() as *mut u8,
        view.shape().to_vec(),
        byte_strides(view.strides(), size_of::<f64>())?,
        diffsol_c::ScalarType::F64,
    ))
}

/// Converts a 2D NumPy array of f64 to an owned Array2<f32>
pub(crate) fn pyarray2_to_owned_f32_host(
    array: PyReadonlyArray2<'_, f64>,
) -> Result<(Array2<f32>, HostArray), PyDiffsolError> {
    let owned = array.as_array().mapv(|value| value as f32);
    let host = HostArray::new(
        owned.as_ptr() as *mut u8,
        owned.shape().to_vec(),
        byte_strides(owned.strides(), size_of::<f32>())?,
        diffsol_c::ScalarType::F32,
    );
    Ok((owned, host))
}

/// Converts a 2D NumPy array of f32 to an owned Array2<f64>
pub(crate) fn pyarray2_to_owned_f64_host(
    array: PyReadonlyArray2<'_, f32>,
) -> Result<(Array2<f64>, HostArray), PyDiffsolError> {
    let owned = array.as_array().mapv(|value| value as f64);
    let host = HostArray::new(
        owned.as_ptr() as *mut u8,
        owned.shape().to_vec(),
        byte_strides(owned.strides(), size_of::<f64>())?,
        diffsol_c::ScalarType::F64,
    );
    Ok((owned, host))
}

/// Copies a host array to a PyArray in Python wrapped in a PyAny
///
/// Note: only supports 1D and 2D arrays of f32 and f64, and will return an error for unsupported types or dimensions
pub(crate) fn host_array_to_py<'py>(
    py: Python<'py>,
    array: HostArray,
) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
    // for 1D arrays, we convert to a slice and then COPY (to_pyarray) to a new PyArray 
    if let Ok(values) = array.as_slice::<f32>() {
        return Ok(values.to_pyarray(py).into_any());
    }
    if let Ok(values) = array.as_slice::<f64>() {
        return Ok(values.to_pyarray(py).into_any());
    }
    
    // for 2D arrays, we convert to an ArrayView and then COPY (to_pyarray) to a new PyArray
    if let Ok(values) = array.as_array::<f32>() {
        return Ok(values.to_pyarray(py).into_any());
    }
    if let Ok(values) = array.as_array::<f64>() {
        return Ok(values.to_pyarray(py).into_any());
    }
    
    // anything else is unsupported and we return an error
    Err(PyDiffsolError::Conversion(
        "Unsupported host array returned by diffsol-c".to_string(),
    ))
}

pub(crate) fn host_array_vec_to_py<'py>(
    py: Python<'py>,
    arrays: Vec<HostArray>,
) -> Result<Vec<Bound<'py, PyAny>>, PyDiffsolError> {
    arrays
        .into_iter()
        .map(|array| host_array_to_py(py, array))
        .collect()
}
