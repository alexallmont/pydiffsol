use std::mem::size_of;

use diffsol_c::host_array::HostArray;
use numpy::{
    ndarray::{Array2, ArrayView2},
    PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::{prelude::*, PyAny};

use crate::error::PyDiffsolError;

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

fn host_array_from_f64_view(view: ArrayView2<'_, f64>) -> Result<HostArray, PyDiffsolError> {
    Ok(HostArray::new(
        view.as_ptr() as *mut u8,
        view.shape().to_vec(),
        byte_strides(view.strides(), size_of::<f64>())?,
        diffsol_c::ScalarType::F64,
    ))
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

pub(crate) fn pyarray2_to_host_f64(
    array: PyReadonlyArray2<'_, f64>,
) -> Result<HostArray, PyDiffsolError> {
    host_array_from_f64_view(array.as_array())
}

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

pub(crate) fn host_array_to_py<'py>(
    py: Python<'py>,
    array: HostArray,
) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
    if let Ok(values) = array.as_slice::<f32>() {
        return Ok(values.to_vec().to_pyarray(py).into_any());
    }
    if let Ok(values) = array.as_slice::<f64>() {
        return Ok(values.to_vec().to_pyarray(py).into_any());
    }
    if let Ok(values) = array.as_array::<f32>() {
        return Ok(values.to_owned().to_pyarray(py).into_any());
    }
    if let Ok(values) = array.as_array::<f64>() {
        return Ok(values.to_owned().to_pyarray(py).into_any());
    }
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
