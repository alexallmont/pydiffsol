// Data type Python enum

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};

// FIXME rename this to ScalarType and scalar_type in Python to reflect diffsol API.

#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataType {
    #[pyo3(name = "f32")]
    F32,

    #[pyo3(name = "f64")]
    F64,
}

impl DataType {
    pub(crate) fn all_enums() -> Vec<DataType> {
        vec![
            DataType::F32,
            DataType::F64,
        ]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            DataType::F32 => "f32",
            DataType::F64 => "f64",
        }
    }
}

#[pymethods]
impl DataType {
    /// Create DataType from string name
    /// :param name: string representation of data type
    /// :return: valid DataType or exception if name is invalid
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        match name {
            "f32" => Ok(DataType::F32),
            "f64" => Ok(DataType::F64),
            _ => Err(PyValueError::new_err("Invalid DataType value")),
        }
    }

    /// Get all available data types
    /// :return: list of DataType
    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), DataType::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            DataType::F32 => 0,
            DataType::F64 => 1,
        }
    }
}
