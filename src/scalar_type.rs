// Data type Python enum

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyType},
};

#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarType {
    #[pyo3(name = "f32")]
    F32,

    #[pyo3(name = "f64")]
    F64,
}

impl ScalarType {
    pub(crate) fn all_enums() -> Vec<ScalarType> {
        vec![ScalarType::F32, ScalarType::F64]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
        }
    }
}

#[pymethods]
impl ScalarType {
    /// Create ScalarType from string name
    /// :param name: string representation of data type
    /// :return: valid ScalarType or exception if name is invalid
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        match name {
            "f32" => Ok(ScalarType::F32),
            "f64" => Ok(ScalarType::F64),
            _ => Err(PyValueError::new_err("Invalid ScalarType value")),
        }
    }

    /// Get all available data types
    /// :return: list of ScalarType
    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), ScalarType::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            ScalarType::F32 => 0,
            ScalarType::F64 => 1,
        }
    }
}
