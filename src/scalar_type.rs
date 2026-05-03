// Python enum for underlying data type used in matrix types. Supports 32 or 64 bit float.

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
    pub(crate) fn all_enums() -> Vec<Self> {
        vec![Self::F32, Self::F64]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl From<ScalarType> for diffsol_c::ScalarType {
    fn from(value: ScalarType) -> Self {
        match value {
            ScalarType::F32 => diffsol_c::ScalarType::F32,
            ScalarType::F64 => diffsol_c::ScalarType::F64,
        }
    }
}

impl From<diffsol_c::ScalarType> for ScalarType {
    fn from(value: diffsol_c::ScalarType) -> Self {
        match value {
            diffsol_c::ScalarType::F32 => ScalarType::F32,
            diffsol_c::ScalarType::F64 => ScalarType::F64,
        }
    }
}

#[pymethods]
impl ScalarType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err(PyValueError::new_err("Invalid ScalarType value")),
        }
    }

    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), Self::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            Self::F32 => 0,
            Self::F64 => 1,
        }
    }
}
