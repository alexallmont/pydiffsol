// Convert diffsol errors to custom pydiffsol error type

use diffsol::error::DiffsolError;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

pub enum PyDiffsolError {
    Diffsol(DiffsolError),
    Conversion(String),
}

impl From<PyDiffsolError> for PyErr {
    fn from(err: PyDiffsolError) -> Self {
        match err {
            PyDiffsolError::Diffsol(e) => PyRuntimeError::new_err(e.to_string()),
            PyDiffsolError::Conversion(msg) => PyValueError::new_err(msg),
        }
    }
}

impl From<DiffsolError> for PyDiffsolError {
    fn from(other: DiffsolError) -> Self {
        PyDiffsolError::Diffsol(other)
    }
}
