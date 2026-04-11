use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    PyErr,
};

pub enum PyDiffsolError {
    Diffsol(diffsol_c::DiffsolJsError),
    Conversion(String),
    Python(PyErr),
}

impl From<PyDiffsolError> for PyErr {
    fn from(err: PyDiffsolError) -> Self {
        match err {
            PyDiffsolError::Diffsol(err) => PyRuntimeError::new_err(err.to_string()),
            PyDiffsolError::Conversion(msg) => PyValueError::new_err(msg),
            PyDiffsolError::Python(err) => err,
        }
    }
}

impl From<diffsol_c::DiffsolJsError> for PyDiffsolError {
    fn from(err: diffsol_c::DiffsolJsError) -> Self {
        PyDiffsolError::Diffsol(err)
    }
}

impl From<PyErr> for PyDiffsolError {
    fn from(err: PyErr) -> Self {
        PyDiffsolError::Python(err)
    }
}
