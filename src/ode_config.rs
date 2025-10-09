use pyo3::prelude::*;

use crate::matrix_type::MatrixType;

#[pyclass]
#[derive(Clone)]
pub(crate) struct OdeConfig {
    #[pyo3(get, set)]
    pub(crate) rtol: f64,
    #[pyo3(get, set)]
    pub(crate) atol: f64,
    #[pyo3(get, set)]
    pub(crate) matrix_type: MatrixType,
}