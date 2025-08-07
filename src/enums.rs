use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;

#[pyclass(eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum MatrixType {
    #[pyo3(name = "nalgebra_dense_f64")]
    NalgebraDenseF64,

    #[pyo3(name = "faer_sparse_f64")]
    FaerSparseF64,
}

#[pymethods]
impl MatrixType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "nalgebra_dense_f64" => Ok(MatrixType::NalgebraDenseF64),
            "faer_sparse_f64" => Ok(MatrixType::FaerSparseF64),
            _ => Err(PyValueError::new_err("Invalid MatrixType value")),
        }
    }
}

#[pyclass(eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum SolverType {
    #[pyo3(name = "lu")]
    Lu,

    #[pyo3(name = "klu")]
    Klu,
}

#[pymethods]
impl SolverType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "lu" => Ok(SolverType::Lu),
            "klu" => Ok(SolverType::Klu),
            _ => Err(PyValueError::new_err("Invalid SolverType value")),
        }
    }
}

#[pyclass(eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum SolverMethod {
    #[pyo3(name = "bdf")]
    Bdf,

    #[pyo3(name = "esdirk34")]
    Esdirk34,
}

#[pymethods]
impl SolverMethod {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "bdf" => Ok(SolverMethod::Bdf),
            "esdirk34" => Ok(SolverMethod::Esdirk34),
            _ => Err(PyValueError::new_err("Invalid SolverMethod value")),
        }
    }
}
