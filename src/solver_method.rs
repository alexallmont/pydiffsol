use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyType};

#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolverMethod {
    #[pyo3(name = "bdf")]
    Bdf,

    #[pyo3(name = "esdirk34")]
    Esdirk34,

    #[pyo3(name = "tr_bdf2")]
    TrBdf2,

    #[pyo3(name = "tsit45")]
    Tsit45,
}

impl SolverMethod {
    pub(crate) fn all_enums() -> Vec<SolverMethod> {
        vec![
            SolverMethod::Bdf,
            SolverMethod::Esdirk34,
            SolverMethod::TrBdf2,
            SolverMethod::Tsit45,
        ]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            SolverMethod::Bdf => "bdf",
            SolverMethod::Esdirk34 => "esdirk34",
            SolverMethod::TrBdf2 => "tr_bdf2",
            SolverMethod::Tsit45 => "tsit45",
        }
    }
}

#[pymethods]
impl SolverMethod {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "bdf" => Ok(SolverMethod::Bdf),
            "esdirk34" => Ok(SolverMethod::Esdirk34),
            "tr_bdf2" => Ok(SolverMethod::TrBdf2),
            "tsit45" => Ok(SolverMethod::Tsit45),
            _ => Err(PyValueError::new_err("Invalid SolverMethod value")),
        }
    }

    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), SolverMethod::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            SolverMethod::Bdf => 0,
            SolverMethod::Esdirk34 => 1,
            SolverMethod::TrBdf2 => 2,
            SolverMethod::Tsit45 => 3,
        }
    }
}
