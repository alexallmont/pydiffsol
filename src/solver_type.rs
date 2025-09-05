use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyType};

#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolverType {
    #[pyo3(name = "default")]
    Default,

    #[pyo3(name = "lu")]
    Lu,

    #[pyo3(name = "klu")]
    Klu,
}

impl SolverType {
    pub(crate) fn all_enums() -> Vec<SolverType> {
        vec![
            SolverType::Default,
            SolverType::Lu,
            SolverType::Klu,
        ]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            SolverType::Default => "default",
            SolverType::Lu => "lu",
            SolverType::Klu => "klu",
        }
    }
}

#[pymethods]
impl SolverType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "default" => Ok(SolverType::Default),
            "lu" => Ok(SolverType::Lu),
            "klu" => Ok(SolverType::Klu),
            _ => Err(PyValueError::new_err("Invalid SolverType value")),
        }
    }

    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), SolverType::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            SolverType::Default => 0,
            SolverType::Lu => 1,
            SolverType::Klu => 2,
        }
    }
}
