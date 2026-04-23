// Wrap diffsol-c solver type with Python enum. This is the internal solver
// mechanism, either LU or KLU in diffsol, with default selecting whichever
// is most appropriate given the matrix type.

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyType},
};

/// Enumerates the possible linear solver types for diffsol
///
/// :attr default: use the solver's default linear solver choice, typically LU
/// :attr lu: use LU decomposition linear solver (dense or sparse as appropriate)
/// :attr klu: use KLU sparse linear solver
#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinearSolverType {
    #[pyo3(name = "default")]
    Default,

    #[pyo3(name = "lu")]
    Lu,

    #[pyo3(name = "klu")]
    Klu,
}

impl LinearSolverType {
    pub(crate) fn all_enums() -> Vec<Self> {
        vec![Self::Default, Self::Lu, Self::Klu]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            Self::Default => "default",
            Self::Lu => "lu",
            Self::Klu => "klu",
        }
    }
}

impl From<LinearSolverType> for diffsol_c::LinearSolverType {
    fn from(value: LinearSolverType) -> Self {
        match value {
            LinearSolverType::Default => diffsol_c::LinearSolverType::Default,
            LinearSolverType::Lu => diffsol_c::LinearSolverType::Lu,
            LinearSolverType::Klu => diffsol_c::LinearSolverType::Klu,
        }
    }
}

impl From<diffsol_c::LinearSolverType> for LinearSolverType {
    fn from(value: diffsol_c::LinearSolverType) -> Self {
        match value {
            diffsol_c::LinearSolverType::Default => LinearSolverType::Default,
            diffsol_c::LinearSolverType::Lu => LinearSolverType::Lu,
            diffsol_c::LinearSolverType::Klu => LinearSolverType::Klu,
        }
    }
}

#[pymethods]
impl LinearSolverType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "default" => Ok(Self::Default),
            "lu" => Ok(Self::Lu),
            "klu" => Ok(Self::Klu),
            _ => Err(PyValueError::new_err("Invalid LinearSolverType value")),
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
            Self::Default => 0,
            Self::Lu => 1,
            Self::Klu => 2,
        }
    }
}
