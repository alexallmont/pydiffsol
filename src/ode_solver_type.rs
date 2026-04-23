// Wrap diffsol-c ode solver type - such as bdf or esdirk34 - with Pytho
// enum, and provide dynamic dispatch to underlying solve methods based on value.

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyType},
};

#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OdeSolverType {
    #[pyo3(name = "bdf")]
    Bdf,

    #[pyo3(name = "esdirk34")]
    Esdirk34,

    #[pyo3(name = "tr_bdf2")]
    TrBdf2,

    #[pyo3(name = "tsit45")]
    Tsit45,
}

impl OdeSolverType {
    pub(crate) fn all_enums() -> Vec<Self> {
        vec![Self::Bdf, Self::Esdirk34, Self::TrBdf2, Self::Tsit45]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            Self::Bdf => "bdf",
            Self::Esdirk34 => "esdirk34",
            Self::TrBdf2 => "tr_bdf2",
            Self::Tsit45 => "tsit45",
        }
    }
}

impl From<OdeSolverType> for diffsol_c::OdeSolverType {
    fn from(value: OdeSolverType) -> Self {
        match value {
            OdeSolverType::Bdf => diffsol_c::OdeSolverType::Bdf,
            OdeSolverType::Esdirk34 => diffsol_c::OdeSolverType::Esdirk34,
            OdeSolverType::TrBdf2 => diffsol_c::OdeSolverType::TrBdf2,
            OdeSolverType::Tsit45 => diffsol_c::OdeSolverType::Tsit45,
        }
    }
}

impl From<diffsol_c::OdeSolverType> for OdeSolverType {
    fn from(value: diffsol_c::OdeSolverType) -> Self {
        match value {
            diffsol_c::OdeSolverType::Bdf => OdeSolverType::Bdf,
            diffsol_c::OdeSolverType::Esdirk34 => OdeSolverType::Esdirk34,
            diffsol_c::OdeSolverType::TrBdf2 => OdeSolverType::TrBdf2,
            diffsol_c::OdeSolverType::Tsit45 => OdeSolverType::Tsit45,
        }
    }
}

#[pymethods]
impl OdeSolverType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "bdf" => Ok(Self::Bdf),
            "esdirk34" => Ok(Self::Esdirk34),
            "tr_bdf2" => Ok(Self::TrBdf2),
            "tsit45" => Ok(Self::Tsit45),
            _ => Err(PyValueError::new_err("Invalid OdeSolverType value")),
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
            Self::Bdf => 0,
            Self::Esdirk34 => 1,
            Self::TrBdf2 => 2,
            Self::Tsit45 => 3,
        }
    }
}
