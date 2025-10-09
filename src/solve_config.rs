use pyo3::prelude::*;
use crate::solver_method::SolverMethod;
use crate::solver_type::SolverType;

#[pyclass]
#[derive(Clone)]
pub(crate) struct SolveConfig {
    #[pyo3(get, set)]
    pub(crate) method: SolverMethod,
    #[pyo3(get, set)]
    pub(crate) linear_solver: SolverType,
}

impl SolveConfig {
    pub fn new(method: SolverMethod, linear_solver: SolverType) -> Self {
        SolveConfig {
            method,
            linear_solver,
        }
    }
}