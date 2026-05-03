use pyo3::prelude::*;

use crate::error::PyDiffsolError;

#[pyclass]
#[derive(Clone)]
pub struct InitialConditionSolverOptions(diffsol_c::InitialConditionSolverOptions);

impl InitialConditionSolverOptions {
    pub(crate) fn new(inner: diffsol_c::InitialConditionSolverOptions) -> Self {
        Self(inner)
    }
}

#[pymethods]
impl InitialConditionSolverOptions {
    #[getter]
    fn get_use_linesearch(&self) -> Result<bool, PyDiffsolError> {
        Ok(self.0.get_use_linesearch()?)
    }

    #[setter]
    fn set_use_linesearch(&self, value: bool) -> Result<(), PyDiffsolError> {
        self.0.set_use_linesearch(value)?;
        Ok(())
    }

    #[getter]
    fn get_max_linesearch_iterations(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_max_linesearch_iterations()?)
    }

    #[setter]
    fn set_max_linesearch_iterations(&self, value: usize) -> Result<(), PyDiffsolError> {
        self.0.set_max_linesearch_iterations(value)?;
        Ok(())
    }

    #[getter]
    fn get_max_newton_iterations(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_max_newton_iterations()?)
    }

    #[setter]
    fn set_max_newton_iterations(&self, value: usize) -> Result<(), PyDiffsolError> {
        self.0.set_max_newton_iterations(value)?;
        Ok(())
    }

    #[getter]
    fn get_max_linear_solver_setups(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_max_linear_solver_setups()?)
    }

    #[setter]
    fn set_max_linear_solver_setups(&self, value: usize) -> Result<(), PyDiffsolError> {
        self.0.set_max_linear_solver_setups(value)?;
        Ok(())
    }

    #[getter]
    fn get_step_reduction_factor(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_step_reduction_factor()?)
    }

    #[setter]
    fn set_step_reduction_factor(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_step_reduction_factor(value)?;
        Ok(())
    }

    #[getter]
    fn get_armijo_constant(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_armijo_constant()?)
    }

    #[setter]
    fn set_armijo_constant(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_armijo_constant(value)?;
        Ok(())
    }
}
