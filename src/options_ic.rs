use std::sync::{Arc, Mutex};

use crate::ode::Ode;

use pyo3::prelude::*;

#[pyclass]
pub struct InitialConditionSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl InitialConditionSolverOptions {
    pub(crate) fn new(ode: Arc<Mutex<Ode>>) -> Self {
        Self { ode }
    }
    fn guard(&self) -> PyResult<std::sync::MutexGuard<'_, Ode>> {
        self.ode.lock().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Failed to acquire lock on Ode object",
            )
        })
    }
}

#[pymethods]
impl InitialConditionSolverOptions {
    #[getter]
    fn get_use_linesearch(&self) -> PyResult<bool> {
        Ok(self.guard()?.py_solve.ic_use_linesearch())
    }
    #[setter]
    fn set_use_linesearch(&self, value: bool) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_use_linesearch(value);
        Ok(())
    }
    #[getter]
    fn get_max_linesearch_iterations(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ic_max_linesearch_iterations())
    }
    #[setter]
    fn set_max_linesearch_iterations(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ic_max_linesearch_iterations(value);
        Ok(())
    }
    #[getter]
    fn get_max_newton_iterations(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ic_max_newton_iterations())
    }
    #[setter]
    fn set_max_newton_iterations(&self, value: usize) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_max_newton_iterations(value);
        Ok(())
    }
    #[getter]
    fn get_max_linear_solver_setups(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ic_max_linear_solver_setups())
    }
    #[setter]
    fn set_max_linear_solver_setups(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ic_max_linear_solver_setups(value);
        Ok(())
    }
    #[getter]
    fn get_step_reduction_factor(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.ic_step_reduction_factor())
    }
    #[setter]
    fn set_step_reduction_factor(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_step_reduction_factor(value);
        Ok(())
    }
    #[getter]
    fn get_armijo_constant(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.ic_armijo_constant())
    }
    #[setter]
    fn set_armijo_constant(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_armijo_constant(value);
        Ok(())
    }
}
