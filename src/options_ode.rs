use std::sync::{Arc, Mutex};

use crate::ode::Ode;

use pyo3::prelude::*;

#[pyclass]
pub struct OdeSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl OdeSolverOptions {
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
impl OdeSolverOptions {
    #[getter]
    fn get_max_nonlinear_solver_iterations(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ode_max_nonlinear_solver_iterations())
    }
    #[setter]
    fn set_max_nonlinear_solver_iterations(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_max_nonlinear_solver_iterations(value);
        Ok(())
    }
    #[getter]
    fn get_max_error_test_failures(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ode_max_error_test_failures())
    }
    #[setter]
    fn set_max_error_test_failures(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_max_error_test_failures(value);
        Ok(())
    }
    #[getter]
    fn get_update_jacobian_after_steps(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ode_update_jacobian_after_steps())
    }
    #[setter]
    fn set_update_jacobian_after_steps(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_update_jacobian_after_steps(value);
        Ok(())
    }
    #[getter]
    fn get_update_rhs_jacobian_after_steps(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ode_update_rhs_jacobian_after_steps())
    }
    #[setter]
    fn set_update_rhs_jacobian_after_steps(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_update_rhs_jacobian_after_steps(value);
        Ok(())
    }
    #[getter]
    fn get_threshold_to_update_jacobian(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.ode_threshold_to_update_jacobian())
    }
    #[setter]
    fn set_threshold_to_update_jacobian(&self, value: f64) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_threshold_to_update_jacobian(value);
        Ok(())
    }
    #[getter]
    fn get_threshold_to_update_rhs_jacobian(&self) -> PyResult<f64> {
        Ok(self
            .guard()?
            .py_solve
            .ode_threshold_to_update_rhs_jacobian())
    }
    #[setter]
    fn set_threshold_to_update_rhs_jacobian(&self, value: f64) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_threshold_to_update_rhs_jacobian(value);
        Ok(())
    }
    #[getter]
    fn get_min_timestep(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.ode_min_timestep())
    }
    #[setter]
    fn set_min_timestep(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_ode_min_timestep(value);
        Ok(())
    }
}
