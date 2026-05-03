use pyo3::prelude::*;

use crate::error::PyDiffsolError;

#[pyclass]
#[derive(Clone)]
pub struct OdeSolverOptions(diffsol_c::OdeSolverOptions);

impl OdeSolverOptions {
    pub(crate) fn new(inner: diffsol_c::OdeSolverOptions) -> Self {
        Self(inner)
    }
}

#[pymethods]
impl OdeSolverOptions {
    #[getter]
    fn get_max_nonlinear_solver_iterations(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_max_nonlinear_solver_iterations()?)
    }

    #[setter]
    fn set_max_nonlinear_solver_iterations(&self, value: usize) -> Result<(), PyDiffsolError> {
        self.0.set_max_nonlinear_solver_iterations(value)?;
        Ok(())
    }

    #[getter]
    fn get_max_error_test_failures(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_max_error_test_failures()?)
    }

    #[setter]
    fn set_max_error_test_failures(&self, value: usize) -> Result<(), PyDiffsolError> {
        self.0.set_max_error_test_failures(value)?;
        Ok(())
    }

    #[getter]
    fn get_update_jacobian_after_steps(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_update_jacobian_after_steps()?)
    }

    #[setter]
    fn set_update_jacobian_after_steps(&self, value: usize) -> Result<(), PyDiffsolError> {
        self.0.set_update_jacobian_after_steps(value)?;
        Ok(())
    }

    #[getter]
    fn get_update_rhs_jacobian_after_steps(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_update_rhs_jacobian_after_steps()?)
    }

    #[setter]
    fn set_update_rhs_jacobian_after_steps(&self, value: usize) -> Result<(), PyDiffsolError> {
        self.0.set_update_rhs_jacobian_after_steps(value)?;
        Ok(())
    }

    #[getter]
    fn get_threshold_to_update_jacobian(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_threshold_to_update_jacobian()?)
    }

    #[setter]
    fn set_threshold_to_update_jacobian(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_threshold_to_update_jacobian(value)?;
        Ok(())
    }

    #[getter]
    fn get_threshold_to_update_rhs_jacobian(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_threshold_to_update_rhs_jacobian()?)
    }

    #[setter]
    fn set_threshold_to_update_rhs_jacobian(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_threshold_to_update_rhs_jacobian(value)?;
        Ok(())
    }

    #[getter]
    fn get_min_timestep(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_min_timestep()?)
    }

    #[setter]
    fn set_min_timestep(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_min_timestep(value)?;
        Ok(())
    }
}
