use std::sync::{Arc, Mutex};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use crate::enums::*;

#[pyclass]
pub(crate) struct Config {
    pub(crate) method: SolverMethod,
    pub(crate) linear_solver: SolverType,
    pub(crate) rtol: f64,
}

#[pyclass]
#[pyo3(name = "Config")]
#[derive(Clone)]
pub(crate) struct ConfigWrapper(pub(crate) Arc<Mutex<Config>>);

#[pymethods]
impl ConfigWrapper {
    #[new]
    pub fn new() -> Self {
        ConfigWrapper(Arc::new(Mutex::new(
            Config {
                method: SolverMethod::Bdf,
                linear_solver: SolverType::Lu,
                rtol: 1e-6,
            }
        )))
    }

    #[setter]
    fn set_method(&self, method: SolverMethod) -> PyResult<()> {
        let mut guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        guard.method = method;
        Ok(())
    }

    #[setter]
    fn set_linear_solver(&self, linear_solver: SolverType) -> PyResult<()> {
        let mut guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        guard.linear_solver = linear_solver;
        Ok(())
    }

    #[setter]
    fn set_rtol(&self, rtol: f64) -> PyResult<()> {
        let mut guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        guard.rtol = rtol;
        Ok(())
    }
}