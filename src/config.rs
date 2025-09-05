use std::sync::{Arc, Mutex};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use crate::matrix_type::MatrixType;
use crate::solver_method::SolverMethod;
use crate::solver_type::SolverType;

#[pyclass]
pub(crate) struct Config {
    pub(crate) method: SolverMethod,
    pub(crate) linear_solver: SolverType,
    pub(crate) rtol: f64,
}

impl Config {
    // See ConfigWrapper::solver_for_matrix_type docstring for purpose of this code
    pub fn solver_for_matrix_type(&self, matrix_type: MatrixType) -> SolverType {
        if self.linear_solver == SolverType::Default {
            if self.method != SolverMethod::Tsit45 {
                if matrix_type == MatrixType::FaerSparseF64 {
                    // Klu is specific to faer sparse
                    return SolverType::Klu;
                } else {
                    // Faer dense and nalgebra default to Lu
                    return SolverType::Lu;
                }
            }
        }
        self.linear_solver
    }
}

#[pyclass]
#[pyo3(name = "Config")]
#[derive(Clone)]
pub(crate) struct ConfigWrapper(pub(crate) Arc<Mutex<Config>>);

#[pymethods]
impl ConfigWrapper {
    #[new]
    #[pyo3(signature=(method=SolverMethod::Bdf, linear_solver=SolverType::Default, rtol=1e-6))]
    pub fn new(
        method: SolverMethod,
        linear_solver: SolverType,
        rtol: f64
    ) -> Self {
        ConfigWrapper(Arc::new(Mutex::new(
            Config {
                method: method,
                linear_solver: linear_solver,
                rtol: rtol,
            }
        )))
    }

    #[getter]
    fn get_method(&self) -> PyResult<SolverMethod> {
        let guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        Ok(guard.method)
    }

    #[setter]
    fn set_method(&self, method: SolverMethod) -> PyResult<()> {
        let mut guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        guard.method = method;
        Ok(())
    }

    #[getter]
    fn get_linear_solver(&self) -> PyResult<SolverType> {
        let guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        Ok(guard.linear_solver)
    }

    #[setter]
    fn set_linear_solver(&self, linear_solver: SolverType) -> PyResult<()> {
        let mut guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        guard.linear_solver = linear_solver;
        Ok(())
    }

    #[getter]
    fn get_rtol(&self) -> PyResult<f64> {
        let guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        Ok(guard.rtol)
    }

    #[setter]
    fn set_rtol(&self, rtol: f64) -> PyResult<()> {
        let mut guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        guard.rtol = rtol;
        Ok(())
    }

    /// Return the best possible solver type fit given the Config.
    ///
    /// This is particularly for the special case of Tsit45 which does not
    /// require a solver type like the others. But also allows the default
    /// config SolverType to be left to be chosen at runtime given the Ode
    /// matrix type, i.e. Klu for Faer sparse, and Lu for all other cases,
    /// unless Tsit45 in which case it is left as Default.
    ///
    /// If a non-default solver type is specified, then that overrides the
    /// all cases.
    fn solver_for_matrix_type(&self, matrix_type: MatrixType) -> PyResult<SolverType> {
        let guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        Ok(guard.solver_for_matrix_type(matrix_type))
    }

    fn __repr__(&self) -> PyResult<String> {
        let guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        Ok(format!(
            "Config(method={:?}, linear_solver={:?}, rtol={})",
            guard.method, guard.linear_solver, guard.rtol
        ))
    }
}