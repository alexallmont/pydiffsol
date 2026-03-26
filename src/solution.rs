use std::sync::{Arc, Mutex};

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    error::PyDiffsolError,
    py_solution::PySolution,
    py_types::{PyUntypedArray1, PyUntypedArray2},
};

#[pyclass]
pub(crate) struct Solution {
    py_solution: Option<Box<dyn PySolution>>,
}

#[pyclass]
#[pyo3(name = "Solution")]
#[derive(Clone)]
pub struct SolutionWrapper(Arc<Mutex<Solution>>);

impl SolutionWrapper {
    pub(crate) fn new(py_solution: Box<dyn PySolution>) -> Self {
        Self(Arc::new(Mutex::new(Solution {
            py_solution: Some(py_solution),
        })))
    }

    fn guard(&self) -> PyResult<std::sync::MutexGuard<'_, Solution>> {
        self.0
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Solution mutex poisoned"))
    }

    /// Takes the PySolution out of the wrapper, leaving None in its place. This is used to temporarily
    /// take ownership of the PySolution for operations that require mutable access, while ensuring that the wrapper
    /// is left in a consistent state.
    ///
    /// The caller is responsible for putting the PySolution back into the wrapper
    /// after the operation is complete using `replace_py_solution`. This is only
    /// relevant for operations from python that might raise an exception, as you
    /// want to ensure that the PySolution is put back into the wrapper if an error
    /// occurs, since the user will expect that the solution input argument is still
    /// valid after an error is raised.
    pub(crate) fn take_py_solution(&self) -> Result<Box<dyn PySolution>, PyDiffsolError> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| PyDiffsolError::Conversion("Solution mutex poisoned".to_string()))?;
        guard
            .py_solution
            .take()
            .ok_or_else(|| PyDiffsolError::Conversion("Solution payload missing".to_string()))
    }

    pub(crate) fn replace_py_solution(
        &self,
        py_solution: Box<dyn PySolution>,
    ) -> Result<(), PyDiffsolError> {
        let mut guard = self
            .0
            .lock()
            .map_err(|_| PyDiffsolError::Conversion("Solution mutex poisoned".to_string()))?;
        guard.py_solution = Some(py_solution);
        Ok(())
    }
}

#[pymethods]
impl SolutionWrapper {
    #[getter]
    fn get_ys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyUntypedArray2>> {
        let guard = self.guard()?;
        let py_solution = guard
            .py_solution
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Solution payload missing"))?;
        Ok(py_solution.get_ys(py))
    }

    #[getter]
    fn get_ts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyUntypedArray1>> {
        let guard = self.guard()?;
        let py_solution = guard
            .py_solution
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Solution payload missing"))?;
        Ok(py_solution.get_ts(py))
    }

    #[getter]
    fn get_sens<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyUntypedArray2>>> {
        let guard = self.guard()?;
        let py_solution = guard
            .py_solution
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Solution payload missing"))?;
        Ok(py_solution.get_sens(py))
    }

    #[setter]
    fn set_current_state(&self, y: Vec<f64>) -> PyResult<()> {
        let mut guard = self.guard()?;
        let py_solution = guard
            .py_solution
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Solution payload missing"))?;
        py_solution.set_state_y(&y)?;
        Ok(())
    }

    #[getter]
    fn get_current_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyUntypedArray1>> {
        let guard = self.guard()?;
        let py_solution = guard
            .py_solution
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Solution payload missing"))?;
        Ok(py_solution.get_state_y(py))
    }
}
