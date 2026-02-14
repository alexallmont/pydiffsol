use std::sync::{Arc, Mutex};

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    py_state::PySolution,
    py_types::{PyUntypedArray1, PyUntypedArray2},
};

#[pyclass]
pub(crate) struct Solution {
    py_solution: Box<dyn PySolution>,
}

unsafe impl Send for Solution {}
unsafe impl Sync for Solution {}

#[pyclass]
#[pyo3(name = "Solution")]
#[derive(Clone)]
pub struct SolutionWrapper(Arc<Mutex<Solution>>);

impl SolutionWrapper {
    pub(crate) fn new(py_solution: Box<dyn PySolution>) -> Self {
        Self(Arc::new(Mutex::new(Solution { py_solution })))
    }

    fn guard(&self) -> PyResult<std::sync::MutexGuard<'_, Solution>> {
        self.0
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Solution mutex poisoned"))
    }
}

#[pymethods]
impl SolutionWrapper {
    #[getter]
    fn get_ys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyUntypedArray2>> {
        let guard = self.guard()?;
        Ok(guard.py_solution.get_ys(py))
    }

    #[getter]
    fn get_ts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyUntypedArray1>> {
        let guard = self.guard()?;
        Ok(guard.py_solution.get_ts(py))
    }

    #[getter]
    fn get_sens<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyUntypedArray2>>> {
        let guard = self.guard()?;
        Ok(guard.py_solution.get_sens(py))
    }

    #[setter]
    fn set_current_state(&self, y: Vec<f64>) -> PyResult<()> {
        let mut guard = self.guard()?;
        guard.py_solution.set_state_y(&y);
        Ok(())
    }

    #[getter]
    fn get_current_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyUntypedArray1>> {
        let guard = self.guard()?;
        Ok(guard.py_solution.get_state_y(py))
    }
}
