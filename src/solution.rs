use pyo3::{prelude::*, PyAny};

use crate::error::PyDiffsolError;
use crate::host_array::{host_array_to_py, host_array_vec_to_py};

#[pyclass]
#[pyo3(name = "Solution")]
#[derive(Clone)]
pub struct SolutionWrapper(diffsol_c::SolutionWrapper);

impl SolutionWrapper {
    pub(crate) fn new(inner: diffsol_c::SolutionWrapper) -> Self {
        Self(inner)
    }
}

#[pymethods]
impl SolutionWrapper {
    #[getter]
    fn get_ys<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(py, self.0.get_ys()?)
    }

    #[getter]
    fn get_ts<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(py, self.0.get_ts()?)
    }

    #[getter]
    fn get_sens<'py>(&self, py: Python<'py>) -> Result<Vec<Bound<'py, PyAny>>, PyDiffsolError> {
        host_array_vec_to_py(py, self.0.get_sens()?)
    }
}
