// Wrap diffsol-c solution arrays with Python properties returning NumPy arrays.

use pyo3::{prelude::*, PyAny};
use pyo3_stub_gen::derive::{
    gen_stub_pyclass,
    gen_stub_pymethods,
};

use crate::error::PyDiffsolError;
use crate::host_array::{host_array_to_py, host_array_vec_to_py};

#[gen_stub_pyclass]
#[pyclass]
#[pyo3(from_py_object, name = "Solution")]
#[derive(Clone)]
pub struct SolutionWrapper(diffsol_c::SolutionWrapper);

impl SolutionWrapper {
    pub(crate) fn new(inner: diffsol_c::SolutionWrapper) -> Self {
        Self(inner)
    }

    pub(crate) fn inner(&self) -> &diffsol_c::SolutionWrapper {
        &self.0
    }
}

// Note the gen_stub override_return_type ensures the autocomplete shows a numpy array and not an any type
#[gen_stub_pymethods]
#[pymethods]
impl SolutionWrapper {
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[typing.Any]", imports = ("numpy.typing", "typing")))]
    #[getter]
    fn get_ys<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(py, self.0.get_ys()?)
    }

    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[typing.Any]", imports = ("numpy.typing", "typing")))]
    #[getter]
    fn get_ts<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(py, self.0.get_ts()?)
    }

    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[typing.Any]", imports = ("numpy.typing", "typing")))]
    #[getter]
    fn get_sens<'py>(&self, py: Python<'py>) -> Result<Vec<Bound<'py, PyAny>>, PyDiffsolError> {
        host_array_vec_to_py(py, self.0.get_sens()?)
    }
}
