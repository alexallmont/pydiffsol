use std::sync::{Arc, Mutex};
use pyo3::prelude::*;
use crate::{config::ConfigWrapper, enums::*};

#[pyclass]
struct Ode {
    mtype: MatrixType,
}

#[pyclass]
#[pyo3(name = "Ode")]
#[derive(Clone)]
pub struct OdeWrapper(Arc<Mutex<Ode>>);

#[pymethods]
impl OdeWrapper {
    #[new]
    fn new(code: &str, mtype: MatrixType) -> PyResult<Self> {
        Ok(OdeWrapper(Arc::new(Mutex::new(
            Ode {
                mtype: mtype
            }
        ))))
    }

    // FIXME skeleton code, wrong arguments
    #[pyo3(signature=(numpy_array, config = ConfigWrapper::new()))]
    fn solve<'py>(
        slf: PyRefMut<'py, Self>,
        numpy_array: &str,
        config: ConfigWrapper
    ) -> PyResult<String> {
        let guard = slf.0.lock().unwrap();
        Ok("numpy_array".to_string())
    }
}

