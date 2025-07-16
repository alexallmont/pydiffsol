
use std::sync::{Arc, Mutex};

use crate::{config::ConfigWrapper, enums::*};

use diffsol::{MatrixCommon, LlvmModule, OdeEquations, OdeSolverMethod, Vector};
use diffsol::error::DiffsolError;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::DMatrix;
use numpy::{ToPyArray, PyReadonlyArray1, PyArray1, PyArray2};
use numpy::ndarray::{Array1, ArrayView2, ShapeBuilder};

#[pyclass]
struct Ode {
    code: String,
    matrix_type: MatrixType,
}

#[pyclass]
#[pyo3(name = "Ode")]
#[derive(Clone)]
pub struct OdeWrapper(Arc<Mutex<Ode>>);

// FIXME separate file
struct PyDiffsolError(DiffsolError);

impl From<PyDiffsolError> for PyErr {
    fn from(error: PyDiffsolError) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

impl From<DiffsolError> for PyDiffsolError {
    fn from(other: DiffsolError) -> Self {
        Self(other)
    }
}

// FIXME conversion in separate file
pub trait MatrixToPy<'py> {
    fn to_pyarray_view(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>>;
}

impl<'py> MatrixToPy<'py> for DMatrix<f64> {
    fn to_pyarray_view(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let view = unsafe {
            ArrayView2::from_shape_ptr(
                self.shape().strides(self.strides()),
                self.as_ptr()
            )
        };
        view.to_pyarray(py).into()
    }
}

#[pymethods]
impl OdeWrapper {
    #[new]
    fn new(code: &str, matrix_type: MatrixType) -> PyResult<Self> {
        Ok(OdeWrapper(Arc::new(Mutex::new(
            Ode {
                code: code.to_string(),
                matrix_type: matrix_type
            }
        ))))
    }

    #[pyo3(signature=(params, time, config = ConfigWrapper::new()))]
    fn solve<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        time: f64,
        config: ConfigWrapper
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        let self_guard = slf.0.lock().unwrap();
        match self_guard.matrix_type {
            MatrixType::NalgebraDenseF64 => {
                type M = diffsol::NalgebraMat<f64>;
                type V = diffsol::NalgebraVec<f64>;
                type C = diffsol::NalgebraContext;
                type T = <M as MatrixCommon>::T;
                type LS = diffsol::NalgebraLU<f64>;

                let params = V::from_slice(params.as_array().as_slice().unwrap(), C::default());
                let mut problem = diffsol::OdeBuilder::<M>::new().build_from_diffsl::<LlvmModule>(self_guard.code.as_str())?;
                problem.eqn.set_params(&params);

                let config_guard = config.0.lock().unwrap();
                let (ys, ts): (M, Vec<T>) = match config_guard.method {
                    SolverMethod::Bdf => {
                        let mut solver = problem.bdf::<LS>()?; // FIXME swap out LS at runtime based on config.linear_solver
                        solver.solve(time)?
                    },
                    SolverMethod::Sdirk => {
                        // FIXME handle Sdirk
                        let mut solver = problem.bdf::<LS>()?;
                        solver.solve(time)?
                    },
                };
                let ts_arr = Array1::from(ts);
                Ok((
                    ys.inner().to_pyarray_view(slf.py()),
                    PyArray1::from_owned_array(slf.py(), ts_arr)
                ))
            },
            MatrixType::FaerSparseF64 => {
                // FIXME handle Faer
                Err(DiffsolError::Other("FaerSparseF64 not yet supported".to_string()).into())
            }
        }
    }
}

