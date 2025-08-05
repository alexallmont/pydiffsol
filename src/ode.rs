
use std::sync::{Arc, Mutex};

use crate::config::{Config, ConfigWrapper};
use crate::convert::MatrixToPy;
use crate::enums::{MatrixType, SolverType, SolverMethod};
use crate::error::PyDiffsolError;
use crate::jit::JitModule;

use pyo3::prelude::*;

use diffsol::{OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem};
use diffsol::{MatrixCommon, matrix::MatrixHost};
use diffsol::error::DiffsolError;
use diffsol::{NalgebraMat, NalgebraVec, NalgebraLU};
use diffsol::{FaerMat, FaerVec, FaerLU};
use diffsol::Vector; // for from_slice
use diffsol::Op; // For nparams

use numpy::{PyReadonlyArray1, PyArray1, PyArray2};
use numpy::ndarray::Array1;

#[pyclass]
struct Ode {
    code: String,
    matrix_type: MatrixType,
}

#[pyclass]
#[pyo3(name = "Ode")]
#[derive(Clone)]
pub struct OdeWrapper(Arc<Mutex<Ode>>);

// Construct a diffsol problem for particular matrix type, given diffsl code,
// pydiffsol config and params.
fn build_diffsl<M, V> (code: &str, config: &Config, params: &[f64]) ->
    Result<OdeSolverProblem<diffsol::DiffSl<M, JitModule>>, DiffsolError>
where
    M: MatrixHost<T = f64, V = V>,
    V: Vector<T = f64>
{
    // Compile diffsl for this problem and apply config
    let mut problem = OdeBuilder::<M>::new()
        .rtol(config.rtol)
        .build_from_diffsl::<JitModule>(code)?;

    // Return valid problem if correct number of params specified
    let params = V::from_slice(&params, V::C::default());
    let nparams = problem.eqn.nparams();
    if params.len() == nparams {
        problem.eqn.set_params(&params);
        Ok(problem)
    } else {
        Err(DiffsolError::Other(format!(
            "Expecting {} params but got {}",
            nparams,
            params.len()
        )).into())
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
        let config_guard = config.0.lock().unwrap();
        let params = params.as_array();

        match self_guard.matrix_type {
            MatrixType::NalgebraDenseF64 => {
                match config_guard.linear_solver {
                    SolverType::Lu => {
                        let problem = build_diffsl::<NalgebraMat<f64>, NalgebraVec<f64>>(
                            self_guard.code.as_str(),
                            &config_guard,
                            &params.as_slice().unwrap()
                        )?;
                        let (ys, ts) = match config_guard.method {
                            SolverMethod::Bdf => problem.bdf::<NalgebraLU<f64>>()?.solve(time)?,
                            SolverMethod::Esdirk34 => problem.esdirk34::<NalgebraLU<f64>>()?.solve(time)?,
                        };
                        Ok((
                            ys.inner().to_pyarray2(slf.py()),
                            PyArray1::from_owned_array(slf.py(), Array1::from(ts))
                        ))
                    },
                    SolverType::Klu => {
                        Err(DiffsolError::Other("KLU not supported for nalgebra".to_string()).into())
                    }
                }
            },
            MatrixType::FaerSparseF64 => {
                match config_guard.linear_solver {
                    SolverType::Lu => {
                        let problem = build_diffsl::<FaerMat<f64>, FaerVec<f64>>(
                            self_guard.code.as_str(),
                            &config_guard,
                            &params.as_slice().unwrap()
                        )?;
                        let (ys, ts) = match config_guard.method {
                            SolverMethod::Bdf => problem.bdf::<FaerLU<f64>>()?.solve(time)?,
                            SolverMethod::Esdirk34 => problem.esdirk34::<FaerLU<f64>>()?.solve(time)?,
                        };
                        Ok((
                            ys.inner().to_pyarray2(slf.py()),
                            PyArray1::from_owned_array(slf.py(), Array1::from(ts))
                        ))
                    },
                    SolverType::Klu => {
                        Err(DiffsolError::Other("KLU not supported for faer".to_string()).into())
                    }
                }
            }
        }
    }
}
