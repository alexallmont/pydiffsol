
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
    /// Construct an ODE solver for specified diffsol using a given matrix type
    #[new]
    fn new(code: &str, matrix_type: MatrixType) -> PyResult<Self> {
        Ok(OdeWrapper(Arc::new(Mutex::new(
            Ode {
                code: code.to_string(),
                matrix_type: matrix_type
            }
        ))))
    }

    /// Using the provided state, solve the problem up to time `final_time`.
    ///
    /// The number of params must match the expected params in the diffsl code.
    /// If specified, the config can be used to override the solver method
    /// (Bdf by default) and SolverType (Lu by default) along with other solver
    /// params like `rtol`.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param final_time: end time of solver
    /// :type final_time: float
    /// :param config: optional solver configuration
    /// :type config: pydiffsol.Config, optional
    /// :return: `(ys, ts)` tuple where `ys` is a 2D array of values at times
    ///     `ts` chosen by the solver
    /// :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    ///
    /// Example:
    ///     >>> print(ode.solve(np.array([]), 0.5))
    #[pyo3(signature=(params, final_time, config = ConfigWrapper::new()))]
    fn solve<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        final_time: f64,
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
                            SolverMethod::Bdf => problem.bdf::<NalgebraLU<f64>>()?.solve(final_time)?,
                            SolverMethod::Esdirk34 => problem.esdirk34::<NalgebraLU<f64>>()?.solve(final_time)?,
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
                            SolverMethod::Bdf => problem.bdf::<FaerLU<f64>>()?.solve(final_time)?,
                            SolverMethod::Esdirk34 => problem.esdirk34::<FaerLU<f64>>()?.solve(final_time)?,
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

    /// Using the provided state, solve the problem up to time
    /// `t_eval[t_eval.len()-1]`. Returns 2D array of solution values at
    /// timepoints given by `t_eval`.
    ///
    /// The number of params must match the expected params in the diffsl code.
    /// The config may be optionally specified to override solver settings.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type params: numpy.ndarray
    /// :param config: optional solver configuration
    /// :type config: pydiffsol.Config, optional
    /// :return: 2D array of values at times `t_eval`
    /// :rtype: numpy.ndarray
    #[pyo3(signature=(params, t_eval, config = ConfigWrapper::new()))]
    fn solve_dense<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
        config: ConfigWrapper
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        let self_guard = slf.0.lock().unwrap();
        let config_guard = config.0.lock().unwrap();
        let params = params.as_array();
        let t_eval = t_eval.as_array();

        match self_guard.matrix_type {
            MatrixType::NalgebraDenseF64 => {
                match config_guard.linear_solver {
                    SolverType::Lu => {
                        let problem = build_diffsl::<NalgebraMat<f64>, NalgebraVec<f64>>(
                            self_guard.code.as_str(),
                            &config_guard,
                            &params.as_slice().unwrap()
                        )?;
                        Ok(match config_guard.method {
                            SolverMethod::Bdf => problem.bdf::<NalgebraLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap())?,
                            SolverMethod::Esdirk34 => problem.esdirk34::<NalgebraLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap())?,
                        }.inner().to_pyarray2(slf.py()))
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
                        Ok(match config_guard.method {
                            SolverMethod::Bdf => problem.bdf::<FaerLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap())?,
                            SolverMethod::Esdirk34 => problem.esdirk34::<FaerLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap())?,
                        }.inner().to_pyarray2(slf.py()))
                    },
                    SolverType::Klu => {
                        Err(DiffsolError::Other("KLU not supported for faer".to_string()).into())
                    }
                }
            }
        }
    }
}
