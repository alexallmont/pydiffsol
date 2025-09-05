
use std::sync::{Arc, Mutex};

use crate::config::{Config, ConfigWrapper};
use crate::convert::MatrixToPy;
use crate::matrix_type::MatrixType;
use crate::solver_method::SolverMethod;
use crate::solver_type::SolverType;
use crate::error::PyDiffsolError;
use crate::jit::JitModule;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use diffsol::{DiffSl, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem};
use diffsol::{MatrixCommon, matrix::MatrixHost};
use diffsol::error::DiffsolError;
use diffsol::{NalgebraMat, NalgebraLU};
use diffsol::{FaerMat, FaerLU, FaerSparseMat};
use diffsol::Vector; // for from_slice
use diffsol::Op; // For nparams

#[cfg(feature = "suitesparse")]
use diffsol::KLU;

use numpy::{PyReadonlyArray1, PyArray1, PyArray2};
use numpy::ndarray::Array1;

// FIXME move PySolve and related into separate file
fn setup_problem<M>(
    problem: &mut OdeSolverProblem<diffsol::DiffSl<M, JitModule>>,
    config: &Config,
    params: &[f64],
) -> Result<(), PyDiffsolError>
where
    M: MatrixHost<T = f64>,
{
    let params = M::V::from_slice(&params, M::C::default());

    // Attempt to set problem from params and config
    let nparams = problem.eqn.nparams();
    if params.len() == nparams {
        problem.eqn.set_params(&params);
        problem.rtol = config.rtol;
        Ok(())
    } else {
        Err(DiffsolError::Other(format!(
            "Expecting {} params but got {}",
            nparams,
            params.len()
        )).into())
    }
}

trait PySolve {
    fn matrix_type(&self) -> MatrixType;

    /// FIXME docs
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError>;

    /// FIXME docs
    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError>;
}

struct PySolveNalgabraF64 {
    problem: OdeSolverProblem<DiffSl<NalgebraMat<f64>, JitModule>>,
}

impl PySolveNalgabraF64 {
    fn new(code: &str) -> Result<Self, DiffsolError> {
        Ok(
            PySolveNalgabraF64{
                problem: OdeBuilder::<NalgebraMat<f64>>::new()
                    .build_from_diffsl::<JitModule>(code)?
            }
        )
    }
}

impl PySolve for PySolveNalgabraF64 {
    fn matrix_type(&self) -> MatrixType {
        MatrixType::NalgebraDenseF64
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        setup_problem(&mut self.problem, config, params)?;

        let (ys, ts) = match solver_type {
            SolverType::Default => {
                match config.method {
                    SolverMethod::Tsit45 => self.problem.tsit45()?.solve(final_time),
                    _ => Err(DiffsolError::Other("Only tsit45 is compatible with Default solver for NalgebraDenseF64".to_string()).into())
                }
            },
            SolverType::Lu => {
                match config.method {
                    SolverMethod::Bdf => self.problem.bdf::<NalgebraLU<f64>>()?.solve(final_time),
                    SolverMethod::Esdirk34 => self.problem.esdirk34::<NalgebraLU<f64>>()?.solve(final_time),
                    SolverMethod::TrBdf2 => self.problem.tr_bdf2::<NalgebraLU<f64>>()?.solve(final_time),
                    SolverMethod::Tsit45 => Err(DiffsolError::Other("Lu solver is compatible with tsit45 for NalgebraDenseF64".to_string()).into()),
                }
            },
            SolverType::Klu => Err(DiffsolError::Other("Klu not supported for NalgebraDenseF64".to_string()).into()),
        }?;

        Ok((
            ys.inner().to_pyarray2(py),
            PyArray1::from_owned_array(py, Array1::from(ts))
        ))
    }

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        setup_problem(&mut self.problem, config, params)?;

        let ys = match solver_type {
            SolverType::Default => {
                match config.method {
                    SolverMethod::Tsit45 => self.problem.tsit45()?.solve_dense(t_eval.as_slice().unwrap()),
                    _ => Err(DiffsolError::Other("Only tsit45 is compatible with Default solver for NalgebraDenseF64".to_string()).into())
                }
            },
            SolverType::Lu => {
                match config.method {
                    SolverMethod::Bdf => self.problem.bdf::<NalgebraLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::Esdirk34 => self.problem.esdirk34::<NalgebraLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::TrBdf2 => self.problem.tr_bdf2::<NalgebraLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::Tsit45 => Err(DiffsolError::Other("Lu solver is compatible with tsit45 for NalgebraDenseF64".to_string()).into()),
                }
            },
            SolverType::Klu => Err(DiffsolError::Other("Klu not supported for NalgebraDenseF64".to_string()).into()),
        }?;

        Ok(ys.inner().to_pyarray2(py))
    }

}

struct PySolveFaerDenseF64 {
    problem: OdeSolverProblem<DiffSl<FaerMat<f64>, JitModule>>,
}

impl PySolveFaerDenseF64 {
    fn new(code: &str) -> Result<Self, DiffsolError> {
        Ok(
            PySolveFaerDenseF64{
                problem: OdeBuilder::<FaerMat<f64>>::new()
                    .build_from_diffsl::<JitModule>(code)?
            }
        )
    }
}

impl PySolve for PySolveFaerDenseF64 {
    fn matrix_type(&self) -> MatrixType {
        MatrixType::FaerDenseF64
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        setup_problem(&mut self.problem, config, params)?;

        let (ys, ts) = match solver_type {
            SolverType::Default => {
                match config.method {
                    SolverMethod::Tsit45 => self.problem.tsit45()?.solve(final_time),
                    _ => Err(DiffsolError::Other("Only tsit45 is compatible with Default solver for FaerDenseF64".to_string()).into())
                }
            },
            SolverType::Lu => {
                match config.method {
                    SolverMethod::Bdf => self.problem.bdf::<FaerLU<f64>>()?.solve(final_time),
                    SolverMethod::Esdirk34 => self.problem.esdirk34::<FaerLU<f64>>()?.solve(final_time),
                    SolverMethod::TrBdf2 => self.problem.tr_bdf2::<FaerLU<f64>>()?.solve(final_time),
                    SolverMethod::Tsit45 => Err(DiffsolError::Other("Lu solver is not compatible with tsit45 for FaerDenseF64".to_string()).into())
                }
            },
            SolverType::Klu => Err(DiffsolError::Other("Klu solver not supported for FaerDenseF64".to_string()).into()),
        }?;

        Ok((
            ys.inner().to_pyarray2(py),
            PyArray1::from_owned_array(py, Array1::from(ts))
        ))
    }

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        setup_problem(&mut self.problem, config, params)?;

        let ys = match solver_type {
            SolverType::Default => {
                match config.method {
                    SolverMethod::Tsit45 => self.problem.tsit45()?.solve_dense(t_eval.as_slice().unwrap()),
                    _ => Err(DiffsolError::Other("Only tsit45 is compatible with Default solver for FaerDenseF64".to_string()).into())
                }
            },
            SolverType::Lu => {
                match config.method {
                    SolverMethod::Bdf => self.problem.bdf::<FaerLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::Esdirk34 => self.problem.esdirk34::<FaerLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::TrBdf2 => self.problem.tr_bdf2::<FaerLU<f64>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::Tsit45 => Err(DiffsolError::Other("Lu solver is not compatible with tsit45 for FaerDenseF64".to_string()).into())
                }
            },
            SolverType::Klu => Err(DiffsolError::Other("Klu solver not supported for FaerDenseF64".to_string()).into()),
        }?;

        Ok(ys.inner().to_pyarray2(py))
    }
}

struct PySolveFaerSparseF64 {
    problem: OdeSolverProblem<DiffSl<FaerSparseMat<f64>, JitModule>>,
}

impl PySolveFaerSparseF64 {
    fn new(code: &str) -> Result<Self, DiffsolError> {
        Ok(
            PySolveFaerSparseF64{
                problem: OdeBuilder::<FaerSparseMat<f64>>::new()
                    .build_from_diffsl::<JitModule>(code)?
            }
        )
    }
}

impl PySolve for PySolveFaerSparseF64 {
    fn matrix_type(&self) -> MatrixType {
        MatrixType::FaerSparseF64
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        setup_problem(&mut self.problem, config, params)?;

        let (ys, ts) = match solver_type {
            SolverType::Default => {
                match config.method {
                    SolverMethod::Tsit45 => self.problem.tsit45()?.solve(final_time),
                    _ => Err(DiffsolError::Other("Only tsit45 is compatible with Default solver for FaerSparseF64".to_string()).into())
                }
            },
            SolverType::Lu => Err(DiffsolError::Other("Lu solver not supported for FaerSparseF64".to_string()).into()),
            SolverType::Klu => {
                #[cfg(feature = "suitesparse")]
                match config.method {
                    SolverMethod::Bdf => self.problem.bdf::<KLU<FaerSparseMat<f64>>>()?.solve(final_time),
                    SolverMethod::Esdirk34 => self.problem.esdirk34::<KLU<FaerSparseMat<f64>>>()?.solve(final_time),
                    SolverMethod::TrBdf2 => self.problem.tr_bdf2::<KLU<FaerSparseMat<f64>>>()?.solve(final_time),
                    SolverMethod::Tsit45 => Err(DiffsolError::Other("Klu solver is not compatible with tsit45 for FaerSparseF64".to_string()).into())
                }
                #[cfg(not(feature = "suitesparse"))]
                Err(DiffsolError::Other("Klu solver is not available in this pydiffsol build; suitesparse is not enabled".to_string()).into())
            },
        }?;

        Ok((
            ys.inner().to_pyarray2(py),
            PyArray1::from_owned_array(py, Array1::from(ts))
        ))
    }

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        setup_problem(&mut self.problem, config, params)?;

        let ys = match solver_type {
            SolverType::Default => {
                match config.method {
                    SolverMethod::Tsit45 => self.problem.tsit45()?.solve_dense(t_eval.as_slice().unwrap()),
                    _ => Err(DiffsolError::Other("Only tsit45 is compatible with Default solver for FaerSparseF64".to_string()).into())
                }
            },
            SolverType::Lu => Err(DiffsolError::Other("Lu solver not supported for FaerSparseF64".to_string()).into()),
            SolverType::Klu => {
                #[cfg(feature = "suitesparse")]
                match config.method {
                    SolverMethod::Bdf => self.problem.bdf::<KLU<FaerSparseMat<f64>>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::Esdirk34 => self.problem.esdirk34::<KLU<FaerSparseMat<f64>>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::TrBdf2 => self.problem.tr_bdf2::<KLU<FaerSparseMat<f64>>>()?.solve_dense(t_eval.as_slice().unwrap()),
                    SolverMethod::Tsit45 => Err(DiffsolError::Other("Klu solver is not compatible with tsit45 for FaerSparseF64".to_string()).into())
                }
                #[cfg(not(feature = "suitesparse"))]
                Err(DiffsolError::Other("Klu solver is not available in this pydiffsol build; suitesparse is not enabled".to_string()).into())
            },
        }?;

        Ok(ys.inner().to_pyarray2(py))
    }
}

#[pyclass]
struct Ode {
    code: String,
    py_solve: Box<dyn PySolve>,
}
unsafe impl Send for Ode {}
unsafe impl Sync for Ode {}

#[pyclass]
#[pyo3(name = "Ode")]
#[derive(Clone)]
pub struct OdeWrapper(Arc<Mutex<Ode>>);

#[pymethods]
impl OdeWrapper {
    /// Construct an ODE solver for specified diffsol using a given matrix type
    #[new]
    fn new(code: &str, matrix_type: MatrixType) -> Result<Self, PyDiffsolError> {
        let py_solve: Box<dyn PySolve> = match matrix_type {
            MatrixType::NalgebraDenseF64 => Box::new(PySolveNalgabraF64::new(code)?),
            MatrixType::FaerDenseF64 => Box::new(PySolveFaerDenseF64::new(code)?),
            MatrixType::FaerSparseF64 => Box::new(PySolveFaerSparseF64::new(code)?),
        };

        Ok(OdeWrapper(Arc::new(Mutex::new(
            Ode {
                code: code.to_string(),
                py_solve
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
    #[pyo3(signature=(params, final_time, config=ConfigWrapper::new(SolverMethod::Bdf, SolverType::Default, 1e-6)))]
    fn solve<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        final_time: f64,
        config: ConfigWrapper
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let config_guard = config.0.lock().unwrap();
        let params = params.as_array();
        let matrix_type = self_guard.py_solve.matrix_type();
        let solver_type = config_guard.solver_for_matrix_type(matrix_type);

        self_guard.py_solve.solve(
            slf.py(),
            solver_type,
            &config_guard,
            &params.as_slice().unwrap(),
            final_time,
        )
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
    #[pyo3(signature=(params, t_eval, config=ConfigWrapper::new(SolverMethod::Bdf, SolverType::Default, 1e-6)))]
    fn solve_dense<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
        config: ConfigWrapper
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let config_guard = config.0.lock().unwrap();
        let params = params.as_array();
        let solver_type = config_guard.solver_for_matrix_type(self_guard.py_solve.matrix_type());

        self_guard.py_solve.solve_dense(
            slf.py(),
            solver_type,
            &config_guard,
            &params.as_slice().unwrap(),
            t_eval,
        )
    }
}
