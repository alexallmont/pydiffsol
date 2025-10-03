use crate::config::Config;
use crate::convert::MatrixToPy;
use crate::error::PyDiffsolError;
use crate::jit::JitModule;
use crate::matrix_type::MatrixType;
use crate::solver_method::SolverMethod;
use crate::solver_type::SolverType;

use diffsol::{DiffSl, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem};
use diffsol::{MatrixCommon, matrix::MatrixHost};
use diffsol::error::DiffsolError;
use diffsol::{NalgebraMat, NalgebraLU};
use diffsol::{FaerMat, FaerLU, FaerSparseMat};
use diffsol::Vector; // for from_slice
use diffsol::Op; // For nparams
use numpy::{PyReadonlyArray1, PyArray1, PyArray2};
use numpy::ndarray::Array1;
use pyo3::prelude::*;

#[cfg(feature = "suitesparse")]
use diffsol::KLU;

// Each matrix type implements PySolve as bridge between diffsol and Python
pub(crate) trait PySolve {
    fn matrix_type(&self) -> MatrixType;

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError>;

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError>;
}

// Public factory method for generating an instance based on matrix type
pub(crate) fn py_solve_factory(
    code: &str,
    matrix_type: MatrixType
) -> Result<Box<dyn PySolve>, PyDiffsolError> {
    let py_solve: Box<dyn PySolve> = match matrix_type {
        MatrixType::NalgebraDenseF64 => Box::new(PySolveNalgabraF64::new(code)?),
        MatrixType::FaerDenseF64 => Box::new(PySolveFaerDenseF64::new(code)?),
        MatrixType::FaerSparseF64 => Box::new(PySolveFaerSparseF64::new(code)?),
    };
    Ok(py_solve)
}

// Called before solve or solve_dense to ensure config and params are set
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
        problem.atol = config.atol;
        Ok(())
    } else {
        Err(DiffsolError::Other(format!(
            "Expecting {} params but got {}",
            nparams,
            params.len()
        )).into())
    }
}

//
// NalgabraF64 implementation
//
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

//
// FaerDenseF64 implementation
//
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

//
// FaerSparseF64 implementation
//
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
