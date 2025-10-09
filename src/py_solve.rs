use diffsol::{error::DiffsolError, matrix::{MatrixHost, MatrixRef}, DefaultDenseMatrix, DefaultSolver, DiffSl, Matrix, MatrixCommon, OdeBuilder, OdeEquations, OdeSolverProblem, Op, Vector, VectorHost, VectorRef};
use numpy::{ndarray::Array1, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::{Bound, Python};

use crate::{convert::MatrixToPy, ode_config::OdeConfig};
use crate::{solve_config::SolveConfig, error::PyDiffsolError, jit::JitModule, matrix_type::MatrixType, solver_type::SolverType};

// Each matrix type implements PySolve as bridge between diffsol and Python
pub(crate) trait PySolve {
    fn matrix_type(&self) -> MatrixType;

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        config: &SolveConfig,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError>;

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        config: &SolveConfig,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError>;
}

// Public factory method for generating an instance based on matrix type
pub(crate) fn py_solve_factory(
    code: &str,
    config: &OdeConfig,
) -> Result<Box<dyn PySolve>, PyDiffsolError> {
    let py_solve: Box<dyn PySolve> = match config.matrix_type {
        MatrixType::NalgebraDenseF64 => Box::new(GenericPySolve::<diffsol::NalgebraMat<f64>>::new(code, config)?),
        MatrixType::FaerDenseF64 => Box::new(GenericPySolve::<diffsol::FaerMat<f64>>::new(code, config)?),
        MatrixType::FaerSparseF64 => Box::new(GenericPySolve::<diffsol::FaerSparseMat<f64>>::new(code, config)?),
    };
    Ok(py_solve)
}



// Called before solve or solve_dense to ensure config and params are set
fn setup_problem<M>(
    problem: &mut OdeSolverProblem<diffsol::DiffSl<M, JitModule>>,
    params: &[f64],
) -> Result<(), PyDiffsolError>
where
    M: MatrixHost<T = f64>,
{
    let params = M::V::from_slice(params, M::C::default());

    // Attempt to set problem from params and config
    let nparams = problem.eqn.nparams();
    if params.len() == nparams {
        problem.eqn.set_params(&params);
        Ok(())
    } else {
        Err(DiffsolError::Other(format!(
            "Expecting {} params but got {}",
            nparams,
            params.len()
        )).into())
    }
}

trait Klu<M: Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

#[cfg(feature = "suitesparse")]
impl Klu<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::KLU<diffsol::FaerSparseMat<f64>>;
    fn valid() -> bool {
        true
    }
}


#[cfg(not(feature = "suitesparse"))]
impl Klu<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::FaerSparseLU<f64>;
    fn valid() -> bool {
        false
    }
}

impl Klu<diffsol::NalgebraMat<f64>> for diffsol::NalgebraMat<f64> {
    type LS = diffsol::NalgebraLU<f64>;
    fn valid() -> bool {
        false
    }
}

impl Klu<diffsol::FaerMat<f64>> for diffsol::FaerMat<f64> {
    type LS = diffsol::FaerLU<f64>;
    fn valid() -> bool {
        false
    }
}


trait Lu<M: Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

impl Lu<diffsol::NalgebraMat<f64>> for diffsol::NalgebraMat<f64> {
    type LS = diffsol::NalgebraLU<f64>;
    fn valid() -> bool {
        true
    }
}

impl Lu<diffsol::FaerMat<f64>> for diffsol::FaerMat<f64> {
    type LS = diffsol::FaerLU<f64>;
    fn valid() -> bool {
        true
    }
}

impl Lu<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::FaerSparseLU<f64>;
    fn valid() -> bool {
        true
    }
}

pub(crate) struct GenericPySolve<M> 
where 
    M: Matrix<T=f64>,
    M::V: VectorHost,
{
    problem: OdeSolverProblem<DiffSl<M, JitModule>>,
}

impl<M> GenericPySolve<M> 
where 
    M: Matrix<T=f64>,
    M::V: VectorHost,
{
    pub fn new(code: &str, config: &OdeConfig) -> Result<Self, PyDiffsolError> {
        let problem = OdeBuilder::<M>::new()
            .rtol(config.rtol)
            .atol([config.atol])
            .build_from_diffsl::<JitModule>(code)?;
        Ok(GenericPySolve { problem })
    }
}

impl<M> PySolve for GenericPySolve<M> 
where
    M: Matrix<T=f64> + DefaultSolver + Lu<M> + Klu<M>,
    for<'b> <<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: MatrixToPy<'b>,
    M::V: VectorHost + DefaultDenseMatrix,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    fn matrix_type(&self) -> MatrixType {
        MatrixType::from_diffsol::<M>().expect("Unknown matrix type")
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        config: &SolveConfig,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        setup_problem(&mut self.problem, params)?;

        let (ys, ts) = match config.linear_solver {
            SolverType::Default => config.method.solve::<M, <M as DefaultSolver>::LS>(&mut self.problem, final_time),
            SolverType::Lu => {
                if !<M as Lu<M>>::valid() {
                    return Err(DiffsolError::Other(format!("Lu solver not supported for {}", self.matrix_type().get_name())).into());
                }
                config.method.solve::<M, <M as Lu<M>>::LS>(&mut self.problem, final_time)
            },
            SolverType::Klu => {
                if !<M as Klu<M>>::valid() {
                    return Err(DiffsolError::Other(format!("Klu solver not supported for {}", self.matrix_type().get_name())).into());
                }
                config.method.solve::<M, <M as Klu<M>>::LS>(&mut self.problem, final_time)
            }
        }?;

        Ok((
            ys.inner().to_pyarray2(py),
            PyArray1::from_owned_array(py, Array1::from(ts))
        ))
    }

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        config: &SolveConfig,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        setup_problem(&mut self.problem, params)?;

        let ys = match config.linear_solver {
            SolverType::Default => config.method.solve_dense::<M, <M as DefaultSolver>::LS>(&mut self.problem, t_eval.as_slice().unwrap()),
            SolverType::Lu =>  {
                if !<M as Lu<M>>::valid() {
                    return Err(DiffsolError::Other(format!("Lu solver not supported for {}", self.matrix_type().get_name())).into());
                }
                config.method.solve_dense::<M, <M as Lu<M>>::LS>(&mut self.problem, t_eval.as_slice().unwrap())
            },
            SolverType::Klu =>  {
                if !<M as Klu<M>>::valid() {
                    return Err(DiffsolError::Other(format!("Klu solver not supported for {}", self.matrix_type().get_name())).into());
                }
                config.method.solve_dense::<M, <M as Klu<M>>::LS>(&mut self.problem, t_eval.as_slice().unwrap())
            }
        }?;

        Ok(ys.inner().to_pyarray2(py))
    }
}



