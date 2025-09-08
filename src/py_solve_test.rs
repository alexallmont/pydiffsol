use diffsol::{error::DiffsolError, matrix::{MatrixHost, MatrixRef}, DefaultDenseMatrix, DefaultSolver, DiffSl, Matrix, OdeEquations, OdeSolverMethod, OdeSolverProblem, Op, Vector, VectorHost, VectorRef, MatrixCommon};
use numpy::{ndarray::Array1, PyArray1, PyArray2};
use pyo3::{Bound, Python};

use crate::convert::MatrixToPy;
use crate::{config::Config, error::PyDiffsolError, jit::JitModule, matrix_type::MatrixType, solver_method::SolverMethod, solver_type::SolverType};

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

struct PySolve<M> 
where 
    M: Matrix<T=f64>,
    M::V: VectorHost,
{
    problem: OdeSolverProblem<DiffSl<M, JitModule>>,
}

impl<M> PySolve<M> 
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
        solver_type: SolverType,
        config: &Config,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        setup_problem(&mut self.problem, config, params)?;

        let (ys, ts) = match solver_type {
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
}



struct PySolveNalgabraF64(PySolve<diffsol::NalgebraMat<f64>>);