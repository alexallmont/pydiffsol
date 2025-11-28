// Delegate solver types selected at runtime in Python to concrete solver types
// in Rust.

use diffsol::{
    error::DiffsolError, matrix::MatrixRef, DefaultDenseMatrix, DefaultSolver, DiffSl, Matrix,
    MatrixCommon, OdeBuilder, OdeEquations, OdeSolverProblem, Op, Vector, VectorHost, VectorRef,
};
use numpy::{ndarray::Array1, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::{Bound, Python};

use crate::valid_linear_solver::{KluValidator, LuValidator};
use crate::{convert::MatrixToPy, solver_method::SolverMethod};
use crate::{
    error::PyDiffsolError, jit::JitModule, matrix_type::MatrixType, solver_type::SolverType,
    valid_linear_solver::validate_linear_solver,
};

// Each matrix type implements PySolve as bridge between diffsol and Python
pub(crate) trait PySolve {
    fn matrix_type(&self) -> MatrixType;

    #[allow(clippy::type_complexity)]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError>;

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError>;

    #[allow(clippy::type_complexity)]
    fn solve_fwd_sens<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>), PyDiffsolError>;

    fn check(&self, linear_solver: SolverType) -> Result<(), PyDiffsolError>;
    fn set_rtol(&mut self, rtol: f64);
    fn rtol(&self) -> f64;
    fn set_atol(&mut self, atol: f64);
    fn atol(&self) -> f64;
}

// Public factory method for generating an instance based on matrix type
pub(crate) fn py_solve_factory(
    code: &str,
    matrix_type: MatrixType,
) -> Result<Box<dyn PySolve>, PyDiffsolError> {
    let py_solve: Box<dyn PySolve> = match matrix_type {
        MatrixType::NalgebraDenseF64 => {
            Box::new(GenericPySolve::<diffsol::NalgebraMat<f64>>::new(code)?)
        }
        MatrixType::FaerDenseF64 => Box::new(GenericPySolve::<diffsol::FaerMat<f64>>::new(code)?),
        MatrixType::FaerSparseF64 => {
            Box::new(GenericPySolve::<diffsol::FaerSparseMat<f64>>::new(code)?)
        }
    };
    Ok(py_solve)
}

pub(crate) struct GenericPySolve<M>
where
    M: Matrix<T = f64>,
    M::V: VectorHost,
{
    problem: OdeSolverProblem<DiffSl<M, JitModule>>,
}

impl<M> GenericPySolve<M>
where
    M: Matrix<T = f64>,
    M::V: VectorHost,
{
    pub fn new(code: &str) -> Result<Self, PyDiffsolError> {
        let problem = OdeBuilder::<M>::new().build_from_diffsl::<JitModule>(code)?;
        Ok(GenericPySolve { problem })
    }

    pub(crate) fn setup_problem(&mut self, params: &[f64]) -> Result<(), PyDiffsolError> {
        let params = M::V::from_slice(params, M::C::default());

        // Attempt to set problem from params and config
        let nparams = self.problem.eqn.nparams();
        if params.len() == nparams {
            self.problem.eqn.set_params(&params);
            Ok(())
        } else {
            Err(DiffsolError::Other(format!(
                "Expecting {} params but got {}",
                nparams,
                params.len()
            ))
            .into())
        }
    }
}

impl<M> PySolve for GenericPySolve<M>
where
    M: Matrix<T = f64> + DefaultSolver + LuValidator<M> + KluValidator<M>,
    for<'b> <<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: MatrixToPy<'b>,
    M::V: VectorHost + DefaultDenseMatrix,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    fn matrix_type(&self) -> MatrixType {
        MatrixType::from_diffsol::<M>().expect("Unknown matrix type")
    }

    fn check(&self, linear_solver: SolverType) -> Result<(), PyDiffsolError> {
        validate_linear_solver::<M>(linear_solver)
    }

    fn set_atol(&mut self, atol: f64) {
        self.problem.atol.fill(atol);
    }

    fn atol(&self) -> f64 {
        self.problem.atol[0]
    }

    fn set_rtol(&mut self, rtol: f64) {
        self.problem.rtol = rtol;
    }

    fn rtol(&self) -> f64 {
        self.problem.rtol
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;
        let (ys, ts) = match linear_solver {
            SolverType::Default => {
                method.solve::<M, <M as DefaultSolver>::LS>(&mut self.problem, final_time)
            }
            SolverType::Lu => {
                method.solve::<M, <M as LuValidator<M>>::LS>(&mut self.problem, final_time)
            }
            SolverType::Klu => {
                method.solve::<M, <M as KluValidator<M>>::LS>(&mut self.problem, final_time)
            }
        }?;

        Ok((
            ys.inner().to_pyarray2(py),
            PyArray1::from_owned_array(py, Array1::from(ts)),
        ))
    }

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let ys = match linear_solver {
            SolverType::Default => method.solve_dense::<M, <M as DefaultSolver>::LS>(
                &mut self.problem,
                t_eval.as_slice().unwrap(),
            ),
            SolverType::Lu => method.solve_dense::<M, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                t_eval.as_slice().unwrap(),
            ),
            SolverType::Klu => method.solve_dense::<M, <M as KluValidator<M>>::LS>(
                &mut self.problem,
                t_eval.as_slice().unwrap(),
            ),
        }?;

        Ok(ys.inner().to_pyarray2(py))
    }

    fn solve_fwd_sens<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>), PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let (ys, sens) = match linear_solver {
            SolverType::Default => method.solve_fwd_sens::<M, <M as DefaultSolver>::LS>(
                &mut self.problem,
                t_eval.as_slice().unwrap(),
            ),
            SolverType::Lu => method.solve_fwd_sens::<M, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                t_eval.as_slice().unwrap(),
            ),
            SolverType::Klu => method.solve_fwd_sens::<M, <M as KluValidator<M>>::LS>(
                &mut self.problem,
                t_eval.as_slice().unwrap(),
            ),
        }?;

        Ok((
            ys.inner().to_pyarray2(py),
            sens.into_iter()
                .map(|s| s.inner().to_pyarray2(py))
                .collect(),
        ))
    }
}
