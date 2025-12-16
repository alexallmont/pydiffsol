// Delegate solver types selected at runtime in Python to concrete solver types
// in Rust.

use diffsol::Scalar;
use diffsol::{
    error::DiffsolError, matrix::MatrixRef, DefaultDenseMatrix, DefaultSolver, DiffSl, Matrix,
    MatrixCommon, OdeBuilder, OdeEquations, OdeSolverProblem, Op, Vector, VectorCommon,
    VectorHost, VectorRef, DiffSlScalar,
};
use numpy::PyReadonlyArray2;
use numpy::{ndarray::Array1, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::{Bound, Python, PyAny};

use crate::convert::{PyCompatibleScalar, RealF32OrF64};
use crate::data_type::DataType;
use crate::valid_linear_solver::{KluValidator, LuValidator};
use crate::{
    convert::{MatrixToPy, VectorToPy},
    solver_method::SolverMethod,
};
use crate::{
    error::PyDiffsolError, jit::JitModule, matrix_type::MatrixType, matrix_type::MatrixKind, solver_type::SolverType,
    valid_linear_solver::validate_linear_solver,
};

// Each matrix type implements PySolve as bridge between diffsol and Python
pub(crate) trait PySolve<T: RealF32OrF64> {
    fn matrix_type(&self) -> MatrixType;

    fn check(&self, linear_solver: SolverType) -> Result<(), PyDiffsolError>;
    fn set_rtol(&mut self, rtol: T);
    fn rtol(&self) -> T;
    fn set_atol(&mut self, atol: T);
    fn atol(&self) -> T;

    #[allow(clippy::type_complexity)]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[T],
        final_time: T,
    ) -> Result<(Bound<'py, PyArray2<T>>, Bound<'py, PyArray1<T>>), PyDiffsolError>;

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[T],
        t_eval: PyReadonlyArray1<'py, T>,
    ) -> Result<Bound<'py, PyArray2<T>>, PyDiffsolError>;

    #[allow(clippy::type_complexity)]
    fn solve_fwd_sens<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[T],
        t_eval: PyReadonlyArray1<'py, T>,
    ) -> Result<(Bound<'py, PyArray2<T>>, Vec<Bound<'py, PyArray2<T>>>), PyDiffsolError>;

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn solve_sum_squares_adj<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        backwards_method: SolverMethod,
        backwards_linear_solver: SolverType,
        params: &[T],
        data: PyReadonlyArray2<'py, T>,
        t_eval: PyReadonlyArray1<'py, T>,
    ) -> Result<(T, Bound<'py, PyArray1<T>>), PyDiffsolError>;
}

// Public factory method for generating an instance based on matrix type
pub(crate) fn py_solve_factory<T>(
    code: &str,
    matrix_type: MatrixType,
) -> Result<Box<dyn PySolve<T>>, PyDiffsolError>
where
    T: RealF32OrF64 + Scalar,
    <T as RealF32OrF64>::MatrixType: RealF32OrF64,
{
    let py_solve: Box<dyn PySolve<T>> = match matrix_type {
        MatrixType::NalgebraDense => {
            Box::new(GenericPySolve::<diffsol::NalgebraMat<T>>::new(code)?)
        },
        MatrixType::FaerDense => {
            Box::new(GenericPySolve::<diffsol::FaerMat<T>>::new(code)?)
        },
        MatrixType::FaerSparse => {
            Box::new(GenericPySolve::<diffsol::FaerSparseMat<T>>::new(code)?)
        },
    };
    Ok(py_solve)
}

pub(crate) struct GenericPySolve<M>
where
    M: Matrix<T: RealF32OrF64>,
    M::V: VectorHost,
{
    problem: OdeSolverProblem<DiffSl<M, JitModule>>,
}

impl<M> GenericPySolve<M>
where
    M: Matrix<T: RealF32OrF64>,
    M::V: VectorHost,
{
    pub fn new(code: &str) -> Result<Self, PyDiffsolError> {
        let problem = OdeBuilder::<M>::new().build_from_diffsl::<JitModule>(code)?;
        Ok(GenericPySolve { problem })
    }

    pub(crate) fn setup_problem(&mut self, params: &[M::T]) -> Result<(), PyDiffsolError> {
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

impl<M, T> PySolve<T> for GenericPySolve<M>
where
    T: RealF32OrF64,
    <T as RealF32OrF64>::MatrixType: RealF32OrF64,
    M: Matrix<T = T> + DefaultSolver + LuValidator<M> + KluValidator<M> + MatrixKind,
    for<'b> <<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: MatrixToPy<'b, M::T>,
    for<'b> <M::V as VectorCommon>::Inner: VectorToPy<'b, T>,
    M::V: VectorHost + DefaultDenseMatrix,
    for<'b> &'b M::V: VectorRef<M::V>,
    for<'b> &'b M: MatrixRef<M>,
{
    fn matrix_type(&self) -> MatrixType {
        MatrixType::from_diffsol::<M>()
    }

    fn check(&self, linear_solver: SolverType) -> Result<(), PyDiffsolError> {
        validate_linear_solver::<M>(linear_solver)
    }

    fn set_atol(&mut self, atol: M::T) {
        self.problem.atol.fill(atol);
    }

    fn atol(&self) -> M::T {
        self.problem.atol[0]
    }

    fn set_rtol(&mut self, rtol: M::T) {
        self.problem.rtol = rtol;
    }

    fn rtol(&self) -> M::T {
        self.problem.rtol
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[M::T],
        final_time: M::T,
    ) -> Result<(Bound<'py, PyArray2<M::T>>, Bound<'py, PyArray1<M::T>>), PyDiffsolError> {
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
        params: &[M::T],
        t_eval: PyReadonlyArray1<'py, M::T>,
    ) -> Result<Bound<'py, PyArray2<M::T>>, PyDiffsolError> {
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
        params: &[M::T],
        t_eval: PyReadonlyArray1<'py, M::T>,
    ) -> Result<(Bound<'py, PyArray2<M::T>>, Vec<Bound<'py, PyArray2<M::T>>>), PyDiffsolError> {
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

    fn solve_sum_squares_adj<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        backwards_method: SolverMethod,
        backwards_linear_solver: SolverType,
        params: &[M::T],
        data: PyReadonlyArray2<'py, M::T>,
        t_eval: PyReadonlyArray1<'py, M::T>,
    ) -> Result<(M::T, Bound<'py, PyArray1<M::T>>), PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let (y, y_sens) = match linear_solver {
            SolverType::Default => method.solve_sum_squares_adj::<M::T, M, <M as DefaultSolver>::LS>(
                &mut self.problem,
                data.as_array(),
                t_eval.as_slice().unwrap(),
                backwards_method,
                backwards_linear_solver,
            ),
            SolverType::Lu => method.solve_sum_squares_adj::<M::T, M, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                data.as_array(),
                t_eval.as_slice().unwrap(),
                backwards_method,
                backwards_linear_solver,
            ),
            SolverType::Klu => method.solve_sum_squares_adj::<M::T, M, <M as KluValidator<M>>::LS>(
                &mut self.problem,
                data.as_array(),
                t_eval.as_slice().unwrap(),
                backwards_method,
                backwards_linear_solver,
            ),
        }?;

        Ok((y, y_sens.inner().to_pyarray1(py)))
    }
}
