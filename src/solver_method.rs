// Solver method Python enum. This is used to select the overarching solver
// stragegy like bdf or esdirk34 in diffsol.

use diffsol::error::DiffsolError;
use diffsol::{
    matrix::MatrixRef, DefaultDenseMatrix, DiffSl, LinearSolver, Matrix, OdeSolverMethod,
    OdeSolverProblem, Vector, VectorHost, VectorRef,
};
use diffsol::{
    AdjointOdeSolverMethod, Checkpointing, DefaultSolver, DenseMatrix, DiffSlScalar, MatrixCommon,
    OdeSolverState, Op, SensitivitiesOdeSolverMethod, VectorViewMut,
};
use nalgebra::ComplexField; // for powi
use num_traits::{FromPrimitive, Zero}; // for generic nums in _solve_sum_squares_adj
use numpy::ndarray::ArrayView2;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyType},
};

use crate::{
    is_sens_available,
    jit::JitModule,
    py_solution::GenericPyState,
    solver_type::SolverType,
    valid_linear_solver::{KluValidator, LuValidator},
};

/// Enumerates the possible ODE solver methods for diffsol. See the solver descriptions in the diffsol documentation (https://github.com/martinjrobins/diffsol) for more details.
///
/// :attr bdf: Backward Differentiation Formula (BDF) method for stiff ODEs and singular mass matrices
/// :attr esdirk34: Explicit Singly Diagonally Implicit Runge-Kutta (ESDIRK) method for moderately stiff ODEs and singular mass matrices.
/// :attr tr_bdf2: Trapezoidal Backward Differentiation Formula of order 2 (TR-BDF2) method for moderately stiff ODEs and singular mass matrices.
/// :attr tsit45: Tsitouras 4/5th order Explicit Runge-Kutta (TSIT45) method for non-stiff ODEs. This is an explicit method, it cannot handle singular mass matrices and does not require a linear solver.
#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolverMethod {
    #[pyo3(name = "bdf")]
    Bdf,

    #[pyo3(name = "esdirk34")]
    Esdirk34,

    #[pyo3(name = "tr_bdf2")]
    TrBdf2,

    #[pyo3(name = "tsit45")]
    Tsit45,
}

impl SolverMethod {
    pub(crate) fn all_enums() -> Vec<SolverMethod> {
        vec![
            SolverMethod::Bdf,
            SolverMethod::Esdirk34,
            SolverMethod::TrBdf2,
            SolverMethod::Tsit45,
        ]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            SolverMethod::Bdf => "bdf",
            SolverMethod::Esdirk34 => "esdirk34",
            SolverMethod::TrBdf2 => "tr_bdf2",
            SolverMethod::Tsit45 => "tsit45",
        }
    }

    pub(crate) fn solve<M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule>>,
        final_time: M::T,
        initial_state: Option<GenericPyState<M::V>>,
    ) -> Result<
        (
            <M::V as DefaultDenseMatrix>::M,
            Vec<M::T>,
            GenericPyState<M::V>,
        ),
        DiffsolError,
    >
    where
        M: Matrix<T: DiffSlScalar>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            SolverMethod::Bdf => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Bdf(state)) => {
                        let mut solver = problem.bdf_solver::<LS>(state)?;
                        solver.state_mut(); // ensure any user mutations to the state are reflected in the solver's internal state
                        solver
                    }
                    Some(GenericPyState::Rk(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected a BDF state for bdf method".to_string(),
                        ));
                    }
                    None => problem.bdf::<LS>()?,
                };

                let (ys, ts) = solver.solve(final_time)?;
                Ok((ys, ts, GenericPyState::Bdf(solver.into_state())))
            }
            SolverMethod::Esdirk34 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.esdirk34_solver::<LS>(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for esdirk34 method".to_string(),
                        ));
                    }
                    None => problem.esdirk34::<LS>()?,
                };
                let (ys, ts) = solver.solve(final_time)?;
                Ok((ys, ts, GenericPyState::Rk(solver.into_state())))
            }
            SolverMethod::TrBdf2 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.tr_bdf2_solver::<LS>(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tr_bdf2 method".to_string(),
                        ));
                    }
                    None => problem.tr_bdf2::<LS>()?,
                };
                let (ys, ts) = solver.solve(final_time)?;
                Ok((ys, ts, GenericPyState::Rk(solver.into_state())))
            }
            SolverMethod::Tsit45 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.tsit45_solver(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tsit45 method".to_string(),
                        ));
                    }
                    None => problem.tsit45()?,
                };
                let (ys, ts) = solver.solve(final_time)?;
                Ok((ys, ts, GenericPyState::Rk(solver.into_state())))
            }
        }
    }

    pub(crate) fn solve_dense<M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule>>,
        t_eval: &[M::T],
        initial_state: Option<GenericPyState<M::V>>,
    ) -> Result<(<M::V as DefaultDenseMatrix>::M, GenericPyState<M::V>), DiffsolError>
    where
        M: Matrix<T: DiffSlScalar>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            SolverMethod::Bdf => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Bdf(state)) => {
                        let mut solver = problem.bdf_solver::<LS>(state)?;
                        solver.state_mut(); // ensure any user mutations to the state are reflected in the solver's internal state
                        solver
                    }
                    Some(GenericPyState::Rk(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected a BDF state for bdf method".to_string(),
                        ));
                    }
                    None => problem.bdf::<LS>()?,
                };
                let ys = solver.solve_dense(t_eval)?;
                Ok((ys, GenericPyState::Bdf(solver.into_state())))
            }
            SolverMethod::Esdirk34 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.esdirk34_solver::<LS>(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for esdirk34 method".to_string(),
                        ));
                    }
                    None => problem.esdirk34::<LS>()?,
                };
                let ys = solver.solve_dense(t_eval)?;
                Ok((ys, GenericPyState::Rk(solver.into_state())))
            }
            SolverMethod::TrBdf2 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.tr_bdf2_solver::<LS>(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tr_bdf2 method".to_string(),
                        ));
                    }
                    None => problem.tr_bdf2::<LS>()?,
                };
                let ys = solver.solve_dense(t_eval)?;
                Ok((ys, GenericPyState::Rk(solver.into_state())))
            }
            SolverMethod::Tsit45 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.tsit45_solver(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tsit45 method".to_string(),
                        ));
                    }
                    None => problem.tsit45()?,
                };
                let ys = solver.solve_dense(t_eval)?;
                Ok((ys, GenericPyState::Rk(solver.into_state())))
            }
        }
    }

    fn check_sens_available() -> Result<(), DiffsolError> {
        if !is_sens_available() {
            return Err(DiffsolError::Other(
                "Sensitivity analysis is not supported on Windows, please use a linux or macOS system.".to_string(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn solve_fwd_sens<M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule>>,
        t_eval: &[M::T],
        initial_state: Option<GenericPyState<M::V>>,
    ) -> Result<
        (
            <M::V as DefaultDenseMatrix>::M,
            Vec<<M::V as DefaultDenseMatrix>::M>,
            GenericPyState<M::V>,
        ),
        DiffsolError,
    >
    where
        M: Matrix<T: DiffSlScalar> + DefaultSolver,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            SolverMethod::Bdf => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Bdf(state)) => {
                        let mut solver = problem.bdf_solver_sens::<LS>(state)?;
                        solver.state_mut(); // ensure any user mutations to the state are reflected in the solver's internal state
                        solver
                    }
                    Some(GenericPyState::Rk(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected a BDF state for bdf method".to_string(),
                        ));
                    }
                    None => problem.bdf_sens::<LS>()?,
                };
                let (ys, sens) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericPyState::Bdf(solver.into_state())))
            }
            SolverMethod::Esdirk34 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.esdirk34_solver_sens::<LS>(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for esdirk34 method".to_string(),
                        ));
                    }
                    None => problem.esdirk34_sens::<LS>()?,
                };
                let (ys, sens) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericPyState::Rk(solver.into_state())))
            }
            SolverMethod::TrBdf2 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.tr_bdf2_solver_sens::<LS>(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tr_bdf2 method".to_string(),
                        ));
                    }
                    None => problem.tr_bdf2_sens::<LS>()?,
                };
                let (ys, sens) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericPyState::Rk(solver.into_state())))
            }
            SolverMethod::Tsit45 => {
                let mut solver = match initial_state {
                    Some(GenericPyState::Rk(state)) => problem.tsit45_solver_sens(state)?,
                    Some(GenericPyState::Bdf(_)) => {
                        return Err(DiffsolError::Other(
                            "Expected an RK state for tsit45 method".to_string(),
                        ));
                    }
                    None => problem.tsit45_sens()?,
                };
                let (ys, sens) = solver.solve_dense_sensitivities(t_eval)?;
                Ok((ys, sens, GenericPyState::Rk(solver.into_state())))
            }
        }
    }

    pub(crate) fn solve_sum_squares_adj<'a, M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule>>,
        data: ArrayView2<'a, M::T>,
        t_eval: &[M::T],
        backwards_method: SolverMethod,
        backwards_linear_solver: SolverType,
    ) -> Result<(M::T, M::V), DiffsolError>
    where
        M: Matrix<T: DiffSlScalar> + DefaultSolver + LuValidator<M> + KluValidator<M>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        Self::check_sens_available()?;
        match self {
            SolverMethod::Bdf => self._solve_sum_squares_adj(
                problem.bdf::<LS>()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            SolverMethod::Esdirk34 => self._solve_sum_squares_adj(
                problem.esdirk34::<LS>()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            SolverMethod::TrBdf2 => self._solve_sum_squares_adj(
                problem.tr_bdf2::<LS>()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            SolverMethod::Tsit45 => self._solve_sum_squares_adj(
                problem.tsit45()?,
                data,
                t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
        }
    }

    pub(crate) fn _solve_sum_squares_adj<'data, 'solver, M, S>(
        &self,
        mut solver: S,
        data: ArrayView2<'data, M::T>,
        t_eval: &[M::T],
        backwards_method: SolverMethod,
        backwards_linear_solver: SolverType,
    ) -> Result<(M::T, M::V), DiffsolError>
    where
        M: Matrix<T: DiffSlScalar> + DefaultSolver + LuValidator<M> + KluValidator<M>,
        M::V: VectorHost + DefaultDenseMatrix,
        S: OdeSolverMethod<'solver, DiffSl<M, JitModule>>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let (chk, ys) = solver.solve_dense_with_checkpointing(t_eval, None)?;
        let eqn = solver.problem().eqn();
        let ctx = eqn.context();
        let mut g_m = <M::V as DefaultDenseMatrix>::M::zeros(eqn.nout(), t_eval.len(), ctx.clone());
        let mut y = M::T::zero();
        for j in 0..g_m.ncols() {
            let ys_col = ys.column(j);
            // TODO: can we avoid this allocation? (I can't see how right now)
            let mut tmp = M::V::from_slice(data.column(j).as_slice().unwrap(), ctx.clone());
            // tmp = 2 * ys_col - 2 * tmp
            tmp.axpy_v(
                M::T::from_f64(2.0).unwrap(),
                &ys_col,
                M::T::from_f64(-2.0).unwrap(),
            );
            g_m.column_mut(j).copy_from(&tmp);

            // y = (1/4) * dot(tmp, tmp) + y
            y += M::T::from_f64(1.0 / 4.0).unwrap() * tmp.norm(2).powi(2);
        }
        let mut y_sens = match backwards_linear_solver {
            SolverType::Default => backwards_method
                .solve_adjoint_backwards::<M, <M as DefaultSolver>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
            SolverType::Lu => backwards_method
                .solve_adjoint_backwards::<M, <M as LuValidator<M>>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
            SolverType::Klu => backwards_method
                .solve_adjoint_backwards::<M, <M as KluValidator<M>>::LS, S>(
                    solver.problem(),
                    chk,
                    &g_m,
                    t_eval,
                    Some(1),
                )?,
        };
        Ok((y, y_sens.pop().unwrap()))
    }

    pub(crate) fn solve_adjoint_backwards<'solver, M, LS, S>(
        &self,
        problem: &'solver OdeSolverProblem<DiffSl<M, JitModule>>,
        checkpointing: Checkpointing<'solver, DiffSl<M, JitModule>, S>,
        g_m: &<M::V as DefaultDenseMatrix>::M,
        t_eval: &[M::T],
        nout_override: Option<usize>,
    ) -> Result<Vec<M::V>, DiffsolError>
    where
        M: Matrix<T: DiffSlScalar> + DefaultSolver,
        M::V: VectorHost + DefaultDenseMatrix,
        S: OdeSolverMethod<'solver, DiffSl<M, JitModule>>,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            SolverMethod::Bdf => problem
                .bdf_solver_adjoint::<LS, _>(checkpointing, nout_override)?
                .solve_adjoint_backwards_pass(t_eval, &[g_m])
                .map(|res| res.into_common().sg),
            SolverMethod::Esdirk34 => problem
                .esdirk34_solver_adjoint::<LS, _>(checkpointing, nout_override)?
                .solve_adjoint_backwards_pass(t_eval, &[g_m])
                .map(|res| res.into_common().sg),
            SolverMethod::TrBdf2 => problem
                .tr_bdf2_solver_adjoint::<LS, _>(checkpointing, nout_override)?
                .solve_adjoint_backwards_pass(t_eval, &[g_m])
                .map(|res| res.into_common().sg),
            SolverMethod::Tsit45 => Err(DiffsolError::Other(
                "Tsit45 solver does not support adjoint sensitivity analysis.".to_string(),
            )),
        }
    }
}

#[pymethods]
impl SolverMethod {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "bdf" => Ok(SolverMethod::Bdf),
            "esdirk34" => Ok(SolverMethod::Esdirk34),
            "tr_bdf2" => Ok(SolverMethod::TrBdf2),
            "tsit45" => Ok(SolverMethod::Tsit45),
            _ => Err(PyValueError::new_err("Invalid SolverMethod value")),
        }
    }

    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), SolverMethod::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            SolverMethod::Bdf => 0,
            SolverMethod::Esdirk34 => 1,
            SolverMethod::TrBdf2 => 2,
            SolverMethod::Tsit45 => 3,
        }
    }
}
