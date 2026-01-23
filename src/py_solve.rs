// Delegate solver types selected at runtime in Python to concrete solver types
// in Rust.

use diffsol::{
    error::DiffsolError, matrix::MatrixRef, ConstantOp, DefaultDenseMatrix, DefaultSolver, DiffSl,
    DiffSlScalar, Matrix, MatrixCommon, NonLinearOp, NonLinearOpJacobian, OdeBuilder, OdeEquations,
    OdeSolverProblem, Op, Vector, VectorCommon, VectorHost, VectorRef,
};
use num_traits::{FromPrimitive, ToPrimitive}; // for from_f64 and to_f64
use numpy::{ndarray::Array1, Element, PyArray1};
use paste::paste;
use pyo3::{Bound, Python};

use crate::{
    generate_ic_option_accessors, generate_ode_option_accessors, generate_option_accessors,
    generate_trait_ic_option_accessors, generate_trait_ode_option_accessors,
    option_value_from_store, option_value_to_store,
};

use crate::{
    error::PyDiffsolError,
    jit::JitModule,
    matrix_type::{MatrixKind, MatrixType},
    py_convert::{to_arrayview2, MatrixToPy, VectorToPy},
    py_types::{PyReadonlyUntypedArray2, PyUntypedArray1, PyUntypedArray2},
    scalar_type::ScalarType,
    solver_method::SolverMethod,
    solver_type::SolverType,
    valid_linear_solver::{validate_linear_solver, KluValidator, LuValidator},
};

// Each matrix type implements PySolve as bridge between diffsol and Python
pub(crate) trait PySolve {
    fn matrix_type(&self) -> MatrixType;

    fn rhs<'py>(
        &mut self,
        py: Python<'py>,
        params: &[f64],
        t: f64,
        y: &[f64],
    ) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError>;

    fn rhs_jac_mul<'py>(
        &mut self,
        py: Python<'py>,
        params: &[f64],
        t: f64,
        y: &[f64],
        v: &[f64],
    ) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError>;

    fn y0<'py>(&mut self, py: Python<'py>, params: &[f64]) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError>;

    fn check(&self, linear_solver: SolverType) -> Result<(), PyDiffsolError>;
    fn set_rtol(&mut self, rtol: f64);
    fn rtol(&self) -> f64;
    fn set_atol(&mut self, atol: f64);
    fn atol(&self) -> f64;

    // Result: (2D solution array, 1D timepoints) tuple
    #[allow(clippy::type_complexity)]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyUntypedArray2>, Bound<'py, PyUntypedArray1>), PyDiffsolError>;

    // Result: 2D solution array
    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> Result<Bound<'py, PyUntypedArray2>, PyDiffsolError>;

    // Result: (2D solution array, Vec of 2D sensitivity array per param) tuple
    #[allow(clippy::type_complexity)]
    fn solve_fwd_sens<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> Result<
        (
            Bound<'py, PyUntypedArray2>,
            Vec<Bound<'py, PyUntypedArray2>>,
        ),
        PyDiffsolError,
    >;

    // Result: (sum of squares, 1D timepoints) tuple
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn solve_sum_squares_adj<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        backwards_method: SolverMethod,
        backwards_linear_solver: SolverType,
        params: &[f64],
        data: Bound<'py, PyReadonlyUntypedArray2>,
        t_eval: &[f64],
    ) -> Result<(f64, Bound<'py, PyUntypedArray1>), PyDiffsolError>;

    generate_trait_ic_option_accessors! {
        use_linesearch: bool,
        max_linesearch_iterations: usize,
        max_newton_iterations: usize,
        max_linear_solver_setups: usize,
        step_reduction_factor: f64,
        armijo_constant: f64
    }
    generate_trait_ode_option_accessors! {
        max_nonlinear_solver_iterations: usize,
        max_error_test_failures: usize,
        min_timestep: f64,
        update_jacobian_after_steps: usize,
        update_rhs_jacobian_after_steps: usize,
        threshold_to_update_jacobian: f64,
        threshold_to_update_rhs_jacobian: f64
    }
}

// Public factory method for generating an instance based on matrix type
pub(crate) fn py_solve_factory(
    code: &str,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn PySolve>, PyDiffsolError> {
    let py_solve: Box<dyn PySolve> = match matrix_type {
        MatrixType::NalgebraDense => match scalar_type {
            ScalarType::F32 => Box::new(GenericPySolve::<diffsol::NalgebraMat<f32>>::new(code)?),
            ScalarType::F64 => Box::new(GenericPySolve::<diffsol::NalgebraMat<f64>>::new(code)?),
        },
        MatrixType::FaerDense => match scalar_type {
            ScalarType::F32 => Box::new(GenericPySolve::<diffsol::FaerMat<f32>>::new(code)?),
            ScalarType::F64 => Box::new(GenericPySolve::<diffsol::FaerMat<f64>>::new(code)?),
        },
        MatrixType::FaerSparse => match scalar_type {
            ScalarType::F32 => Box::new(GenericPySolve::<diffsol::FaerSparseMat<f32>>::new(code)?),
            ScalarType::F64 => Box::new(GenericPySolve::<diffsol::FaerSparseMat<f64>>::new(code)?),
        },
    };
    Ok(py_solve)
}

pub(crate) struct GenericPySolve<M>
where
    M: Matrix<T: DiffSlScalar>,
    M::V: VectorHost,
{
    problem: OdeSolverProblem<DiffSl<M, JitModule>>,
}

impl<M> GenericPySolve<M>
where
    M: Matrix<T: DiffSlScalar + Element>,
    M::V: VectorHost,
{
    pub fn new(code: &str) -> Result<Self, PyDiffsolError> {
        let problem = OdeBuilder::<M>::new().build_from_diffsl::<JitModule>(code)?;
        Ok(GenericPySolve { problem })
    }

    pub(crate) fn setup_problem(&mut self, params: &[f64]) -> Result<(), PyDiffsolError> {
        let params: Vec<M::T> = params.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let params = M::V::from_slice(&params, M::C::default());

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
    M: Matrix<T: DiffSlScalar + numpy::Element + ToPrimitive>
        + DefaultSolver
        + LuValidator<M>
        + KluValidator<M>
        + MatrixKind,
    for<'b> <<M::V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: MatrixToPy<'b, M::T>,
    for<'b> <M::V as VectorCommon>::Inner: VectorToPy<'b, M::T>,
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

    fn set_atol(&mut self, atol: f64) {
        self.problem.atol.fill(M::T::from_f64(atol).unwrap());
    }

    fn atol(&self) -> f64 {
        self.problem.atol[0].to_f64().unwrap()
    }

    fn set_rtol(&mut self, rtol: f64) {
        self.problem.rtol = M::T::from_f64(rtol).unwrap();
    }

    fn rtol(&self) -> f64 {
        self.problem.rtol.to_f64().unwrap()
    }

    generate_ic_option_accessors! {
        use_linesearch: bool,
        max_linesearch_iterations: usize,
        max_newton_iterations: usize,
        max_linear_solver_setups: usize,
        step_reduction_factor: f64,
        armijo_constant: f64
    }

    generate_ode_option_accessors! {
        max_nonlinear_solver_iterations: usize,
        max_error_test_failures: usize,
        min_timestep: f64,
        update_jacobian_after_steps: usize,
        update_rhs_jacobian_after_steps: usize,
        threshold_to_update_jacobian: f64,
        threshold_to_update_rhs_jacobian: f64
    }

    fn y0<'py>(&mut self, py: Python<'py>, params: &[f64]) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let mut y0 = M::V::zeros(n, M::C::default());
        let t0 = self.problem.t0;
        self.problem.eqn.init().call_inplace(t0, &mut y0);
        Ok(y0.inner().to_pyarray1(py).into_any())
    }

    fn rhs<'py>(
        &mut self,
        py: Python<'py>,
        params: &[f64],
        t: f64,
        y: &[f64],
    ) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let y = y
            .iter()
            .map(|&x| M::T::from_f64(x).unwrap())
            .collect::<Vec<_>>();
        let y_vec = M::V::from_slice(&y, M::C::default());
        let mut dydt = M::V::zeros(n, M::C::default());
        self.problem
            .eqn
            .rhs()
            .call_inplace(&y_vec, M::T::from_f64(t).unwrap(), &mut dydt);
        Ok(dydt.inner().to_pyarray1(py).into_any())
    }

    fn rhs_jac_mul<'py>(
        &mut self,
        py: Python<'py>,
        params: &[f64],
        t: f64,
        y: &[f64],
        v: &[f64],
    ) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError> {
        self.setup_problem(params)?;
        let n = self.problem.eqn.nstates();
        let y = y
            .iter()
            .map(|&x| M::T::from_f64(x).unwrap())
            .collect::<Vec<_>>();
        let v = v
            .iter()
            .map(|&x| M::T::from_f64(x).unwrap())
            .collect::<Vec<_>>();
        let y_vec = M::V::from_slice(&y, M::C::default());
        let v_vec = M::V::from_slice(&v, M::C::default());
        let mut dydt = M::V::zeros(n, M::C::default());
        self.problem.eqn.rhs().jac_mul_inplace(
            &y_vec,
            M::T::from_f64(t).unwrap(),
            &v_vec,
            &mut dydt,
        );
        Ok(dydt.inner().to_pyarray1(py).into_any())
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        final_time: f64,
    ) -> Result<(Bound<'py, PyUntypedArray2>, Bound<'py, PyUntypedArray1>), PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;
        let final_time = M::T::from_f64(final_time).unwrap();
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
            ys.inner().to_pyarray2(py).into_any(),
            PyArray1::from_owned_array(py, Array1::from(ts)).into_any(),
        ))
    }

    fn solve_dense<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> Result<Bound<'py, PyUntypedArray2>, PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let ys = match linear_solver {
            SolverType::Default => {
                method.solve_dense::<M, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval)
            }
            SolverType::Lu => {
                method.solve_dense::<M, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval)
            }
            SolverType::Klu => {
                method.solve_dense::<M, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval)
            }
        }?;

        Ok(ys.inner().to_pyarray2(py).into_any())
    }

    fn solve_fwd_sens<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        params: &[f64],
        t_eval: &[f64],
    ) -> Result<
        (
            Bound<'py, PyUntypedArray2>,
            Vec<Bound<'py, PyUntypedArray2>>,
        ),
        PyDiffsolError,
    > {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();
        let (ys, sens) =
            match linear_solver {
                SolverType::Default => {
                    method.solve_fwd_sens::<M, <M as DefaultSolver>::LS>(&mut self.problem, &t_eval)
                }
                SolverType::Lu => method
                    .solve_fwd_sens::<M, <M as LuValidator<M>>::LS>(&mut self.problem, &t_eval),
                SolverType::Klu => method
                    .solve_fwd_sens::<M, <M as KluValidator<M>>::LS>(&mut self.problem, &t_eval),
            }?;

        Ok((
            ys.inner().to_pyarray2(py).into_any(),
            sens.into_iter()
                .map(|s| s.inner().to_pyarray2(py).into_any())
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
        params: &[f64],
        data: Bound<'py, PyReadonlyUntypedArray2>,
        t_eval: &[f64],
    ) -> Result<(f64, Bound<'py, PyUntypedArray1>), PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let data = to_arrayview2::<M::T>(&data)?;
        let t_eval: Vec<M::T> = t_eval.iter().map(|&x| M::T::from_f64(x).unwrap()).collect();

        let (y, y_sens) = match linear_solver {
            SolverType::Default => method.solve_sum_squares_adj::<M, <M as DefaultSolver>::LS>(
                &mut self.problem,
                data,
                &t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            SolverType::Lu => method.solve_sum_squares_adj::<M, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                data,
                &t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
            SolverType::Klu => method.solve_sum_squares_adj::<M, <M as KluValidator<M>>::LS>(
                &mut self.problem,
                data,
                &t_eval,
                backwards_method,
                backwards_linear_solver,
            ),
        }?;

        Ok((
            y.to_f64().unwrap(),
            y_sens.inner().to_pyarray1(py).into_any(),
        ))
    }
}
