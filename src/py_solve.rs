// Delegate solver types selected at runtime in Python to concrete solver types
// in Rust.

use diffsol::{ConstantOp, VectorCommon};
use diffsol::{
    error::DiffsolError, matrix::MatrixRef, DefaultDenseMatrix, DefaultSolver, DiffSl, Matrix,
    MatrixCommon, NonLinearOp, OdeBuilder, OdeEquations, OdeSolverProblem, Op, Vector, VectorHost,
    VectorRef, NonLinearOpJacobian
};
use numpy::PyReadonlyArray2;
use numpy::PyUntypedArrayMethods;
use numpy::{ndarray::Array1, PyArray1, PyArray2, PyReadonlyArray1};
use paste::paste;
use pyo3::{Bound, Python};

use crate::valid_linear_solver::{KluValidator, LuValidator};
use crate::{
    convert::{MatrixToPy, VectorToPy},
    solver_method::SolverMethod,
};
use crate::{
    error::PyDiffsolError, jit::JitModule, matrix_type::MatrixType, solver_type::SolverType,
    valid_linear_solver::validate_linear_solver,
};

// macro to generate all the trait methods for accessing ic_options
macro_rules! generate_trait_ic_option_accessors {
    ($($field:ident : $type:ty),*) => {
        $(
            paste! {
                fn [<set_ic_ $field>](&mut self, value: $type);
                fn [<ic_ $field>](&self) -> $type;
            }
        )*
    };
}

// macro to generate all the trait methods for accessing ode_options
macro_rules! generate_trait_ode_option_accessors {
    ($($field:ident : $type:ty),*) => {
        $(
            paste! {
                fn [<set_ode_ $field>](&mut self, value: $type);
                fn [<ode_ $field>](&self) -> $type;
            }
        )*
    };
}

// macro to generate all the setters and getters for ic_options
macro_rules! generate_ic_option_accessors {
    ($($field:ident : $type:ty),*) => {
        $(
            paste! {
                fn [<set_ic_ $field>](&mut self, value: $type) {
                    self.problem.ic_options.$field = value;
                }

                fn [<ic_ $field>](&self) -> $type {
                    self.problem.ic_options.$field
                }
            }
        )*
    };
}

// macro to generate all the setters and getters for ode_options
macro_rules! generate_ode_option_accessors {
    ($($field:ident : $type:ty),*) => {
        $(
            paste! {
                fn [<set_ode_ $field>](&mut self, value: $type) {
                    self.problem.ode_options.$field = value;
                }
                fn [<ode_ $field>](&self) -> $type {
                    self.problem.ode_options.$field
                }
            }
        )*
    };
}

// Each matrix type implements PySolve as bridge between diffsol and Python
pub(crate) trait PySolve {
    fn matrix_type(&self) -> MatrixType;

    fn rhs<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>, PyDiffsolError>;
    
    fn rhs_jac_mul<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>, PyDiffsolError>;
    
    fn y0<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyArray1<f64>>, PyDiffsolError>;

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
        data: PyReadonlyArray2<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(f64, Bound<'py, PyArray1<f64>>), PyDiffsolError>;

    fn check(&self, linear_solver: SolverType) -> Result<(), PyDiffsolError>;
    fn set_rtol(&mut self, rtol: f64);
    fn rtol(&self) -> f64;
    fn set_atol(&mut self, atol: f64);
    fn atol(&self) -> f64;
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
    for<'b> <M::V as VectorCommon>::Inner: VectorToPy<'b>,
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
    
    fn y0<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyArray1<f64>>, PyDiffsolError> {
        let n = self.problem.eqn.nstates();
        let mut y0= M::V::zeros(n, M::C::default());
        let t0 = self.problem.t0;
        self.problem.eqn.init().call_inplace(t0, &mut y0);
        Ok(y0.inner().to_pyarray1(py))
    }

    fn rhs<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>, PyDiffsolError> {
        let n = self.problem.eqn.nstates();
        
        let y_vec = M::V::from_slice(y.as_slice().unwrap(), M::C::default());
        let mut dydt = M::V::zeros(n, M::C::default());
        self.problem.eqn.rhs().call_inplace(&y_vec, t, &mut dydt);
        Ok(dydt.inner().to_pyarray1(py))
    }
    
    fn rhs_jac_mul<'py>(
            &mut self,
            py: Python<'py>,
            t: f64,
            y: PyReadonlyArray1<'py, f64>,
            v: PyReadonlyArray1<'py, f64>,
        ) -> Result<Bound<'py, PyArray1<f64>>, PyDiffsolError> {
        let n = self.problem.eqn.nstates();
        let y_vec = M::V::from_slice(y.as_slice().unwrap(), M::C::default());
        let v_vec = M::V::from_slice(v.as_slice().unwrap(), M::C::default());
        let mut dydt = M::V::zeros(n, M::C::default());
        self.problem.eqn.rhs().jac_mul_inplace(&y_vec, t, &v_vec, &mut dydt);
        Ok(dydt.inner().to_pyarray1(py))
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

    fn solve_sum_squares_adj<'py>(
        &mut self,
        py: Python<'py>,
        method: SolverMethod,
        linear_solver: SolverType,
        backwards_method: SolverMethod,
        backwards_linear_solver: SolverType,
        params: &[f64],
        data: PyReadonlyArray2<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(f64, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        self.check(linear_solver)?;
        self.setup_problem(params)?;

        let (y, y_sens) = match linear_solver {
            SolverType::Default => method.solve_sum_squares_adj::<M, <M as DefaultSolver>::LS>(
                &mut self.problem,
                data.as_array(),
                t_eval.as_slice().unwrap(),
                backwards_method,
                backwards_linear_solver,
            ),
            SolverType::Lu => method.solve_sum_squares_adj::<M, <M as LuValidator<M>>::LS>(
                &mut self.problem,
                data.as_array(),
                t_eval.as_slice().unwrap(),
                backwards_method,
                backwards_linear_solver,
            ),
            SolverType::Klu => method.solve_sum_squares_adj::<M, <M as KluValidator<M>>::LS>(
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
