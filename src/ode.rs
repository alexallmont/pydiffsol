// Ode Python class, this wraps up a diffsol Problem class

use std::sync::{Arc, Mutex};

use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    error::PyDiffsolError,
    matrix_type::MatrixType,
    options_ic::InitialConditionSolverOptions,
    options_ode::OdeSolverOptions,
    py_solve::{py_solve_factory, PySolve},
    py_types::{PyReadonlyUntypedArray2, PyUntypedArray1, PyUntypedArray2},
    scalar_type::ScalarType,
    solver_method::SolverMethod,
    solver_type::SolverType,
};

#[pyclass]
pub(crate) struct Ode {
    code: String,
    linear_solver: SolverType,
    method: SolverMethod,
    pub(crate) py_solve: Box<dyn PySolve>,
}
unsafe impl Send for Ode {}
unsafe impl Sync for Ode {}

#[pyclass]
#[pyo3(name = "Ode")]
#[derive(Clone)]
pub struct OdeWrapper(Arc<Mutex<Ode>>);

impl OdeWrapper {
    fn guard(&self) -> PyResult<std::sync::MutexGuard<'_, Ode>> {
        self.0
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Ode mutex poisoned"))
    }
}

#[pymethods]
impl OdeWrapper {
    /// Construct an ODE solver for specified diffsol using a given matrix type.
    /// The code is JIT-compiled immediately based on the matrix type, so after
    /// construction, both code and matrix_type fields are read-only.
    /// All other fields are editable, for example setting the solver type or
    /// method, or changing solver tolerances.
    #[new]
    #[pyo3(signature=(code, matrix_type=MatrixType::NalgebraDense, scalar_type=ScalarType::F64, method=SolverMethod::Bdf, linear_solver=SolverType::Default))]
    fn new(
        code: &str,
        matrix_type: MatrixType,
        scalar_type: ScalarType,
        method: SolverMethod,
        linear_solver: SolverType,
    ) -> Result<Self, PyDiffsolError> {
        let py_solve = py_solve_factory(code, matrix_type, scalar_type)?;
        py_solve.check(linear_solver)?;
        Ok(OdeWrapper(Arc::new(Mutex::new(Ode {
            code: code.to_string(),
            py_solve,
            method,
            linear_solver,
        }))))
    }

    /// Matrix type used in the ODE solver. This is fixed after construction.
    #[getter]
    fn get_matrix_type(&self) -> PyResult<MatrixType> {
        Ok(self.guard()?.py_solve.matrix_type())
    }

    /// Ode solver method, default Bdf (backward differentiation formula).
    #[getter]
    fn get_method(&self) -> PyResult<SolverMethod> {
        Ok(self.guard()?.method)
    }

    #[setter]
    fn set_method(&self, value: SolverMethod) -> PyResult<()> {
        self.guard()?.method = value;
        Ok(())
    }

    /// Linear solver type used in the ODE solver. Set to default to use the
    /// solver's default choice, which is typically an LU solver.
    #[getter]
    fn get_linear_solver(&self) -> PyResult<SolverType> {
        Ok(self.guard()?.linear_solver)
    }

    #[setter]
    fn set_linear_solver(&self, value: SolverType) -> PyResult<()> {
        self.guard()?.py_solve.check(value)?;
        self.guard()?.linear_solver = value;
        Ok(())
    }

    /// Relative tolerance for the solver, default 1e-6. Governs the error relative to the solution size.
    #[getter]
    fn get_rtol(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.rtol())
    }

    #[setter]
    fn set_rtol(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_rtol(value);
        Ok(())
    }

    /// Absolute tolerance for the solver, default 1e-6. Governs the error as the solution goes to zero.
    #[getter]
    fn get_atol(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.atol())
    }

    #[setter]
    fn set_atol(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_atol(value);
        Ok(())
    }

    /// Get the DiffSl compiled to generate this ODE
    #[getter]
    fn get_code(&self) -> PyResult<String> {
        Ok(self.guard()?.code.clone())
    }

    #[getter]
    fn get_ic_options(&self) -> InitialConditionSolverOptions {
        InitialConditionSolverOptions::new(self.0.clone())
    }

    #[getter]
    fn get_options(&self) -> OdeSolverOptions {
        OdeSolverOptions::new(self.0.clone())
    }

    /// Get the initial condition vector y0 as a 1D numpy array.
    fn y0<'py>(slf: PyRefMut<'py, Self>, params: PyReadonlyArray1<'py, f64>) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        self_guard.py_solve.y0(slf.py(), params.as_slice().unwrap())
    }

    /// evaluate the right-hand side function at time `t` and state `y`.
    fn rhs<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        self_guard.py_solve.rhs(slf.py(), params.as_slice().unwrap(), t, y.as_slice().unwrap())
    }

    /// evaluate the right-hand side Jacobian-vector product `Jv`` at time `t` and state `y`.
    fn rhs_jac_mul<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyUntypedArray1>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        self_guard
            .py_solve
            .rhs_jac_mul(slf.py(), params.as_slice().unwrap(), t, y.as_slice().unwrap(), v.as_slice().unwrap())
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
    /// :return: `(ys, ts)` tuple where `ys` is a 2D array of values at times
    ///     `ts` chosen by the solver
    /// :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    ///
    /// Example:
    ///     >>> print(ode.solve(np.array([]), 0.5))
    #[allow(clippy::type_complexity)]
    #[pyo3(signature=(params, final_time))]
    fn solve<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        final_time: f64,
    ) -> Result<(Bound<'py, PyUntypedArray2>, Bound<'py, PyUntypedArray1>), PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();

        let linear_solver = self_guard.linear_solver;
        let method = self_guard.method;
        self_guard.py_solve.solve(
            slf.py(),
            method,
            linear_solver,
            params.as_slice().unwrap(),
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
    /// :return: 2D array of values at times `t_eval`
    /// :rtype: numpy.ndarray
    #[pyo3(signature=(params, t_eval))]
    fn solve_dense<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyUntypedArray2>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();
        let t_eval = t_eval.as_array();

        let linear_solver = self_guard.linear_solver;
        let method = self_guard.method;

        self_guard.py_solve.solve_dense(
            slf.py(),
            method,
            linear_solver,
            params.as_slice().unwrap(),
            t_eval.as_slice().unwrap(),
        )
    }

    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`.
    /// Returns 2D array of solution values at timepoints given by `t_eval`.
    /// Also returns a list of 2D arrays of sensitivities at the same timepoints
    /// as the solution.
    /// The number of params must match the expected params in the diffsl code.
    /// The config may be optionally specified to override solver settings.
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type params: numpy.ndarray
    /// :return: 2D array of values at times `t_eval` and a list of 2D arrays of sensitivities at the same timepoints
    /// :rtype: (numpy.ndarray, List[numpy.ndarray])
    #[allow(clippy::type_complexity)]
    #[pyo3(signature=(params, t_eval))]
    fn solve_fwd_sens<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<
        (
            Bound<'py, PyUntypedArray2>,
            Vec<Bound<'py, PyUntypedArray2>>,
        ),
        PyDiffsolError,
    > {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();
        let t_eval = t_eval.as_array();

        let linear_solver = self_guard.linear_solver;
        let method = self_guard.method;

        self_guard.py_solve.solve_fwd_sens(
            slf.py(),
            method,
            linear_solver,
            params.as_slice().unwrap(),
            t_eval.as_slice().unwrap(),
        )
    }

    /// Using the provided state, solve the adjoint problem for the sum of squares
    /// objective given data at timepoints `t_eval`.
    /// Returns the objective value and a list of 1D arrays of adjoint sensitivities
    /// for each parameter.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature=(params, data, t_eval))]
    fn solve_sum_squares_adj<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        data: Bound<'py, PyReadonlyUntypedArray2>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(f64, Bound<'py, PyUntypedArray1>), PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();
        let t_eval = t_eval.as_array();

        let linear_solver = self_guard.linear_solver;
        let method = self_guard.method;

        self_guard.py_solve.solve_sum_squares_adj(
            slf.py(),
            method,
            linear_solver,
            method,
            linear_solver,
            params.as_slice().unwrap(),
            data,
            t_eval.as_slice().unwrap(),
        )
    }
}
