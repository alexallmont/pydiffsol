// Ode Python class, this wraps up a diffsol Problem class

use std::sync::{Arc, Mutex};

use crate::error::PyDiffsolError;
use crate::matrix_type::MatrixType;
use crate::py_solve::{py_solve_factory, PySolve};
use crate::solver_method::SolverMethod;
use crate::solver_type::SolverType;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass]
struct Ode {
    code: String,
    linear_solver: SolverType,
    method: SolverMethod,
    py_solve: Box<dyn PySolve>,
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
    #[pyo3(signature=(code, matrix_type=MatrixType::NalgebraDenseF64, method=SolverMethod::Bdf, linear_solver=SolverType::Default))]
    fn new(
        code: &str,
        matrix_type: MatrixType,
        method: SolverMethod,
        linear_solver: SolverType,
    ) -> Result<Self, PyDiffsolError> {
        let py_solve = py_solve_factory(code, matrix_type)?;
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

    #[getter]
    fn get_ic_use_linesearch(&self) -> PyResult<bool> {
        Ok(self.guard()?.py_solve.ic_use_linesearch())
    }
    #[setter]
    fn set_ic_use_linesearch(&self, value: bool) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_use_linesearch(value);
        Ok(())
    }
    #[getter]
    fn get_ic_max_linesearch_iterations(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ic_max_linesearch_iterations())
    }
    #[setter]
    fn set_ic_max_linesearch_iterations(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ic_max_linesearch_iterations(value);
        Ok(())
    }
    #[getter]
    fn get_ic_max_newton_iterations(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ic_max_newton_iterations())
    }
    #[setter]
    fn set_ic_max_newton_iterations(&self, value: usize) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_max_newton_iterations(value);
        Ok(())
    }
    #[getter]
    fn get_ic_max_linear_solver_setups(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ic_max_linear_solver_setups())
    }
    #[setter]
    fn set_ic_max_linear_solver_setups(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ic_max_linear_solver_setups(value);
        Ok(())
    }
    #[getter]
    fn get_ic_step_reduction_factor(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.ic_step_reduction_factor())
    }
    #[setter]
    fn set_ic_step_reduction_factor(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_step_reduction_factor(value);
        Ok(())
    }
    #[getter]
    fn get_ic_armijo_constant(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.ic_armijo_constant())
    }
    #[setter]
    fn set_ic_armijo_constant(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_ic_armijo_constant(value);
        Ok(())
    }
    #[getter]
    fn get_max_nonlinear_solver_iterations(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ode_max_nonlinear_solver_iterations())
    }
    #[setter]
    fn set_max_nonlinear_solver_iterations(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_max_nonlinear_solver_iterations(value);
        Ok(())
    }
    #[getter]
    fn get_max_error_test_failures(&self) -> PyResult<usize> {
        Ok(self.guard()?.py_solve.ode_max_error_test_failures())
    }
    #[setter]
    fn set_max_error_test_failures(&self, value: usize) -> PyResult<()> {
        self.guard()?
            .py_solve
            .set_ode_max_error_test_failures(value);
        Ok(())
    }
    #[getter]
    fn get_min_timestep(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.ode_min_timestep())
    }
    #[setter]
    fn set_min_timestep(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_ode_min_timestep(value);
        Ok(())
    }

    /// Get the DiffSl compiled to generate this ODE
    #[getter]
    fn get_code(&self) -> PyResult<String> {
        Ok(self.guard()?.code.clone())
    }

    /// evaluate the right-hand side function at time `t` and state `y`.
    fn rhs<'py>(
        slf: PyRefMut<'py, Self>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        self_guard.py_solve.rhs(slf.py(), t, y)
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
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
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
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();

        let linear_solver = self_guard.linear_solver;
        let method = self_guard.method;

        self_guard.py_solve.solve_dense(
            slf.py(),
            method,
            linear_solver,
            params.as_slice().unwrap(),
            t_eval,
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
    ) -> Result<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>), PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();
        let linear_solver = self_guard.linear_solver;
        let method = self_guard.method;
        self_guard.py_solve.solve_fwd_sens(
            slf.py(),
            method,
            linear_solver,
            params.as_slice().unwrap(),
            t_eval,
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
        data: PyReadonlyArray2<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(f64, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();
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
            t_eval,
        )
    }
}
