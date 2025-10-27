// Ode Python class, this wraps up a diffsol Problem class

use std::sync::{Arc, Mutex};

use crate::error::PyDiffsolError;
use crate::matrix_type::MatrixType;
use crate::py_solve::{py_solve_factory, PySolve};
use crate::solver_method::SolverMethod;
use crate::solver_type::SolverType;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
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

    #[getter]
    fn get_matrix_type(&self) -> PyResult<MatrixType> {
        Ok(self.guard()?.py_solve.matrix_type())
    }

    #[getter]
    fn get_method(&self) -> PyResult<SolverMethod> {
        Ok(self.guard()?.method)
    }

    #[setter]
    fn set_method(&self, value: SolverMethod) -> PyResult<()> {
        self.guard()?.method = value;
        Ok(())
    }

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

    #[getter]
    fn get_rtol(&self) -> PyResult<f64> {
        Ok(self.guard()?.py_solve.rtol())
    }

    #[setter]
    fn set_rtol(&self, value: f64) -> PyResult<()> {
        self.guard()?.py_solve.set_rtol(value);
        Ok(())
    }

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
    /// :param config: optional solver configuration
    /// :type config: pydiffsol.Config, optional
    /// :return: `(ys, ts)` tuple where `ys` is a 2D array of values at times
    ///     `ts` chosen by the solver
    /// :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    ///
    /// Example:
    ///     >>> print(ode.solve(np.array([]), 0.5))
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
    /// :param config: optional solver configuration
    /// :type config: pydiffsol.Config, optional
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
}
