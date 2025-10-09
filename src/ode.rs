use std::sync::{Arc, Mutex};

use crate::error::PyDiffsolError;
use crate::ode_config::OdeConfig;
use crate::py_solve::{PySolve, py_solve_factory};
use crate::solve_config::SolveConfig;
use crate::solver_method::SolverMethod;
use crate::solver_type::SolverType;

use numpy::{PyReadonlyArray1, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[pyclass]
struct Ode {
    code: String,
    py_solve: Box<dyn PySolve>,
}
unsafe impl Send for Ode {}
unsafe impl Sync for Ode {}

#[pyclass]
#[pyo3(name = "Ode")]
#[derive(Clone)]
pub struct OdeWrapper(Arc<Mutex<Ode>>);

#[pymethods]
impl OdeWrapper {
    /// Construct an ODE solver for specified diffsol using a given matrix type
    #[new]
    fn new(code: &str, config: &OdeConfig) -> Result<Self, PyDiffsolError> {
        let py_solve = py_solve_factory(code, config)?;
        Ok(OdeWrapper(Arc::new(Mutex::new(
            Ode {
                code: code.to_string(),
                py_solve
            }
        ))))
    }

    /// Get the DiffSl compiled to generate this ODE
    #[getter]
    fn get_code(&self) -> PyResult<String> {
        let guard = self.0.lock().map_err(|_| PyRuntimeError::new_err("Config mutex poisoned"))?;
        Ok(guard.code.clone())
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
    #[pyo3(signature=(params, final_time, config=SolveConfig::new(SolverMethod::Bdf, SolverType::Default)))]
    fn solve<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        final_time: f64,
        config: SolveConfig
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>), PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();

        self_guard.py_solve.solve(
            slf.py(),
            &config,
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
    #[pyo3(signature=(params, t_eval, config=SolveConfig::new(SolverMethod::Bdf, SolverType::Default)))]
    fn solve_dense<'py>(
        slf: PyRefMut<'py, Self>,
        params: PyReadonlyArray1<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
        config: SolveConfig,
    ) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        let mut self_guard = slf.0.lock().unwrap();
        let params = params.as_array();

        self_guard.py_solve.solve_dense(
            slf.py(),
            &config,
            params.as_slice().unwrap(),
            t_eval,
        )
    }
}
