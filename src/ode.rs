use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, PyAny};

use crate::{
    error::PyDiffsolError,
    host_array::{
        host_array_to_py, pyarray1_to_host, pyarray2_to_host_f32, pyarray2_to_host_f64,
        pyarray2_to_owned_f32_host, pyarray2_to_owned_f64_host,
    },
    jit::JitBackendType,
    linear_solver_type::LinearSolverType,
    matrix_type::MatrixType,
    ode_solver_type::OdeSolverType,
    options_ic::InitialConditionSolverOptions,
    options_ode::OdeSolverOptions,
    scalar_type::ScalarType,
    solution::SolutionWrapper,
};

#[pyclass]
#[pyo3(name = "Ode")]
#[derive(Clone)]
pub struct OdeWrapper(diffsol_c::OdeWrapper);

impl OdeWrapper {
    fn resolve_jit_backend(
        jit_backend: Option<JitBackendType>,
    ) -> Result<diffsol_c::JitBackendType, PyDiffsolError> {
        match jit_backend {
            Some(jit_backend) => Ok(jit_backend.into()),
            None => diffsol_c::default_enabled_jit_backend().ok_or_else(|| {
                PyDiffsolError::Conversion(
                    "No default JIT backend is available; pass jit_backend explicitly".to_string(),
                )
            }),
        }
    }
}

#[pymethods]
impl OdeWrapper {
    /// Construct an ODE solver for specified diffsol using a given matrix type.
    /// The code is JIT-compiled immediately based on the matrix type and jit_backend, 
    /// so after construction, both code and matrix_type fields are read-only.
    /// All other fields are editable, for example setting the solver type or
    /// method, or changing solver tolerances.
    #[new]
    #[pyo3(signature=(code, jit_backend=None, scalar_type=ScalarType::F64, matrix_type=MatrixType::NalgebraDense, linear_solver=LinearSolverType::Default, ode_solver=OdeSolverType::Bdf))]
    fn new(
        code: &str,
        jit_backend: Option<JitBackendType>,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, PyDiffsolError> {
        let inner = diffsol_c::OdeWrapper::new_jit(
            code,
            Self::resolve_jit_backend(jit_backend)?,
            scalar_type.into(),
            matrix_type.into(),
            linear_solver.into(),
            ode_solver.into(),
        )?;
        Ok(Self(inner))
    }

    /// Matrix type used in the ODE solver. This is fixed after construction.
    #[getter]
    fn get_matrix_type(&self) -> Result<MatrixType, PyDiffsolError> {
        Ok(self.0.get_matrix_type()?.into())
    }

    /// Scalar type used in the ODE solver. This is fixed after construction.
    #[getter]
    fn get_scalar_type(&self) -> Result<ScalarType, PyDiffsolError> {
        Ok(self.0.get_scalar_type()?.into())
    }

    /// JIT backend used for compilation. This is fixed after construction.
    #[getter]
    fn get_jit_backend(&self) -> Result<Option<JitBackendType>, PyDiffsolError> {
        Ok(self.0.get_jit_backend()?.map(Into::into))
    }

    /// ODE solver method, default Bdf (backward differentiation formula).
    #[getter]
    fn get_ode_solver(&self) -> Result<OdeSolverType, PyDiffsolError> {
        Ok(self.0.get_ode_solver()?.into())
    }

    #[setter]
    fn set_ode_solver(&self, value: OdeSolverType) -> Result<(), PyDiffsolError> {
        self.0.set_ode_solver(value.into())?;
        Ok(())
    }

    /// Linear solver type used in the ODE solver. Set to default to use the
    /// solver's default choice, which is typically an LU solver.
    #[getter]
    fn get_linear_solver(&self) -> Result<LinearSolverType, PyDiffsolError> {
        Ok(self.0.get_linear_solver()?.into())
    }

    #[setter]
    fn set_linear_solver(&self, value: LinearSolverType) -> Result<(), PyDiffsolError> {
        self.0.set_linear_solver(value.into())?;
        Ok(())
    }

    /// Relative tolerance for the solver, default 1e-6. Governs the error relative to the solution size.
    #[getter]
    fn get_rtol(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_rtol()?)
    }

    #[setter]
    fn set_rtol(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_rtol(value)?;
        Ok(())
    }

    /// Absolute tolerance for the solver, default 1e-6. Governs the error as the solution goes to zero.
    #[getter]
    fn get_atol(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_atol()?)
    }

    #[setter]
    fn set_atol(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_atol(value)?;
        Ok(())
    }

    /// Get the DiffSl code used to generate this ODE.
    #[getter]
    fn get_code(&self) -> Result<String, PyDiffsolError> {
        Ok(self.0.get_code()?)
    }

    /// Get the number of parameters expected by the diffsl code.
    #[getter]
    fn get_nparams(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_nparams()?)
    }

    /// Get the number of states in the ODE system.
    #[getter]
    fn get_nstates(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_nstates()?)
    }

    /// Get the number of outputs in the ODE system.
    /// If there is no out tensor, this is the number of states.
    #[getter]
    fn get_nout(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_nout()?)
    }

    /// Check if the diffsl code has a stop event defined.
    fn has_stop(&self) -> Result<bool, PyDiffsolError> {
        Ok(self.0.has_stop()?)
    }

    #[getter]
    fn get_ic_options(&self) -> InitialConditionSolverOptions {
        InitialConditionSolverOptions::new(self.0.get_ic_options())
    }

    #[getter]
    fn get_options(&self) -> OdeSolverOptions {
        OdeSolverOptions::new(self.0.get_options())
    }

    /// Get the initial condition vector y0 as a 1D numpy array.
    fn y0<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(py, self.0.y0(pyarray1_to_host(params)?)?)
    }

    /// Evaluate the right-hand side function at time `t` and state `y`.
    fn rhs<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(
            py,
            self.0
                .rhs(pyarray1_to_host(params)?, t, pyarray1_to_host(y)?)?,
        )
    }

    /// Evaluate the right-hand side Jacobian-vector product `Jv` at time `t` and state `y`.
    fn rhs_jac_mul<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
        t: f64,
        y: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(
            py,
            self.0.rhs_jac_mul(
                pyarray1_to_host(params)?,
                t,
                pyarray1_to_host(y)?,
                pyarray1_to_host(v)?,
            )?,
        )
    }

    /// Solve the problem up to time `final_time`.
    ///
    /// The number of params must match the expected params in the diffsl code.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param final_time: end time of solver
    /// :type final_time: float
    /// :return: `Solution` object with fields `ys` and `ts`
    /// :rtype: Solution
    ///
    /// Example:
    ///     >>> print(ode.solve(np.array([]), 0.5))
    fn solve(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        final_time: f64,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(
            self.0.solve(pyarray1_to_host(params)?, final_time)?,
        ))
    }

    /// Solve the problem up to time `final_time`, stopping and restarting at
    /// each stop event defined in the diffsl code.
    ///
    /// The number of params must match the expected params in the diffsl code.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param final_time: end time of solver
    /// :type final_time: float
    /// :return: `Solution` object with fields `ys` and `ts`
    /// :rtype: Solution
    fn solve_hybrid(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        final_time: f64,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(
            self.0.solve_hybrid(pyarray1_to_host(params)?, final_time)?,
        ))
    }

    /// Solve the problem up to time `t_eval[t_eval.len()-1]`.
    /// Returns a `Solution` object with values at timepoints given by `t_eval`.
    ///
    /// The number of params must match the expected params in the diffsl code.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type t_eval: numpy.ndarray
    /// :return: `Solution` object with fields `ys` and `ts`
    /// :rtype: Solution
    fn solve_dense(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        t_eval: PyReadonlyArray1<'_, f64>,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(self.0.solve_dense(
            pyarray1_to_host(params)?,
            pyarray1_to_host(t_eval)?,
        )?))
    }

    /// Solve the problem up to time `t_eval[t_eval.len()-1]`, stopping and
    /// restarting at each stop event defined in the diffsl code.
    /// Returns a `Solution` object with values at timepoints given by `t_eval`.
    ///
    /// The number of params must match the expected params in the diffsl code.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type t_eval: numpy.ndarray
    /// :return: `Solution` object with fields `ys` and `ts`
    /// :rtype: Solution
    fn solve_hybrid_dense(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        t_eval: PyReadonlyArray1<'_, f64>,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(self.0.solve_hybrid_dense(
            pyarray1_to_host(params)?,
            pyarray1_to_host(t_eval)?,
        )?))
    }

    /// Solve the problem up to time `t_eval[t_eval.len()-1]`.
    /// Returns a `Solution` object with values at `t_eval` and sensitivity arrays
    /// at the same timepoints.
    ///
    /// The number of params must match the expected params in the diffsl code.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type t_eval: numpy.ndarray
    /// :return: `Solution` object with fields `ys`, `ts`, and `sens`
    /// :rtype: Solution
    fn solve_fwd_sens(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        t_eval: PyReadonlyArray1<'_, f64>,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(self.0.solve_fwd_sens(
            pyarray1_to_host(params)?,
            pyarray1_to_host(t_eval)?,
        )?))
    }

    /// Solve the problem up to time `t_eval[t_eval.len()-1]`, stopping and
    /// restarting at each stop event defined in the diffsl code.
    /// Returns a `Solution` object with values at `t_eval` and sensitivity arrays
    /// at the same timepoints.
    ///
    /// The number of params must match the expected params in the diffsl code.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type t_eval: numpy.ndarray
    /// :return: `Solution` object with fields `ys`, `ts`, and `sens`
    /// :rtype: Solution
    fn solve_hybrid_fwd_sens(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        t_eval: PyReadonlyArray1<'_, f64>,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(self.0.solve_hybrid_fwd_sens(
            pyarray1_to_host(params)?,
            pyarray1_to_host(t_eval)?,
        )?))
    }

    /// Solve the adjoint problem for the sum of squares objective given data
    /// at timepoints `t_eval`.
    /// Returns the objective value and a 1D array of adjoint sensitivities
    /// for each parameter.
    ///
    /// `data` may be a float32 or float64 NumPy array. When the ODE was
    /// constructed with ``scalar_type=f32`` and float64 data is supplied, the
    /// data is automatically cast to float32.
    ///
    /// :param params: 1D array of solver parameters
    /// :type params: numpy.ndarray
    /// :param data: 2D array of observed data, shape (nout, len(t_eval)); float32 or float64
    /// :type data: numpy.ndarray
    /// :param t_eval: 1D array of solver times
    /// :type t_eval: numpy.ndarray
    /// :return: tuple of (objective value, 1D array of sensitivities)
    /// :rtype: tuple[float, numpy.ndarray]
    fn solve_sum_squares_adj<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
        data: &Bound<'py, PyAny>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(f64, Bound<'py, PyAny>), PyDiffsolError> {
        let params_host = pyarray1_to_host(params)?;
        let t_eval_host = pyarray1_to_host(t_eval)?;
        let scalar_type: ScalarType = self.0.get_scalar_type()?.into();
        let (value, sens) = if let Ok(data_f32) = data.extract::<PyReadonlyArray2<f32>>() {
            match scalar_type {
                ScalarType::F32 => self.0.solve_sum_squares_adj(
                    params_host,
                    pyarray2_to_host_f32(data_f32)?,
                    t_eval_host,
                )?,
                ScalarType::F64 => {
                    let (_owned_data, data_host) = pyarray2_to_owned_f64_host(data_f32)?;
                    self.0.solve_sum_squares_adj(params_host, data_host, t_eval_host)?
                }
            }
        } else if let Ok(data_f64) = data.extract::<PyReadonlyArray2<f64>>() {
            match scalar_type {
                ScalarType::F32 => {
                    let (_owned_data, data_host) = pyarray2_to_owned_f32_host(data_f64)?;
                    self.0.solve_sum_squares_adj(params_host, data_host, t_eval_host)?
                }
                ScalarType::F64 => self.0.solve_sum_squares_adj(
                    params_host,
                    pyarray2_to_host_f64(data_f64)?,
                    t_eval_host,
                )?,
            }
        } else {
            return Err(PyDiffsolError::Conversion(
                "data must be a 2D NumPy array of float32 or float64".to_string(),
            ));
        };
        Ok((value, host_array_to_py(py, sens)?))
    }
}
