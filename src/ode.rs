// Wrap diffsol-c ode solver type with a Python class. This is the main user-facing
// class for ODE solving in pydiffsol, including all creation, configuration and solving
// methods. The ODE solver is JIT-compiled on construction.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyTuple, PyType},
    PyAny,
};

use crate::{
    adjoint_checkpoint::AdjointCheckpointWrapper,
    error::PyDiffsolError,
    host_array::{
        host_array_to_py, pyarray1_to_host_f64, pyarray2_to_host_f32, pyarray2_to_host_f64,
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

#[pyclass(module = "pydiffsol")]
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

    fn serialize_state_bytes(&self) -> Result<Vec<u8>, PyDiffsolError> {
        serde_json::to_vec(&self.0).map_err(|err| PyRuntimeError::new_err(err.to_string()).into())
    }

    fn deserialize_state_bytes(state: &[u8]) -> Result<Self, PyDiffsolError> {
        serde_json::from_slice(state)
            .map(Self)
            .map_err(|err| PyValueError::new_err(err.to_string()).into())
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

    #[classmethod]
    fn _from_state_bytes(_cls: &Bound<'_, PyType>, state: Vec<u8>) -> Result<Self, PyDiffsolError> {
        Self::deserialize_state_bytes(state.as_slice())
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Result<Self, PyDiffsolError> {
        let state = self.serialize_state_bytes()?;
        Self::deserialize_state_bytes(state.as_slice())
    }

    /// Serialize this solver for Python pickling.
    ///
    /// The pickle payload uses diffsol-c's native serde-based serialization,
    /// pickling/unpickling can restore an equivalent solver much faster than
    /// JIT compiling twice.
    fn __getstate__(&self) -> Result<Vec<u8>, PyDiffsolError> {
        self.serialize_state_bytes()
    }

    /// Restore a solver from bytes produced by `__getstate__`.
    ///
    /// See `__getstate__` for picklng details.
    fn __setstate__(&mut self, state: Vec<u8>) -> Result<(), PyDiffsolError> {
        *self = Self::deserialize_state_bytes(state.as_slice())?;
        Ok(())
    }

    /// Implement the Python pickle protocol for solver.
    ///
    /// See `__getstate__` for picklng details.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> Result<(Bound<'py, PyAny>, Bound<'py, PyTuple>), PyDiffsolError> {
        let cls = slf.getattr("__class__")?;
        let reconstructor = cls.getattr("_from_state_bytes")?;
        let state = slf.borrow().serialize_state_bytes()?;
        Ok((reconstructor, PyTuple::new(py, [state])?))
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

    /// Initial time for the solver, default 0.0.
    #[getter]
    fn get_t0(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_t0()?)
    }

    #[setter]
    fn set_t0(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_t0(value)?;
        Ok(())
    }

    /// Initial step size for the solver, default 1.0.
    #[getter]
    fn get_h0(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_h0()?)
    }

    #[setter]
    fn set_h0(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_h0(value)?;
        Ok(())
    }

    /// Whether to integrate output equations alongside state equations.
    #[getter]
    fn get_integrate_out(&self) -> Result<bool, PyDiffsolError> {
        Ok(self.0.get_integrate_out()?)
    }

    #[setter]
    fn set_integrate_out(&self, value: bool) -> Result<(), PyDiffsolError> {
        self.0.set_integrate_out(value)?;
        Ok(())
    }

    /// Relative tolerance for forward sensitivity or adjoint equations.
    #[getter]
    fn get_sens_rtol(&self) -> Result<Option<f64>, PyDiffsolError> {
        Ok(self.0.get_sens_rtol()?)
    }

    #[setter]
    fn set_sens_rtol(&self, value: Option<f64>) -> Result<(), PyDiffsolError> {
        self.0.set_sens_rtol(value)?;
        Ok(())
    }

    /// Absolute tolerance for forward sensitivity or adjoint equations.
    #[getter]
    fn get_sens_atol(&self) -> Result<Option<f64>, PyDiffsolError> {
        Ok(self.0.get_sens_atol()?)
    }

    #[setter]
    fn set_sens_atol(&self, value: Option<f64>) -> Result<(), PyDiffsolError> {
        self.0.set_sens_atol(value)?;
        Ok(())
    }

    /// Relative tolerance for integrated output equations.
    #[getter]
    fn get_out_rtol(&self) -> Result<Option<f64>, PyDiffsolError> {
        Ok(self.0.get_out_rtol()?)
    }

    #[setter]
    fn set_out_rtol(&self, value: Option<f64>) -> Result<(), PyDiffsolError> {
        self.0.set_out_rtol(value)?;
        Ok(())
    }

    /// Absolute tolerance for integrated output equations.
    #[getter]
    fn get_out_atol(&self) -> Result<Option<f64>, PyDiffsolError> {
        Ok(self.0.get_out_atol()?)
    }

    #[setter]
    fn set_out_atol(&self, value: Option<f64>) -> Result<(), PyDiffsolError> {
        self.0.set_out_atol(value)?;
        Ok(())
    }

    /// Relative tolerance for adjoint parameter gradient equations.
    #[getter]
    fn get_param_rtol(&self) -> Result<Option<f64>, PyDiffsolError> {
        Ok(self.0.get_param_rtol()?)
    }

    #[setter]
    fn set_param_rtol(&self, value: Option<f64>) -> Result<(), PyDiffsolError> {
        self.0.set_param_rtol(value)?;
        Ok(())
    }

    /// Absolute tolerance for adjoint parameter gradient equations.
    #[getter]
    fn get_param_atol(&self) -> Result<Option<f64>, PyDiffsolError> {
        Ok(self.0.get_param_atol()?)
    }

    #[setter]
    fn set_param_atol(&self, value: Option<f64>) -> Result<(), PyDiffsolError> {
        self.0.set_param_atol(value)?;
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
        host_array_to_py(py, self.0.y0(pyarray1_to_host_f64(params)?)?)
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
                .rhs(pyarray1_to_host_f64(params)?, t, pyarray1_to_host_f64(y)?)?,
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
                pyarray1_to_host_f64(params)?,
                t,
                pyarray1_to_host_f64(y)?,
                pyarray1_to_host_f64(v)?,
            )?,
        )
    }

    /// Solve the problem up to time `final_time`.
    /// Stop/reset events defined in the DiffSL code are handled automatically.
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
            self.0.solve(pyarray1_to_host_f64(params)?, final_time)?,
        ))
    }

    /// Solve the problem up to time `t_eval[t_eval.len()-1]`.
    /// Returns a `Solution` object with values at timepoints given by `t_eval`.
    /// Stop/reset events defined in the DiffSL code are handled automatically.
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
            pyarray1_to_host_f64(params)?,
            pyarray1_to_host_f64(t_eval)?,
        )?))
    }

    /// Solve the problem up to time `t_eval[t_eval.len()-1]`.
    /// Returns a `Solution` object with values at `t_eval` and sensitivity arrays
    /// at the same timepoints.
    /// Stop/reset events defined in the DiffSL code are handled automatically.
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
            pyarray1_to_host_f64(params)?,
            pyarray1_to_host_f64(t_eval)?,
        )?))
    }

    /// Solve the continuous adjoint problem for the integral of the model output
    /// from the initial time to `final_time`.
    ///
    /// :return: tuple of (integral, gradient)
    /// :rtype: tuple[numpy.ndarray, numpy.ndarray]
    fn solve_continuous_adjoint<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
        final_time: f64,
    ) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>), PyDiffsolError> {
        let (integral, gradient) = self
            .0
            .solve_continuous_adjoint(pyarray1_to_host_f64(params)?, final_time)?;
        Ok((
            host_array_to_py(py, integral)?,
            host_array_to_py(py, gradient)?,
        ))
    }

    /// Solve the forward problem at `t_eval` and retain checkpoint data for a
    /// later discrete adjoint backward pass.
    ///
    /// :return: tuple of (solution, checkpoint)
    /// :rtype: tuple[Solution, AdjointCheckpoint]
    fn solve_adjoint_fwd(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        t_eval: PyReadonlyArray1<'_, f64>,
    ) -> Result<(SolutionWrapper, AdjointCheckpointWrapper), PyDiffsolError> {
        let (solution, checkpoint) = self
            .0
            .solve_adjoint_fwd(pyarray1_to_host_f64(params)?, pyarray1_to_host_f64(t_eval)?)?;
        Ok((
            SolutionWrapper::new(solution),
            AdjointCheckpointWrapper::new(checkpoint),
        ))
    }

    /// Solve the discrete adjoint backward pass using a prior forward adjoint
    /// checkpoint and the gradient of a scalar objective with respect to model
    /// outputs at each saved evaluation time.
    ///
    /// :param dgdu_eval: 2D array, shape (nout, len(solution.ts)); float32 or float64
    /// :type dgdu_eval: numpy.ndarray
    /// :return: parameter gradient matrix
    /// :rtype: numpy.ndarray
    fn solve_adjoint_bkwd<'py>(
        &self,
        py: Python<'py>,
        solution: &SolutionWrapper,
        checkpoint: &AdjointCheckpointWrapper,
        dgdu_eval: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        let scalar_type: ScalarType = self.0.get_scalar_type()?.into();
        let gradient = if let Ok(dgdu_f32) = dgdu_eval.extract::<PyReadonlyArray2<f32>>() {
            match scalar_type {
                ScalarType::F32 => self.0.solve_adjoint_bkwd(
                    solution.inner(),
                    checkpoint.inner(),
                    pyarray2_to_host_f32(dgdu_f32)?,
                )?,
                ScalarType::F64 => {
                    let (_owned_dgdu, dgdu_host) = pyarray2_to_owned_f64_host(dgdu_f32)?;
                    self.0
                        .solve_adjoint_bkwd(solution.inner(), checkpoint.inner(), dgdu_host)?
                }
            }
        } else if let Ok(dgdu_f64) = dgdu_eval.extract::<PyReadonlyArray2<f64>>() {
            match scalar_type {
                ScalarType::F32 => {
                    let (_owned_dgdu, dgdu_host) = pyarray2_to_owned_f32_host(dgdu_f64)?;
                    self.0
                        .solve_adjoint_bkwd(solution.inner(), checkpoint.inner(), dgdu_host)?
                }
                ScalarType::F64 => self.0.solve_adjoint_bkwd(
                    solution.inner(),
                    checkpoint.inner(),
                    pyarray2_to_host_f64(dgdu_f64)?,
                )?,
            }
        } else {
            return Err(PyDiffsolError::Conversion(
                "dgdu_eval must be a 2D NumPy array of float32 or float64".to_string(),
            ));
        };
        host_array_to_py(py, gradient)
    }
}
