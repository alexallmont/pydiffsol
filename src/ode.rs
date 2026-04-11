use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, PyAny};

use crate::{
    error::PyDiffsolError,
    host_array::{
        host_array_to_py, pyarray1_to_host, pyarray2_to_host_f64, pyarray2_to_owned_f32_host,
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

    #[getter]
    fn get_matrix_type(&self) -> Result<MatrixType, PyDiffsolError> {
        Ok(self.0.get_matrix_type()?.into())
    }

    #[getter]
    fn get_scalar_type(&self) -> Result<ScalarType, PyDiffsolError> {
        Ok(self.0.get_scalar_type()?.into())
    }

    #[getter]
    fn get_jit_backend(&self) -> Result<Option<JitBackendType>, PyDiffsolError> {
        Ok(self.0.get_jit_backend()?.map(Into::into))
    }

    #[getter]
    fn get_ode_solver(&self) -> Result<OdeSolverType, PyDiffsolError> {
        Ok(self.0.get_ode_solver()?.into())
    }

    #[setter]
    fn set_ode_solver(&self, value: OdeSolverType) -> Result<(), PyDiffsolError> {
        self.0.set_ode_solver(value.into())?;
        Ok(())
    }

    #[getter]
    fn get_linear_solver(&self) -> Result<LinearSolverType, PyDiffsolError> {
        Ok(self.0.get_linear_solver()?.into())
    }

    #[setter]
    fn set_linear_solver(&self, value: LinearSolverType) -> Result<(), PyDiffsolError> {
        self.0.set_linear_solver(value.into())?;
        Ok(())
    }

    #[getter]
    fn get_rtol(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_rtol()?)
    }

    #[setter]
    fn set_rtol(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_rtol(value)?;
        Ok(())
    }

    #[getter]
    fn get_atol(&self) -> Result<f64, PyDiffsolError> {
        Ok(self.0.get_atol()?)
    }

    #[setter]
    fn set_atol(&self, value: f64) -> Result<(), PyDiffsolError> {
        self.0.set_atol(value)?;
        Ok(())
    }

    #[getter]
    fn get_code(&self) -> Result<String, PyDiffsolError> {
        Ok(self.0.get_code()?)
    }

    #[getter]
    fn get_nparams(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_nparams()?)
    }

    #[getter]
    fn get_nstates(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_nstates()?)
    }

    #[getter]
    fn get_nout(&self) -> Result<usize, PyDiffsolError> {
        Ok(self.0.get_nout()?)
    }

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

    fn y0<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyAny>, PyDiffsolError> {
        host_array_to_py(py, self.0.y0(pyarray1_to_host(params)?)?)
    }

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

    fn solve(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        final_time: f64,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(
            self.0.solve(pyarray1_to_host(params)?, final_time)?,
        ))
    }

    fn solve_hybrid(
        &self,
        params: PyReadonlyArray1<'_, f64>,
        final_time: f64,
    ) -> Result<SolutionWrapper, PyDiffsolError> {
        Ok(SolutionWrapper::new(
            self.0.solve_hybrid(pyarray1_to_host(params)?, final_time)?,
        ))
    }

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

    fn solve_sum_squares_adj<'py>(
        &self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
        data: PyReadonlyArray2<'py, f64>,
        t_eval: PyReadonlyArray1<'py, f64>,
    ) -> Result<(f64, Bound<'py, PyAny>), PyDiffsolError> {
        let scalar_type: ScalarType = self.0.get_scalar_type()?.into();
        let (value, sens) = match scalar_type {
            ScalarType::F32 => {
                let (_owned_data, data_host) = pyarray2_to_owned_f32_host(data)?;
                self.0.solve_sum_squares_adj(
                    pyarray1_to_host(params)?,
                    data_host,
                    pyarray1_to_host(t_eval)?,
                )?
            }
            ScalarType::F64 => self.0.solve_sum_squares_adj(
                pyarray1_to_host(params)?,
                pyarray2_to_host_f64(data)?,
                pyarray1_to_host(t_eval)?,
            )?,
        };
        Ok((value, host_array_to_py(py, sens)?))
    }
}
