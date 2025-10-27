// Solver method Python enum. This is used to select the overarching solver
// stragegy like bdf or esdirk34 in diffsol.

use diffsol::error::DiffsolError;
use diffsol::{
    matrix::MatrixRef, DefaultDenseMatrix, DiffSl, LinearSolver, Matrix, OdeSolverMethod,
    OdeSolverProblem, VectorHost, VectorRef,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};

use crate::jit::JitModule;

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
        final_time: f64,
    ) -> Result<(<M::V as DefaultDenseMatrix>::M, Vec<f64>), DiffsolError>
    where
        M: Matrix<T = f64>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            SolverMethod::Bdf => problem.bdf::<LS>()?.solve(final_time),
            SolverMethod::Esdirk34 => problem.esdirk34::<LS>()?.solve(final_time),
            SolverMethod::TrBdf2 => problem.tr_bdf2::<LS>()?.solve(final_time),
            SolverMethod::Tsit45 => problem.tsit45()?.solve(final_time),
        }
    }

    pub(crate) fn solve_dense<M, LS>(
        &self,
        problem: &mut OdeSolverProblem<DiffSl<M, JitModule>>,
        t_eval: &[f64],
    ) -> Result<<M::V as DefaultDenseMatrix>::M, DiffsolError>
    where
        M: Matrix<T = f64>,
        M::V: VectorHost + DefaultDenseMatrix,
        LS: LinearSolver<M>,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        match self {
            SolverMethod::Bdf => problem.bdf::<LS>()?.solve_dense(t_eval),
            SolverMethod::Esdirk34 => problem.esdirk34::<LS>()?.solve_dense(t_eval),
            SolverMethod::TrBdf2 => problem.tr_bdf2::<LS>()?.solve_dense(t_eval),
            SolverMethod::Tsit45 => problem.tsit45()?.solve_dense(t_eval),
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
