use diffsol::{Matrix, NalgebraMat};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyType};
use pyo3::prelude::*;

#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatrixType {
    #[pyo3(name = "nalgebra_dense_f64")]
    NalgebraDenseF64,

    #[pyo3(name = "faer_dense_f64")]
    FaerDenseF64,

    #[pyo3(name = "faer_sparse_f64")]
    FaerSparseF64,
}

impl MatrixType {
    pub(crate) fn all_enums() -> Vec<MatrixType> {
        vec![
            MatrixType::NalgebraDenseF64,
            MatrixType::FaerDenseF64,
            MatrixType::FaerSparseF64,
        ]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            MatrixType::NalgebraDenseF64 => "nalgebra_dense_f64",
            MatrixType::FaerDenseF64 => "faer_dense_f64",
            MatrixType::FaerSparseF64 => "faer_sparse_f64",
        }
    }
    
    pub(crate) fn from_diffsol<M: Matrix>() -> Option<Self> {
        let id = std::any::TypeId::of::<M>();
        if id == std::any::TypeId::of::<NalgebraMat<f64>>() {
            Some(MatrixType::NalgebraDenseF64)
        } else if id == std::any::TypeId::of::<diffsol::FaerMat<f64>>() {
            Some(MatrixType::FaerDenseF64)
        } else if id == std::any::TypeId::of::<diffsol::FaerSparseMat<f64>>() {
            Some(MatrixType::FaerSparseF64)
        } else {
            None
        }
    }
}

#[pymethods]
impl MatrixType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "nalgebra_dense_f64" => Ok(MatrixType::NalgebraDenseF64),
            "faer_dense_f64" => Ok(MatrixType::FaerDenseF64),
            "faer_sparse_f64" => Ok(MatrixType::FaerSparseF64),
            _ => Err(PyValueError::new_err("Invalid MatrixType value")),
        }
    }

    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), MatrixType::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            MatrixType::NalgebraDenseF64 => 0,
            MatrixType::FaerDenseF64 => 1,
            MatrixType::FaerSparseF64 => 2,
        }
    }
}
