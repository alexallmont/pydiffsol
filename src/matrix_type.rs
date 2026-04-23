// Wrap diffsol-c matrix type with Python enum. This is the internal matrix
// mechanism, either nalgebra_dense, faer_dense, or faer_sparse.

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyType},
};

/// Enumerates the possible matrix types for diffsol
///
/// :attr nalgebra_dense: dense matrix using nalgebra crate (https://nalgebra.rs/)
/// :attr faer_dense: dense matrix using faer crate (https://faer.veganb.tw/)
/// :attr faer_sparse: sparse matrix using faer crate (https://faer.veganb.tw/)
#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatrixType {
    #[pyo3(name = "nalgebra_dense")]
    NalgebraDense,

    #[pyo3(name = "faer_dense")]
    FaerDense,

    #[pyo3(name = "faer_sparse")]
    FaerSparse,
}

impl MatrixType {
    pub(crate) fn all_enums() -> Vec<Self> {
        vec![Self::NalgebraDense, Self::FaerDense, Self::FaerSparse]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            Self::NalgebraDense => "nalgebra_dense",
            Self::FaerDense => "faer_dense",
            Self::FaerSparse => "faer_sparse",
        }
    }
}

impl From<MatrixType> for diffsol_c::MatrixType {
    fn from(value: MatrixType) -> Self {
        match value {
            MatrixType::NalgebraDense => diffsol_c::MatrixType::NalgebraDense,
            MatrixType::FaerDense => diffsol_c::MatrixType::FaerDense,
            MatrixType::FaerSparse => diffsol_c::MatrixType::FaerSparse,
        }
    }
}

impl From<diffsol_c::MatrixType> for MatrixType {
    fn from(value: diffsol_c::MatrixType) -> Self {
        match value {
            diffsol_c::MatrixType::NalgebraDense => MatrixType::NalgebraDense,
            diffsol_c::MatrixType::FaerDense => MatrixType::FaerDense,
            diffsol_c::MatrixType::FaerSparse => MatrixType::FaerSparse,
        }
    }
}

#[pymethods]
impl MatrixType {
    /// Create MatrixType from string name
    /// :param name: string representation of matrix type
    /// :return: valid MatrixType or exception if name is invalid
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            "nalgebra_dense" => Ok(Self::NalgebraDense),
            "faer_dense" => Ok(Self::FaerDense),
            "faer_sparse" => Ok(Self::FaerSparse),
            _ => Err(PyValueError::new_err("Invalid MatrixType value")),
        }
    }

    /// Get all available matrix types
    /// :return: list of MatrixType
    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), Self::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            Self::NalgebraDense => 0,
            Self::FaerDense => 1,
            Self::FaerSparse => 2,
        }
    }
}
