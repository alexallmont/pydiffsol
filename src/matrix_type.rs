// Matrix type Python enum

use diffsol::{Matrix, Scalar};
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

// Internal trait to determine runtime MatrixType from a compile-time diffsol matrix type
pub(crate) trait MatrixKind {
    const MATRIX_TYPE: MatrixType;
}

impl<T: Scalar> MatrixKind for diffsol::NalgebraMat<T> {
    const MATRIX_TYPE: MatrixType = MatrixType::NalgebraDense;
}

impl<T: Scalar> MatrixKind for diffsol::FaerMat<T> {
    const MATRIX_TYPE: MatrixType = MatrixType::FaerDense;
}

impl<T: Scalar> MatrixKind for diffsol::FaerSparseMat<T> {
    const MATRIX_TYPE: MatrixType = MatrixType::FaerSparse;
}

impl MatrixType {
    pub(crate) fn all_enums() -> Vec<MatrixType> {
        vec![
            MatrixType::NalgebraDense,
            MatrixType::FaerDense,
            MatrixType::FaerSparse,
        ]
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            MatrixType::NalgebraDense => "nalgebra_dense",
            MatrixType::FaerDense => "faer_dense",
            MatrixType::FaerSparse => "faer_sparse",
        }
    }

    // Determine runtime matrix type compiled diffsol matrix type
    pub(crate) fn from_diffsol<M: Matrix + MatrixKind>() -> Self {
        M::MATRIX_TYPE
    }
}

#[pymethods]
impl MatrixType {
    /// Create MatrixType from string name
    /// :param name: string representation of matrix type
    /// :return: valid MatrixType or exception if name is invalid
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        match name {
            "nalgebra_dense" => Ok(MatrixType::NalgebraDense),
            "faer_dense" => Ok(MatrixType::FaerDense),
            "faer_sparse" => Ok(MatrixType::FaerSparse),
            _ => Err(PyValueError::new_err("Invalid MatrixType value")),
        }
    }

    /// Get all available matrix types
    /// :return: list of MatrixType
    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), MatrixType::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            MatrixType::NalgebraDense => 0,
            MatrixType::FaerDense => 1,
            MatrixType::FaerSparse => 2,
        }
    }
}
