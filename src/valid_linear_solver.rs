// Validation used by py_solve to determine that solver type and matrix type
// combinations are valid.

use diffsol::error::DiffsolError;

use crate::{error::PyDiffsolError, matrix_type::MatrixType, solver_type::SolverType};

pub(crate) fn validate_linear_solver<M: diffsol::Matrix + LuValidator<M> + KluValidator<M>>(
    linear_solver: SolverType,
) -> Result<(), PyDiffsolError> {
    match linear_solver {
        SolverType::Default => Ok(()),
        SolverType::Lu => {
            if !<M as LuValidator<M>>::valid() {
                return Err(DiffsolError::Other(format!(
                    "Lu solver not supported for {}",
                    MatrixType::from_diffsol::<M>().unwrap().get_name()
                ))
                .into());
            }
            Ok(())
        }
        SolverType::Klu => {
            if !<M as KluValidator<M>>::valid() {
                return Err(DiffsolError::Other(format!(
                    "Klu solver not supported for {}",
                    MatrixType::from_diffsol::<M>().unwrap().get_name()
                ))
                .into());
            }
            Ok(())
        }
    }
}

pub(crate) trait KluValidator<M: diffsol::Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

#[cfg(feature = "suitesparse")]
impl KluValidator<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::KLU<diffsol::FaerSparseMat<f64>>;
    fn valid() -> bool {
        true
    }
}

#[cfg(not(feature = "suitesparse"))]
impl KluValidator<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::FaerSparseLU<f64>;
    fn valid() -> bool {
        false
    }
}

impl KluValidator<diffsol::NalgebraMat<f64>> for diffsol::NalgebraMat<f64> {
    type LS = diffsol::NalgebraLU<f64>;
    fn valid() -> bool {
        false
    }
}

impl KluValidator<diffsol::FaerMat<f64>> for diffsol::FaerMat<f64> {
    type LS = diffsol::FaerLU<f64>;
    fn valid() -> bool {
        false
    }
}

pub(crate) trait LuValidator<M: diffsol::Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

impl LuValidator<diffsol::NalgebraMat<f64>> for diffsol::NalgebraMat<f64> {
    type LS = diffsol::NalgebraLU<f64>;
    fn valid() -> bool {
        true
    }
}

impl LuValidator<diffsol::FaerMat<f64>> for diffsol::FaerMat<f64> {
    type LS = diffsol::FaerLU<f64>;
    fn valid() -> bool {
        true
    }
}

impl LuValidator<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::FaerSparseLU<f64>;
    fn valid() -> bool {
        true
    }
}
