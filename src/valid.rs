use diffsol::error::DiffsolError;

use crate::{error::PyDiffsolError, matrix_type::MatrixType, solver_type::SolverType};

pub(crate) fn check<M: diffsol::Matrix + Lu<M> + Klu<M>>(
    linear_solver: SolverType,
) -> Result<(), PyDiffsolError> {
    match linear_solver {
        SolverType::Default => Ok(()),
        SolverType::Lu => {
            if !<M as Lu<M>>::valid() {
                return Err(DiffsolError::Other(format!(
                    "Lu solver not supported for {}",
                    MatrixType::from_diffsol::<M>().unwrap().get_name()
                ))
                .into());
            }
            Ok(())
        }
        SolverType::Klu => {
            if !<M as Klu<M>>::valid() {
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

pub(crate) trait Klu<M: diffsol::Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

#[cfg(feature = "suitesparse")]
impl Klu<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::KLU<diffsol::FaerSparseMat<f64>>;
    fn valid() -> bool {
        true
    }
}

#[cfg(not(feature = "suitesparse"))]
impl Klu<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::FaerSparseLU<f64>;
    fn valid() -> bool {
        false
    }
}

impl Klu<diffsol::NalgebraMat<f64>> for diffsol::NalgebraMat<f64> {
    type LS = diffsol::NalgebraLU<f64>;
    fn valid() -> bool {
        false
    }
}

impl Klu<diffsol::FaerMat<f64>> for diffsol::FaerMat<f64> {
    type LS = diffsol::FaerLU<f64>;
    fn valid() -> bool {
        false
    }
}

pub(crate) trait Lu<M: diffsol::Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

impl Lu<diffsol::NalgebraMat<f64>> for diffsol::NalgebraMat<f64> {
    type LS = diffsol::NalgebraLU<f64>;
    fn valid() -> bool {
        true
    }
}

impl Lu<diffsol::FaerMat<f64>> for diffsol::FaerMat<f64> {
    type LS = diffsol::FaerLU<f64>;
    fn valid() -> bool {
        true
    }
}

impl Lu<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::FaerSparseLU<f64>;
    fn valid() -> bool {
        true
    }
}
