use pyo3::prelude::*;

mod config;
mod convert;
mod enums;
mod error;
mod jit;
mod ode;

/// A Python module implemented in Rust.
#[pymodule]
fn pydiffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {

    // Register all Python API classes
    m.add_class::<enums::MatrixType>()?;
    m.add_class::<enums::SolverType>()?;
    m.add_class::<enums::SolverMethod>()?;
    m.add_class::<config::ConfigWrapper>()?;
    m.add_class::<ode::OdeWrapper>()?;

    // Per-enum identifiers, e.g. `config.method = ds.bdf`
    m.add("nalgebra_dense_f64", enums::MatrixType::NalgebraDenseF64)?;
    m.add("faer_sparse_f64", enums::MatrixType::FaerSparseF64)?;
    m.add("lu", enums::SolverType::Lu)?;
    m.add("klu", enums::SolverType::Klu)?;
    m.add("bdf", enums::SolverMethod::Bdf)?;
    m.add("esdirk34", enums::SolverMethod::Esdirk34)?;

    Ok(())
}
