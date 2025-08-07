use pyo3::prelude::*;

mod config;
mod convert;
mod enums;
mod error;
mod jit;
mod ode;

/// Get version of this pydiffsol module
#[pyfunction]
fn version() -> String {
    format!("{}", env!("CARGO_PKG_VERSION"))
}

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

    // General utility methods
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
