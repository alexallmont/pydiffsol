use pyo3::prelude::*;

mod config;
mod convert;
mod error;
mod jit;
mod matrix_type;
mod ode;
mod solver_method;
mod solver_type;

/// Get version of this pydiffsol module
#[pyfunction]
fn version() -> String {
    format!("{}", env!("CARGO_PKG_VERSION"))
}

#[pymodule]
fn pydiffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {

    // Register all Python API classes
    m.add_class::<matrix_type::MatrixType>()?;
    m.add_class::<solver_type::SolverType>()?;
    m.add_class::<solver_method::SolverMethod>()?;
    m.add_class::<config::ConfigWrapper>()?;
    m.add_class::<ode::OdeWrapper>()?;

    // Shorthand aliases, e.g. `ds.bdf` rather than `ds.SolverMethod.bdf`
    for mt in matrix_type::MatrixType::all_enums() {
        m.add(mt.get_name(), mt)?;
    }
    for st in solver_type::SolverType::all_enums() {
        m.add(st.get_name(), st)?;
    }
    for sm in solver_method::SolverMethod::all_enums() {
        m.add(sm.get_name(), sm)?;
    }

    // General utility methods
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
