use pyo3::prelude::*;

mod error;
mod jit;
mod matrix_type;
mod ode;
mod options_ic;
mod options_ode;
mod py_convert;
mod py_solve;
mod py_solve_macros;
mod py_types;
mod scalar_type;
mod solver_method;
mod solver_type;
mod valid_linear_solver;

/// Get version of this pydiffsol module
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Inidicate whether Klu functions are available.
/// This depends on whether the library was built with suitesparse support.
#[pyfunction]
fn is_klu_available() -> bool {
    cfg!(feature = "suitesparse")
}

/// Inidicate whether sensitivity analysis is available.
/// Sensitivity analysis is currently limited to Linux and macos, and not supported for Windows.
#[pyfunction]
fn is_sens_available() -> bool {
    cfg!(not(target_os = "windows"))
}

#[pymodule]
fn pydiffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all Python API classes
    m.add_class::<matrix_type::MatrixType>()?;
    m.add_class::<scalar_type::ScalarType>()?;
    m.add_class::<solver_type::SolverType>()?;
    m.add_class::<solver_method::SolverMethod>()?;
    m.add_class::<ode::OdeWrapper>()?;
    m.add_class::<options_ic::InitialConditionSolverOptions>()?;
    m.add_class::<options_ode::OdeSolverOptions>()?;

    // Shorthand aliases, e.g. `ds.bdf` rather than `ds.SolverMethod.bdf`
    for mt in matrix_type::MatrixType::all_enums() {
        m.add(mt.get_name(), mt)?;
    }
    for st in scalar_type::ScalarType::all_enums() {
        m.add(st.get_name(), st)?;
    }
    for st in solver_type::SolverType::all_enums() {
        m.add(st.get_name(), st)?;
    }
    for sm in solver_method::SolverMethod::all_enums() {
        m.add(sm.get_name(), sm)?;
    }

    // General utility methods
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_klu_available, m)?)?;

    pyo3_log::init();

    Ok(())
}
