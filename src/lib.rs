use pyo3::prelude::*;

#[cfg(not(any(feature = "diffsol-cranelift", feature = "diffsol-llvm")))]
compile_error!("pydiffsol requires at least one JIT backend feature enabled");

mod adjoint_checkpoint;
mod error;
mod host_array;
mod jit;
mod linear_solver_type;
mod matrix_type;
mod ode;
mod ode_solver_type;
mod options_ic;
mod options_ode;
mod scalar_type;
mod solution;

/// Get version of this pydiffsol module
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Indicate whether KLU functions are available.
#[pyfunction]
fn is_klu_available() -> bool {
    diffsol_c::utils::is_klu_available()
}

/// Indicate whether sensitivity analysis is available.
#[pyfunction]
fn is_sens_available() -> bool {
    diffsol_c::utils::is_sens_available()
}

#[pyfunction]
fn default_enabled_jit_backend() -> Option<jit::JitBackendType> {
    diffsol_c::default_enabled_jit_backend().map(Into::into)
}

#[pyfunction]
fn diffsol_version() -> String {
    option_env!("PYDIFFSOL_DIFFSOL_VERSION")
        .unwrap_or("unknown")
        .to_string()
}

#[pyfunction]
fn diffsl_version() -> String {
    option_env!("PYDIFFSOL_DIFFSL_VERSION")
        .unwrap_or("unknown")
        .to_string()
}

#[pymodule]
fn pydiffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<matrix_type::MatrixType>()?;
    m.add_class::<scalar_type::ScalarType>()?;
    m.add_class::<linear_solver_type::LinearSolverType>()?;
    m.add_class::<ode_solver_type::OdeSolverType>()?;
    m.add_class::<jit::JitBackendType>()?;
    m.add_class::<adjoint_checkpoint::AdjointCheckpointWrapper>()?;
    m.add_class::<ode::OdeWrapper>()?;
    m.add_class::<options_ic::InitialConditionSolverOptions>()?;
    m.add_class::<options_ode::OdeSolverOptions>()?;
    m.add_class::<solution::SolutionWrapper>()?;

    for mt in matrix_type::MatrixType::all_enums() {
        m.add(mt.get_name(), mt)?;
    }
    for st in scalar_type::ScalarType::all_enums() {
        m.add(st.get_name(), st)?;
    }
    for ls in linear_solver_type::LinearSolverType::all_enums() {
        m.add(ls.get_name(), ls)?;
    }
    for solver in ode_solver_type::OdeSolverType::all_enums() {
        m.add(solver.get_name(), solver)?;
    }
    for backend in jit::JitBackendType::all_enums() {
        m.add(backend.get_name(), backend)?;
    }

    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_klu_available, m)?)?;
    m.add_function(wrap_pyfunction!(is_sens_available, m)?)?;
    m.add_function(wrap_pyfunction!(default_enabled_jit_backend, m)?)?;
    m.add_function(wrap_pyfunction!(diffsol_version, m)?)?;
    m.add_function(wrap_pyfunction!(diffsl_version, m)?)?;

    pyo3_log::init();

    Ok(())
}
