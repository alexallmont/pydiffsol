use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyType},
};

#[pyclass(eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JitBackendType {
    #[cfg(feature = "diffsol-cranelift")]
    #[pyo3(name = "cranelift")]
    Cranelift,

    #[cfg(feature = "diffsol-llvm")]
    #[pyo3(name = "llvm")]
    Llvm,
}

impl JitBackendType {
    pub(crate) fn all_enums() -> Vec<Self> {
        let mut values = Vec::new();
        #[cfg(feature = "diffsol-cranelift")]
        values.push(Self::Cranelift);
        #[cfg(feature = "diffsol-llvm")]
        values.push(Self::Llvm);
        values
    }

    pub(crate) fn get_name(&self) -> &str {
        match self {
            #[cfg(feature = "diffsol-cranelift")]
            Self::Cranelift => "cranelift",
            #[cfg(feature = "diffsol-llvm")]
            Self::Llvm => "llvm",
        }
    }
}

impl From<JitBackendType> for diffsol_c::JitBackendType {
    fn from(value: JitBackendType) -> Self {
        match value {
            #[cfg(feature = "diffsol-cranelift")]
            JitBackendType::Cranelift => diffsol_c::JitBackendType::Cranelift,
            #[cfg(feature = "diffsol-llvm")]
            JitBackendType::Llvm => diffsol_c::JitBackendType::Llvm,
        }
    }
}

impl From<diffsol_c::JitBackendType> for JitBackendType {
    fn from(value: diffsol_c::JitBackendType) -> Self {
        match value {
            #[cfg(feature = "diffsol-cranelift")]
            diffsol_c::JitBackendType::Cranelift => JitBackendType::Cranelift,
            #[cfg(feature = "diffsol-llvm")]
            diffsol_c::JitBackendType::Llvm => JitBackendType::Llvm,
        }
    }
}

#[pymethods]
impl JitBackendType {
    #[classmethod]
    fn from_str(_cls: &Bound<'_, PyType>, value: &str) -> PyResult<Self> {
        match value {
            #[cfg(feature = "diffsol-cranelift")]
            "cranelift" => Ok(Self::Cranelift),
            #[cfg(feature = "diffsol-llvm")]
            "llvm" => Ok(Self::Llvm),
            _ => Err(PyValueError::new_err("Invalid JitBackendType value")),
        }
    }

    #[classmethod]
    fn all<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(cls.py(), Self::all_enums())
    }

    fn __str__(&self) -> String {
        self.get_name().to_string()
    }

    fn __hash__(&self) -> u64 {
        match self {
            #[cfg(feature = "diffsol-cranelift")]
            Self::Cranelift => 0,
            #[cfg(feature = "diffsol-llvm")]
            Self::Llvm => 1,
        }
    }
}
