use pyo3::prelude::*;

#[pyclass(module = "pydiffsol")]
#[pyo3(name = "AdjointCheckpoint")]
#[derive(Clone)]
pub struct AdjointCheckpointWrapper(diffsol_c::AdjointCheckpointWrapper);

impl AdjointCheckpointWrapper {
    pub(crate) fn new(inner: diffsol_c::AdjointCheckpointWrapper) -> Self {
        Self(inner)
    }

    pub(crate) fn inner(&self) -> &diffsol_c::AdjointCheckpointWrapper {
        &self.0
    }
}
