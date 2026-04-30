// Wrap diffsol-c adjoint checkpoint state with Python handle type.
// This does not have any accessors, it's purely for passing underlying
// Rust type between adjoint solve forward and backward calls.

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
