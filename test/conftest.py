import pydiffsol as ds
import pytest


def select_jit_backend():
    backend = ds.default_enabled_jit_backend()
    if backend is not None:
        return backend
    if hasattr(ds, "cranelift"):
        return ds.cranelift
    if hasattr(ds, "llvm"):
        return ds.llvm
    raise RuntimeError("No JIT backend is available")


@pytest.fixture(scope="session")
def jit_backend():
    return select_jit_backend()
