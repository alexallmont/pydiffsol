import pytest
import pydiffsol as ds


def expected_jit_backends():
    values = []
    if hasattr(ds, "cranelift"):
        values.append(ds.JitBackendType.cranelift)
    if hasattr(ds, "llvm"):
        values.append(ds.JitBackendType.llvm)
    return values


def expected_jit_backend_names():
    names = []
    if hasattr(ds, "cranelift"):
        names.append("cranelift")
    if hasattr(ds, "llvm"):
        names.append("llvm")
    return names


def test_enums_all():
    assert ds.MatrixType.all() == [
        ds.MatrixType.nalgebra_dense,
        ds.MatrixType.faer_dense,
        ds.MatrixType.faer_sparse,
    ]
    assert ds.ScalarType.all() == [ds.ScalarType.f32, ds.ScalarType.f64]
    assert ds.LinearSolverType.all() == [
        ds.LinearSolverType.default,
        ds.LinearSolverType.lu,
        ds.LinearSolverType.klu,
    ]
    assert ds.OdeSolverType.all() == [
        ds.OdeSolverType.bdf,
        ds.OdeSolverType.esdirk34,
        ds.OdeSolverType.tr_bdf2,
        ds.OdeSolverType.tsit45,
    ]
    assert ds.JitBackendType.all() == expected_jit_backends()


def test_enums_repr():
    assert repr(ds.nalgebra_dense) == "MatrixType.nalgebra_dense"
    assert repr(ds.faer_dense) == "MatrixType.faer_dense"
    assert repr(ds.faer_sparse) == "MatrixType.faer_sparse"

    assert repr(ds.f32) == "ScalarType.f32"
    assert repr(ds.f64) == "ScalarType.f64"

    assert repr(ds.default) == "LinearSolverType.default"
    assert repr(ds.lu) == "LinearSolverType.lu"
    assert repr(ds.klu) == "LinearSolverType.klu"

    assert repr(ds.bdf) == "OdeSolverType.bdf"
    assert repr(ds.esdirk34) == "OdeSolverType.esdirk34"
    assert repr(ds.tr_bdf2) == "OdeSolverType.tr_bdf2"
    assert repr(ds.tsit45) == "OdeSolverType.tsit45"

    for backend in expected_jit_backends():
        assert repr(backend).startswith("JitBackendType.")


def test_enums_str():
    assert str(ds.nalgebra_dense) == "nalgebra_dense"
    assert str(ds.faer_dense) == "faer_dense"
    assert str(ds.faer_sparse) == "faer_sparse"

    assert str(ds.f32) == "f32"
    assert str(ds.f64) == "f64"

    assert str(ds.default) == "default"
    assert str(ds.lu) == "lu"
    assert str(ds.klu) == "klu"

    assert str(ds.bdf) == "bdf"
    assert str(ds.esdirk34) == "esdirk34"
    assert str(ds.tr_bdf2) == "tr_bdf2"
    assert str(ds.tsit45) == "tsit45"

    assert [str(backend) for backend in expected_jit_backends()] == expected_jit_backend_names()


def test_enums_from_string():
    assert ds.MatrixType.from_str("nalgebra_dense") == ds.nalgebra_dense
    assert ds.MatrixType.from_str("faer_dense") == ds.faer_dense
    assert ds.MatrixType.from_str("faer_sparse") == ds.faer_sparse

    assert ds.ScalarType.from_str("f32") == ds.f32
    assert ds.ScalarType.from_str("f64") == ds.f64

    assert ds.LinearSolverType.from_str("default") == ds.default
    assert ds.LinearSolverType.from_str("lu") == ds.lu
    assert ds.LinearSolverType.from_str("klu") == ds.klu

    assert ds.OdeSolverType.from_str("bdf") == ds.bdf
    assert ds.OdeSolverType.from_str("esdirk34") == ds.esdirk34
    assert ds.OdeSolverType.from_str("tr_bdf2") == ds.tr_bdf2
    assert ds.OdeSolverType.from_str("tsit45") == ds.tsit45

    for backend in expected_jit_backends():
        assert ds.JitBackendType.from_str(str(backend)) == backend

    with pytest.raises(Exception):
        ds.MatrixType.from_str("foo")
    with pytest.raises(Exception):
        ds.ScalarType.from_str("bar")
    with pytest.raises(Exception):
        ds.LinearSolverType.from_str("baz")
    with pytest.raises(Exception):
        ds.OdeSolverType.from_str("qux")
    with pytest.raises(Exception):
        ds.JitBackendType.from_str("bogus")


def test_default_enabled_jit_backend_is_valid():
    backend = ds.default_enabled_jit_backend()
    if backend is None:
        assert len(expected_jit_backends()) > 1
    else:
        assert backend in expected_jit_backends()
