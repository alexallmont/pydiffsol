import pytest
import pydiffsol as ds


def test_enums_all():
    assert ds.MatrixType.all() == [
        ds.MatrixType.nalgebra_dense,
        ds.MatrixType.faer_dense,
        ds.MatrixType.faer_sparse
    ]

    assert ds.ScalarType.all() == [
        ds.ScalarType.f32,
        ds.ScalarType.f64
    ]

    assert ds.SolverType.all() == [
        ds.SolverType.default,
        ds.SolverType.lu,
        ds.SolverType.klu
    ]

    assert ds.SolverMethod.all() == [
        ds.SolverMethod.bdf,
        ds.SolverMethod.esdirk34,
        ds.SolverMethod.tr_bdf2,
        ds.SolverMethod.tsit45
    ]


def test_enums_repr():
    assert repr(ds.nalgebra_dense) == "MatrixType.nalgebra_dense"
    assert repr(ds.faer_sparse) == "MatrixType.faer_sparse"
    assert repr(ds.faer_dense) == "MatrixType.faer_dense"

    assert repr(ds.f32) == "ScalarType.f32"
    assert repr(ds.f64) == "ScalarType.f64"

    assert repr(ds.default) == "SolverType.default"
    assert repr(ds.lu) == "SolverType.lu"
    assert repr(ds.klu) == "SolverType.klu"

    assert repr(ds.bdf) == "SolverMethod.bdf"
    assert repr(ds.esdirk34) == "SolverMethod.esdirk34"
    assert repr(ds.tr_bdf2) == "SolverMethod.tr_bdf2"
    assert repr(ds.tsit45) == "SolverMethod.tsit45"


def test_enums_str():
    assert str(ds.nalgebra_dense) == "nalgebra_dense"
    assert str(ds.faer_sparse) == "faer_sparse"
    assert str(ds.faer_dense) == "faer_dense"

    assert str(ds.f32) == "f32"
    assert str(ds.f64) == "f64"

    assert str(ds.default) == "default"
    assert str(ds.lu) == "lu"
    assert str(ds.klu) == "klu"

    assert str(ds.bdf) == "bdf"
    assert str(ds.esdirk34) == "esdirk34"
    assert str(ds.tr_bdf2) == "tr_bdf2"
    assert str(ds.tsit45) == "tsit45"


def test_enums_from_string():
    # Implicitly checks PartialEq implementation too
    assert ds.MatrixType.from_str("nalgebra_dense") == ds.nalgebra_dense
    assert ds.MatrixType.from_str("faer_sparse") == ds.faer_sparse
    assert ds.MatrixType.from_str("faer_dense") == ds.faer_dense

    assert ds.ScalarType.from_str("f32") == ds.f32
    assert ds.ScalarType.from_str("f64") == ds.f64

    assert ds.SolverType.from_str("default") == ds.default
    assert ds.SolverType.from_str("lu") == ds.lu
    assert ds.SolverType.from_str("klu") == ds.klu

    assert ds.SolverMethod.from_str("bdf") == ds.bdf
    assert ds.SolverMethod.from_str("esdirk34") == ds.esdirk34
    assert ds.SolverMethod.from_str("tr_bdf2") == ds.tr_bdf2
    assert ds.SolverMethod.from_str("tsit45") == ds.tsit45

    with pytest.raises(Exception):
        ds.MatrixType.from_str("foo")

    with pytest.raises(Exception):
        ds.ScalarType.from_str("qux")

    with pytest.raises(Exception):
        ds.SolverType.from_str("bar")

    with pytest.raises(Exception):
        ds.SolverMethod.from_str("etc")
