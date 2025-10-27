import pytest
import pydiffsol as ds


def test_enums_all():
    assert ds.MatrixType.all() == [
        ds.MatrixType.nalgebra_dense_f64,
        ds.MatrixType.faer_dense_f64,
        ds.MatrixType.faer_sparse_f64
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
    assert repr(ds.nalgebra_dense_f64) == "MatrixType.nalgebra_dense_f64"
    assert repr(ds.faer_sparse_f64) == "MatrixType.faer_sparse_f64"
    assert repr(ds.faer_dense_f64) == "MatrixType.faer_dense_f64"

    assert repr(ds.default) == "SolverType.default"
    assert repr(ds.lu) == "SolverType.lu"
    assert repr(ds.klu) == "SolverType.klu"

    assert repr(ds.bdf) == "SolverMethod.bdf"
    assert repr(ds.esdirk34) == "SolverMethod.esdirk34"
    assert repr(ds.tr_bdf2) == "SolverMethod.tr_bdf2"
    assert repr(ds.tsit45) == "SolverMethod.tsit45"


def test_enums_str():
    assert str(ds.nalgebra_dense_f64) == "nalgebra_dense_f64"
    assert str(ds.faer_sparse_f64) == "faer_sparse_f64"
    assert str(ds.faer_dense_f64) == "faer_dense_f64"

    assert str(ds.default) == "default"
    assert str(ds.lu) == "lu"
    assert str(ds.klu) == "klu"

    assert str(ds.bdf) == "bdf"
    assert str(ds.esdirk34) == "esdirk34"
    assert str(ds.tr_bdf2) == "tr_bdf2"
    assert str(ds.tsit45) == "tsit45"


def test_enums_from_string():
    # Implicitly checks PartialEq implementation too
    assert ds.MatrixType.from_str("nalgebra_dense_f64") == ds.nalgebra_dense_f64
    assert ds.MatrixType.from_str("faer_sparse_f64") == ds.faer_sparse_f64
    assert ds.MatrixType.from_str("faer_dense_f64") == ds.faer_dense_f64

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
        ds.SolverType.from_str("bar")

    with pytest.raises(Exception):
        ds.SolverMethod.from_str("etc")
