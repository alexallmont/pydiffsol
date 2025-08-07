import pytest
import pydiffsol as ds

def test_enums_exist():
    # Implicitly checks __repr__ (using PyO3 default)
    assert str(ds.nalgebra_dense_f64) == "MatrixType.nalgebra_dense_f64"
    assert str(ds.faer_sparse_f64) == "MatrixType.faer_sparse_f64"
    assert str(ds.lu) == "SolverType.lu"
    assert str(ds.klu) == "SolverType.klu"
    assert str(ds.bdf) == "SolverMethod.bdf"
    assert str(ds.esdirk34) == "SolverMethod.esdirk34"

def test_enums_from_string():
    # Implicitly checks PartialEq implementation too
    assert ds.MatrixType.from_str("nalgebra_dense_f64") == ds.nalgebra_dense_f64
    assert ds.MatrixType.from_str("faer_sparse_f64") == ds.faer_sparse_f64
    assert ds.SolverType.from_str("lu") == ds.lu
    assert ds.SolverType.from_str("klu") == ds.klu
    assert ds.SolverMethod.from_str("bdf") == ds.bdf
    assert ds.SolverMethod.from_str("esdirk34") == ds.esdirk34

    with pytest.raises(Exception):
        ds.MatrixType.from_str("foo")

    with pytest.raises(Exception):
        ds.SolverType.from_str("bar")

    with pytest.raises(Exception):
        ds.SolverMethod.from_str("etc")
