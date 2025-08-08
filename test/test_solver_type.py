import pydiffsol as ds

def test_config_defaults():
    c = ds.Config()
    assert c.method == ds.bdf
    assert c.linear_solver == ds.default
    assert c.rtol == 1e-6
    assert c.amended_solver_type(ds.nalgebra_dense_f64) == ds.lu
    assert c.amended_solver_type(ds.faer_dense_f64) == ds.lu
    assert c.amended_solver_type(ds.faer_sparse_f64) == ds.klu

    c = ds.Config(ds.tsit45)
    assert c.method == ds.tsit45
    assert c.linear_solver == ds.default
    assert c.rtol == 1e-6
    assert c.amended_solver_type(ds.nalgebra_dense_f64) == ds.default
    assert c.amended_solver_type(ds.faer_dense_f64) == ds.default
    assert c.amended_solver_type(ds.faer_sparse_f64) == ds.default

    c = ds.Config(ds.esdirk34, ds.lu)
    assert c.method == ds.esdirk34
    assert c.linear_solver == ds.lu
    assert c.rtol == 1e-6
    assert c.amended_solver_type(ds.nalgebra_dense_f64) == ds.lu
    assert c.amended_solver_type(ds.faer_dense_f64) == ds.lu
    assert c.amended_solver_type(ds.faer_sparse_f64) == ds.lu
