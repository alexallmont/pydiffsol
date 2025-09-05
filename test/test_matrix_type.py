import numpy as np
import pydiffsol as ds
import pytest

# Instructions to determine which solvers are supported/valid depending on
# linear solver, matrix and method type. These are expanded into triples below
# to drive test parameters.
NO_LS_METHODS = {ds.tsit45}
LS_METHODS = {ds.bdf, ds.esdirk34, ds.tr_bdf2}
ALL_METHODS = NO_LS_METHODS | LS_METHODS
VALID_METHODS = {
    ds.nalgebra_dense_f64: {ds.default: ALL_METHODS, ds.lu: LS_METHODS},
    ds.faer_dense_f64: {ds.default: ALL_METHODS, ds.lu: LS_METHODS},
    ds.faer_sparse_f64: {ds.default: ALL_METHODS, ds.klu: LS_METHODS}
}

DIFFSL_LOGISTIC = \
"""
    r { 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
"""


def _gen_config_triplets(matrix_type, solver_to_methods_map):
    configs = []
    for solver, methods in solver_to_methods_map.items():
        for method in methods:
            configs.append((matrix_type, method, solver))
    return configs


def _valid_config_triplets():
    config_triplets = []
    for matrix_type, solver_methods in VALID_METHODS.items():
        config_triplets += _gen_config_triplets(matrix_type, solver_methods)
    return config_triplets


def _is_config_valid(matrix_type, linear_solver, method):
    if matrix_type in VALID_METHODS:
        if linear_solver in VALID_METHODS[matrix_type]:
            if method in VALID_METHODS[matrix_type][linear_solver]:
                return True


def _invalid_config_triplets():
    configs = []
    for mt in VALID_METHODS.keys():
        for ls in [ds.default, ds.lu, ds.klu]:
            for method in [ds.tsit45, ds.bdf, ds.esdirk34, ds.tr_bdf2]:
                if not _is_config_valid(mt, ls, method):
                    configs += [(mt, method, ls)]
    return configs


@pytest.mark.parametrize("matrix_type,method,linear_solver", _valid_config_triplets())
def test_valid_config_solve(matrix_type, linear_solver, method):
    # Skip faer_sparse_f64 klu bdf until we've solved underlying diffsol issue
    if matrix_type == ds.faer_sparse_f64 and method == ds.bdf:
        print("Skipping test_valid_config_solve for", matrix_type, linear_solver, method)
        return

    ode = ds.Ode(DIFFSL_LOGISTIC, matrix_type)
    config = ds.Config()
    config.method = method
    config.linear_solver = linear_solver

    # All valid solver configs should generate approximately the same value
    ys, _ = ode.solve(np.array([]), 0.4, config)
    last_y = ys[0][-1]
    assert np.isclose(last_y, 0.142189, rtol=1e-4)


@pytest.mark.parametrize("matrix_type,method,linear_solver", _invalid_config_triplets())
def test_invalid_config_solve(matrix_type, linear_solver, method):
    ode = ds.Ode(DIFFSL_LOGISTIC, matrix_type)
    config = ds.Config()
    config.method = method
    config.linear_solver = linear_solver

    # Any invalid solver configs must throw an exception
    with pytest.raises(Exception):
        _, _ = ode.solve(np.array([]), 0.4, config)
