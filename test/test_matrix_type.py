import numpy as np
import pydiffsol as ds
import pytest

# Sets of methods used to build up valid method combinations
NO_LS_METHODS = {ds.tsit45}
LS_METHODS = {ds.bdf, ds.esdirk34, ds.tr_bdf2}
ALL_METHODS = NO_LS_METHODS | LS_METHODS

# Dict specifying which solvers are supported/valid depending on combination of
# linear solver, matrix and method type. These are expanded into triplets below
# to drive test parameters.
VALID_METHODS = {
    ds.nalgebra_dense: {ds.default: ALL_METHODS, ds.lu: ALL_METHODS},
    ds.faer_dense: {ds.default: ALL_METHODS, ds.lu: ALL_METHODS},
    ds.faer_sparse: {ds.default: ALL_METHODS, ds.lu: ALL_METHODS},
}

# See test_klu_available.py to confirm which platforms support KLU
if ds.is_klu_available():
    VALID_METHODS[ds.faer_sparse] = {ds.default: ALL_METHODS, ds.klu: ALL_METHODS, ds.lu: ALL_METHODS}

# Simple logistic diffsl code used throughout tests
DIFFSL_LOGISTIC = \
"""
    r { 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
"""

# Generate (matrix_type, method, solver) tuple from solver type to methods dict.
def _gen_config_triplets(matrix_type, solver_to_methods_map):
    configs = []
    for solver, methods in solver_to_methods_map.items():
        for method in methods:
            configs.append((matrix_type, method, solver))
    return configs

# Generate valid (matrix_type, method, solver) tuple list for positive tests.
def _valid_config_triplets():
    config_triplets = []
    for matrix_type, solver_methods in VALID_METHODS.items():
        config_triplets += _gen_config_triplets(matrix_type, solver_methods)
    return config_triplets

# Config valid check for filtering out positives below
def _is_config_valid(matrix_type, linear_solver, method) -> bool:
    if matrix_type in VALID_METHODS:
        if linear_solver in VALID_METHODS[matrix_type]:
            if method in VALID_METHODS[matrix_type][linear_solver]:
                return True
    return False

# Generate invalid (matrix_type, method, solver) tuple list for negative tests.
def _invalid_config_triplets():
    configs = []
    for mt in VALID_METHODS.keys():
        for ls in [ds.default, ds.lu, ds.klu]:
            for method in [ds.tsit45, ds.bdf, ds.esdirk34, ds.tr_bdf2]:
                if not _is_config_valid(mt, ls, method):
                    configs += [(mt, method, ls)]
    return configs


# Positive check for solve and solve_dense supported on this platform
@pytest.mark.parametrize("matrix_type,method,linear_solver", _valid_config_triplets())
def test_valid_config_solve(matrix_type, linear_solver, method):
    # TODO faer_sparse klu bdf skipped until underlying diffsol bug resolved
    if matrix_type == ds.faer_sparse and linear_solver == ds.klu and method == ds.bdf:
        print(
            "Skipping test_valid_config_solve for",
            matrix_type, linear_solver, method
        )
        return

    ode = ds.Ode(DIFFSL_LOGISTIC, matrix_type=matrix_type, method=method, linear_solver=linear_solver)

    # All valid solver configs should generate approximately the same value
    ys, _ = ode.solve(np.array([]), 0.4)
    last_y = ys[0][-1]
    assert np.isclose(last_y, 0.142189, rtol=1e-4)

    # Also check solve_dense works over set times
    t_eval = np.array([0.0, 0.1, 0.5])
    ys = ode.solve_dense(np.array([]), t_eval)
    assert np.allclose(ys, [[0.1, 0.109366, 0.154828]], rtol=1e-4)


# Negative check for solve and solve_dense not supported on this platform
@pytest.mark.parametrize("matrix_type,method,linear_solver", _invalid_config_triplets())
def test_invalid_config_solve(matrix_type, linear_solver, method):
    with pytest.raises(Exception):
        ode = ds.Ode(DIFFSL_LOGISTIC, matrix_type=matrix_type, method=method, linear_solver=linear_solver)

    ode = ds.Ode(DIFFSL_LOGISTIC)
    ode.method = method
    with pytest.raises(Exception):
        ode.linear_solver = linear_solver
