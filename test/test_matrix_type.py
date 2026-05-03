import numpy as np
import pydiffsol as ds
import pytest


DIFFSL_LOGISTIC = """
    r { 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
"""

ALL_ODE_SOLVERS = [ds.bdf, ds.esdirk34, ds.tr_bdf2, ds.tsit45]

VALID_LINEAR_SOLVERS = {
    ds.nalgebra_dense: [ds.default, ds.lu],
    ds.faer_dense: [ds.default, ds.lu],
    ds.faer_sparse: [ds.default, ds.lu],
}

if ds.is_klu_available():
    VALID_LINEAR_SOLVERS[ds.faer_sparse].append(ds.klu)


def valid_triplets():
    values = []
    for matrix_type, linear_solvers in VALID_LINEAR_SOLVERS.items():
        for linear_solver in linear_solvers:
            for ode_solver in ALL_ODE_SOLVERS:
                values.append((matrix_type, ode_solver, linear_solver))
    return values


def invalid_pairs():
    values = []
    for matrix_type in VALID_LINEAR_SOLVERS:
        for linear_solver in [ds.default, ds.lu, ds.klu]:
            if linear_solver not in VALID_LINEAR_SOLVERS[matrix_type]:
                values.append((matrix_type, linear_solver))
    return values


@pytest.mark.parametrize("matrix_type,ode_solver,linear_solver", valid_triplets())
def test_valid_config_solve(jit_backend, matrix_type, ode_solver, linear_solver):
    if matrix_type == ds.faer_sparse and linear_solver == ds.klu and ode_solver == ds.bdf:
        pytest.skip("Known upstream instability for FaerSparse + KLU + BDF")

    ode = ds.Ode(
        DIFFSL_LOGISTIC,
        jit_backend=jit_backend,
        matrix_type=matrix_type,
        scalar_type=ds.f64,
        ode_solver=ode_solver,
        linear_solver=linear_solver,
    )

    ys = ode.solve(np.array([]), 0.4).ys
    assert np.isclose(ys[0, -1], 0.142189, rtol=1e-4)

    t_eval = np.array([0.0, 0.1, 0.5])
    ys = ode.solve_dense(np.array([]), t_eval).ys
    assert np.allclose(ys, [[0.1, 0.109366, 0.154828]], rtol=1e-4)


@pytest.mark.parametrize("matrix_type,linear_solver", invalid_pairs())
def test_invalid_linear_solver_config(jit_backend, matrix_type, linear_solver):
    with pytest.raises(RuntimeError):
        ds.Ode(
            DIFFSL_LOGISTIC,
            jit_backend=jit_backend,
            matrix_type=matrix_type,
            scalar_type=ds.f64,
            ode_solver=ds.bdf,
            linear_solver=linear_solver,
        )

    ode = ds.Ode(
        DIFFSL_LOGISTIC,
        jit_backend=jit_backend,
        matrix_type=matrix_type,
        scalar_type=ds.f64,
        ode_solver=ds.bdf,
        linear_solver=ds.default,
    )
    with pytest.raises(RuntimeError):
        ode.linear_solver = linear_solver
