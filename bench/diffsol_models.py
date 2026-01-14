import pydiffsol as ds
import numpy as np
from diffsol_lotka_volterra import lokta_volterra_ode_str
from diffsol_robertson import robertson_ode_str


def setup(ngroups: int, tol: float, method: str, problem: str):
    if ngroups < 20:
        matrix_type = ds.nalgebra_dense
    else:
        matrix_type = ds.faer_sparse
    if method == "bdf":
        method = ds.bdf
    elif method == "esdirk34":
        method = ds.esdirk34
    elif method == "tr_bdf2":
        method = ds.tr_bdf2
    elif method == "tsit5":
        method = ds.tsit45
    else:
        raise ValueError(f"Unknown method: {method}")

    if problem == "robertson_ode":
        code, t_final = robertson_ode_str(ngroups=ngroups)
    elif problem == "lotka_volterra_ode":
        code, t_final = lokta_volterra_ode_str()
    else:
        raise ValueError(f"Unknown problem: {problem}")

    ode = ds.Ode(
        code,
        matrix_type=matrix_type,
        scalar_type=ds.f64,
        method=method,
    )
    ode.rtol = tol
    ode.atol = tol

    return ode, t_final


def bench(model):
    ode, t_final = model
    params = np.array([])
    ys = ode.solve_dense(params, np.array([t_final]))
    return ys[:, -1]
