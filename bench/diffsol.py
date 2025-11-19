import pydiffsol as ds
import numpy as np


def robertson_ode_str(ngroups: int):
    u_i = (
        f"(0:{ngroups}): x = 1,\n"
        f"({ngroups}:{2 * ngroups}): y = 0,\n"
        f"({2 * ngroups}:{3 * ngroups}): z = 0,\n"
    )
    code = (
        """
        k1 { 0.04 }
        k2 { 30000000 }
        k3 { 10000 }
        u_i {
        """
        + u_i
        + """
        }
        F_i {
            -k1 * x_i + k3 * y_i * z_i,
            k1 * x_i - k2 * y_i * y_i - k3 * y_i * z_i,
            k2 * y_i * y_i,
        }
        """
    )
    t_final = 1e10
    return code, t_final


def lokta_volterra_ode_str():
    code = """
    a { 2.0 / 3.0 }
    b { 4.0 / 3.0 }
    c { 1.0 }
    d { 1.0 }
    u_i {
        x = 1.0,
        y = 1.0,
    }
    F_i {
        a * x - b * x * y,
        -c * y + d * x * y,
    }
    """
    t_final = 10.0
    return code, t_final


def setup(ngroups: int, tol: float, method: str, problem: str):
    if ngroups < 20:
        matrix_type = ds.nalgebra_dense_f64
    else:
        matrix_type = ds.faer_sparse_f64
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
    elif problem == "lokta_volterra_ode":
        code, t_final = lokta_volterra_ode_str()
    else:
        raise ValueError(f"Unknown problem: {problem}")

    ode = ds.Ode(
        code,
        matrix_type=matrix_type,
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
