import pydiffsol as ds
import numpy as np


def robertson_ode_str(ngroups: int) -> str:
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
    return code


def setup(ngroups: int, tol: float, method: str):
    if ngroups < 20:
        matrix_type = ds.nalgebra_dense_f64
    else:
        matrix_type = ds.faer_sparse_f64
    if method == "bdf":
        method = ds.bdf
    elif method == "esdirk34":
        method = ds.esdirk34
    else:
        raise ValueError(f"Unknown method: {method}")

    ode = ds.Ode(
        robertson_ode_str(ngroups=ngroups),
        matrix_type=matrix_type,
        method=method,
    )
    ode.rtol = tol
    ode.atol = tol

    return ode


def bench(model, t_final: float):
    params = np.array([])
    ys = model.solve_dense(params, np.array([t_final]))
    return ys[:, -1]
