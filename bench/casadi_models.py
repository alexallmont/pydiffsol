import casadi
import numpy as np
from casadi_lotka_volterra import setup_lokta_volterra_ode
from casadi_robertson import setup_robertson_ode

def setup(ngroups: int, tol: float, problem: str):
    if problem == "robertson_ode":
        (ode, t_final, x0) = setup_robertson_ode(ngroups)
    elif problem == "lotka_volterra_ode":
        (ode, t_final, x0) = setup_lokta_volterra_ode()
    else:
        raise ValueError(f"Unknown problem: {problem}")
    F = casadi.integrator(
        "F", "cvodes", ode, 0.0, t_final, {"abstol": tol, "reltol": tol}
    )
    return F, x0


def bench(model) -> np.ndarray:
    F, x0 = model
    return F(x0=x0)["xf"][:, -1]