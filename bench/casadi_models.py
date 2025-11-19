import bench.casadi_models as casadi_models
import numpy as np


def setup_robertson_ode(ngroups: int):
    x = casadi_models.MX.sym("x", ngroups)
    y = casadi_models.MX.sym("y", ngroups)
    z = casadi_models.MX.sym("z", ngroups)
    k1 = 0.04
    k2 = 30000000
    k3 = 10000

    # Expression for ODE right-hand side
    f0 = -k1 * x + k3 * y * z
    f1 = k1 * x - k2 * y**2 - k3 * y * z
    f2 = k2 * y**2

    ode = {}  # ODE declaration
    ode["x"] = casadi_models.vertcat(x, y, z)  # states
    ode["ode"] = casadi_models.vertcat(f0, f1, f2)  # right-hand side
    x0 = np.zeros(3 * ngroups)
    x0[:ngroups] = 1.0
    return (ode, 1e10, x0)


def setup_lokta_volterra_ode():
    x = casadi_models.MX.sym("x")
    y = casadi_models.MX.sym("y")
    a = 2.0 / 3.0
    b = 4.0 / 3.0
    c = 1.0
    d = 1.0

    # Expression for ODE right-hand side
    f0 = a * x - b * x * y
    f1 = -c * y + d * x * y

    ode = {}  # ODE declaration
    ode["x"] = casadi_models.vertcat(x, y)  # states
    ode["ode"] = casadi_models.vertcat(f0, f1)  # right-hand side
    x0 = np.ones(2)
    return (ode, 10.0, x0)


def setup(ngroups: int, tol: float, problem: str):
    if problem == "robertson_ode":
        (ode, t_final, x0) = setup_robertson_ode(ngroups)
    elif problem == "lokta_volterra_ode":
        (ode, t_final, x0) = setup_lokta_volterra_ode()
    else:
        raise ValueError(f"Unknown problem: {problem}")
    F = casadi_models.integrator(
        "F", "cvodes", ode, 0.0, t_final, {"abstol": tol, "reltol": tol}
    )
    return F, x0


def bench(model) -> np.ndarray:
    F, x0 = model
    return F(x0=x0)["xf"][:, -1]