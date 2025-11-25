import casadi
import numpy as np


def setup_robertson_ode(ngroups: int):
    x = casadi.MX.sym("x", ngroups)
    y = casadi.MX.sym("y", ngroups)
    z = casadi.MX.sym("z", ngroups)
    k1 = 0.04
    k2 = 30000000
    k3 = 10000

    # Expression for ODE right-hand side
    f0 = -k1 * x + k3 * y * z
    f1 = k1 * x - k2 * y**2 - k3 * y * z
    f2 = k2 * y**2

    ode = {}  # ODE declaration
    ode["x"] = casadi.vertcat(x, y, z)  # states
    ode["ode"] = casadi.vertcat(f0, f1, f2)  # right-hand side
    x0 = np.zeros(3 * ngroups)
    x0[:ngroups] = 1.0
    return (ode, 1e10, x0)