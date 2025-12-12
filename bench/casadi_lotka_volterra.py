import casadi
import numpy as np

def setup_lokta_volterra_ode():
    x = casadi.MX.sym("x")
    y = casadi.MX.sym("y")
    a = 2.0 / 3.0
    b = 4.0 / 3.0
    c = 1.0
    d = 1.0

    # Expression for ODE right-hand side
    f0 = a * x - b * x * y
    f1 = -c * y + d * x * y

    ode = {}  # ODE declaration
    ode["x"] = casadi.vertcat(x, y)  # states
    ode["ode"] = casadi.vertcat(f0, f1)  # right-hand side
    x0 = np.ones(2)
    return (ode, 10.0, x0)
