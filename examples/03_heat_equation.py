import numpy as np
import matplotlib.pyplot as plt
import pydiffsol as ds


def solve():
    ode = ds.Ode(
        """
        D { 0.1 }
        h { 1.0 / 21.0}
        g { 0.0 }
        m { 1.0 }
        A_ij {
            (0..20, 1..21): 1.0,
            (0..21, 0..21): -2.0,
            (1..21, 0..20): 1.0,
        }
        b_i {
            (0): g,
            (1:20): 0.0,
            (20): g,
        }
        u_i {
            (0:5): g,
            (5:15): g + m,
            (15:21): g,
        }
        heat_i { A_ij * u_j }
        F_i {
            D * (heat_i + b_i) / (h * h)
        }
        out_i {
            u_i
        }
        """,
        ds.nalgebra_dense,
    )
    params = np.array([])
    ys, ts = ode.solve(params, 0.1)

    fig, ax = plt.subplots()
    ax.plot(ys[:,-1], label="x")
    ax.set_xlabel("x")
    ax.set_ylabel("T")
    fig.savefig("docs/images/heat_equation.svg")

solve()
