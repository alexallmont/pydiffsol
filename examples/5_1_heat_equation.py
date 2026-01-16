import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pydiffsol as ds


def solve():
    ode = ds.Ode(
        """
        D { 0.1 }
        h { 1.0 / 21.0 }
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
        # Note that faer_sparse may be a better choice than nalgebra_dense for
        # larger systems because the RHS Jacobian will mostly be zeroes.
        ds.nalgebra_dense,
    )
    params = np.array([])
    ys, ts = ode.solve(params, 0.1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(
        *np.meshgrid(ts, np.arange(ys.shape[0])), ys, # X, Y, Z
        cmap=cm.coolwarm,
    )
    ax.grid(False)
    ax.view_init(15, -30, 0)
    ax.set_box_aspect(None, zoom=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("h")
    ax.set_zlabel("T")
    fig.savefig("docs/images/heat_equation.svg")


if __name__ == "__main__":
    solve()
