import numpy as np
import matplotlib.pyplot as plt
import pydiffsol as ds
from _common import select_jit_backend

def solve():
    ode = ds.Ode(
        """
        a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
        u_i {
            y1 = 1,
            y2 = 1,
        }
        F_i {
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        }
        """,
        jit_backend=select_jit_backend(),
        matrix_type=ds.nalgebra_dense,
        linear_solver=ds.lu,
        ode_solver=ds.bdf,
    )

    ode.rtol = 1e-6

    params = np.array([])
    solution = ode.solve(params, 40.0)
    ys = solution.ys
    ts = solution.ts

    fig, ax = plt.subplots()
    ax.plot(ts, ys[0], label="prey")
    ax.plot(ts, ys[1], label="predator")
    ax.set_xlabel("t")
    ax.set_ylabel("population")
    fig.savefig("docs/images/prey_predator.svg")

if __name__ == "__main__":
    solve()
