import numpy as np
import matplotlib.pyplot as plt
import pydiffsol as ds
from _common import select_jit_backend

def solve():
    ode = ds.Ode(
        """
        k { 1.0 } m { 1.0 } c { 0.1 }
        u_i {
            x = 1,
            v = 0,
        }
        F_i {
            v,
            -k/m * x - c/m * v,
        }
        """,
        jit_backend=select_jit_backend(),
        matrix_type=ds.nalgebra_dense,
    )

    params = np.array([])
    solution = ode.solve(params, 40.0)
    ys = solution.ys
    ts = solution.ts

    fig, ax = plt.subplots()
    ax.plot(ts, ys[0], label="x")
    ax.set_xlabel("t")
    fig.savefig("docs/images/spring_mass_system.svg")

if __name__ == "__main__":
    solve()
