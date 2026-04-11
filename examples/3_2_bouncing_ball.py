import numpy as np
import matplotlib.pyplot as plt
import pydiffsol as ds
from _common import select_jit_backend

def solve():
    final_time = 10.0

    ode = ds.Ode(
        """
        in_i {
            g = 9.81,
            h = 10.0,
        }
        u_i {
            x = h,
            v = 0.0,
        }
        F_i {
            v,
            -g,
        }
        stop {
            x,
        }
        """,
        jit_backend=select_jit_backend(),
        matrix_type=ds.nalgebra_dense,
        ode_solver=ds.tsit45,
        linear_solver=ds.lu,
    )

    params = np.array([9.81, 10.0])
    t_eval = np.linspace(0.0, final_time, 200)
    solution = ode.solve_dense(params, t_eval)

    ts = solution.ts
    x, v = solution.ys

    fig, ax = plt.subplots()
    ax.plot(ts, x, label="x (height)")
    ax.plot(ts, v, label="v (velocity)")
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.legend()
    fig.savefig("docs/images/bouncing_ball.svg")

if __name__ == "__main__":
    solve()
