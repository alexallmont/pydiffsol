import fix_matplotlib_debug as _

import pydiffsol as ds
import numpy as np
import matplotlib.pyplot as plt

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
        ds.nalgebra_dense_f64
    )

    params = np.array([])
    ys, ts = ode.solve(params, 40.0)

    fig, ax = plt.subplots()
    ax.plot(ts, ys[0], label="x")
    ax.set_xlabel("t")
    fig.savefig("docs/images/spring_mass_system.png")

if __name__ == "__main__":
    solve()
