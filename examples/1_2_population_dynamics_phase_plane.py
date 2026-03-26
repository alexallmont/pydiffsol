import numpy as np
import matplotlib.pyplot as plt
import pydiffsol as ds

def phase_plane():
    ode = ds.Ode(
        """
        in { y0 = 1 }
        a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
        u_i {
            y1 = y0,
            y2 = y0,
        }
        F_i {
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        }
        """,
        matrix_type=ds.nalgebra_dense,
        linear_solver=ds.lu,
        method=ds.bdf,
    )

    ode.rtol = 1e-6

    fig, ax = plt.subplots()
    for i in range(5):
        y0 = float(i + 1)
        params = np.array([y0])
        solution = ode.solve(params, 40.0)
        prey, predator = solution.ys
        ax.plot(prey, predator, label=f"y0 = {y0}")
    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    fig.savefig("docs/images/prey_predator2.svg")

if __name__ == "__main__":
    phase_plane()
