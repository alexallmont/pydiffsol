# This example is used in docs/examples/population_dynamics.rst.
# Ensure that code changes are reflected in rst literalinclude blocks.

import numpy as np
import matplotlib
matplotlib.use("SVG") # Ensure tests can run headless and in debug
import matplotlib.pyplot as plt
import pydiffsol as ds


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
        matrix_type=ds.nalgebra_dense,
        linear_solver=ds.lu,
        method=ds.bdf,
    )

    ode.rtol = 1e-6

    params = np.array([])
    ys, ts = ode.solve(params, 40.0)

    fig, ax = plt.subplots()
    ax.plot(ts, ys[0], label="prey")
    ax.plot(ts, ys[1], label="predator")
    ax.set_xlabel("t")
    ax.set_ylabel("population")
    fig.savefig("docs/images/prey_predator.svg")


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
        [prey, predator], _ = ode.solve(params, 40.0)
        ax.plot(prey, predator, label=f"y0 = {y0}")
    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    fig.savefig("docs/images/prey_predator2.svg")


# Smoke test docs code
def test_population_dynamics_docs():
    solve()
    phase_plane()


if __name__ == "__main__":
    solve()
    phase_plane()
