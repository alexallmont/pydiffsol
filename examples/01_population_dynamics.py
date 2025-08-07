# Load matplotlib headless to avoid macos UI/window issues when debugging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
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
        ds.nalgebra_dense_f64
    )

    config = ds.Config()
    config.method = ds.bdf
    config.linear_solver = ds.lu
    config.rtol = 1e-6

    params = np.array([])
    ys, ts = ode.solve(params, 40.0, config)

    fig, ax = plt.subplots()
    ax.plot(ts, ys[0], label="prey")
    ax.plot(ts, ys[1], label="predator")
    ax.set_xlabel("t")
    ax.set_ylabel("population")
    fig.savefig("docs/images/prey-predator.png")


def phase_plane():
    ode = ds.Ode(
        """
        in = [ y0 ]
        y0 { 1.0 }
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
        ds.nalgebra_dense_f64
    )

    config = ds.Config()
    config.method = ds.bdf
    config.linear_solver = ds.lu
    config.rtol = 1e-6

    fig, ax = plt.subplots()
    for i in range(5):
        y0 = float(i + 1)
        params = np.array([y0])
        [prey, predator], _ = ode.solve(params, 40.0, config)
        ax.plot(prey, predator, label=f"y0 = {y0}")
    ax.set_xlabel("prey")
    ax.set_ylabel("predator")
    fig.savefig("docs/images/prey-predator2.png")

if __name__ == "__main__":
    solve()
    phase_plane()
