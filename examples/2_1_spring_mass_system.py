# This example is used in docs/examples/spring_mass_systems.rst.
# Ensure that code changes are reflected in rst literalinclude blocks.

import numpy as np
import matplotlib
matplotlib.use("SVG") # Ensure tests can run headless and in debug
import matplotlib.pyplot as plt
import pydiffsol as ds


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
        ds.nalgebra_dense,
    )

    params = np.array([])
    ys, ts = ode.solve(params, 40.0)

    fig, ax = plt.subplots()
    ax.plot(ts, ys[0], label="x")
    ax.set_xlabel("t")
    fig.savefig("docs/images/spring_mass_system.svg")


# Smoke test docs code
def test_spring_mass_systems_docs():
    solve()


if __name__ == "__main__":
    solve()
