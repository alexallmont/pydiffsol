import pydiffsol as ds
import numpy as np
import matplotlib.pyplot as plt

def solve():
    ode = ds.Ode(
        """
        R { 100.0 } L { 1.0 } C { 0.001 } V0 { 10 } omega { 100.0 }
        Vs { V0 * sin(omega * t) }
        u_i {
            iR = 0,
            iL = 0,
            iC = 0,
            V = 0,
        }
        dudt_i {
            diRdt = 0,
            diLdt = 0,
            diCdt = 0,
            dVdt = 0,
        }
        M_i {
            0,
            diLdt,
            0,
            dVdt,
        }
        F_i {
            V - R * iR,
            (Vs - V) / L,
            iL - iR - iC,
            iC / C,
        }
        out_i {
            iR,
        }
        """,
        ds.nalgebra_dense
    )

    params = np.array([])
    ys, ts = ode.solve(params, 1.0)

    fig, ax = plt.subplots()
    ax.plot(ts, ys[0], label="x")
    ax.set_xlabel("t")
    ax.set_ylabel("current")
    fig.savefig("docs/images/electrical_circuits.svg")


if __name__ == "__main__":
    solve()
