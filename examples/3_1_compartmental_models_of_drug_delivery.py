import numpy as np
import matplotlib.pyplot as plt
import pydiffsol as ds


def solve():
    ode = ds.Ode(
        """
        in_i {
            Vc = 1000.0,
            Vp1 = 1000.0,
            CL = 100.0,
            Qp1 = 50.0,
        }
        tdose_i { -1.0, 6.0, 12.0, 18.0 }
        dose_i { 0.0, 1000.0, 1000.0, 1000.0 }
        u_i {
            qc = 1000.0,
            qp1 = 0.0,
        }
        F_i {
            - qc / Vc * CL - Qp1 * (qc / Vc - qp1 / Vp1),
            Qp1 * (qc / Vc - qp1 / Vp1),
        }
        stop_i {
            tdose_i - t,
        }
        reset_i {
            qc + dose_i[N],
            qp1,
        }
        """,
        matrix_type=ds.nalgebra_dense,
        ode_solver=ds.tsit45,
        linear_solver=ds.lu,
    )

    params = np.array([1000.0, 1000.0, 100.0, 50.0])
    solution = ode.solve_hybrid(params, 24.0)

    ts = solution.ts
    q_c, q_p1 = solution.ys

    fig, ax = plt.subplots()
    ax.plot(ts, q_c, label="q_c (central)")
    ax.plot(ts, q_p1, label="q_p1 (peripheral)")
    ax.set_xlabel("t [h]")
    ax.set_ylabel("amount [ng]")
    ax.legend()
    fig.savefig("docs/images/compartmental_drug_delivery.svg")


if __name__ == "__main__":
    solve()
