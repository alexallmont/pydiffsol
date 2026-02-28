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
            dose0 = 1000.0,
        }
        u_i {
            qc = dose0,
            qp1 = 0.0,
        }
        F_i {
            - qc / Vc * CL - Qp1 * (qc / Vc - qp1 / Vp1),
            Qp1 * (qc / Vc - qp1 / Vp1),
        }
        """,
        matrix_type=ds.nalgebra_dense,
        method=ds.tsit45,
        linear_solver=ds.lu,
    )

    # [Vc, Vp1, CL, Qp1, dose0]
    params = np.array([1000.0, 1000.0, 100.0, 50.0, 1000.0])
    # (time [h], bolus amount [ng])
    doses = [(0.0, 1000.0), (6.0, 1000.0), (12.0, 1000.0), (18.0, 1000.0)]
    final_time = 24.0

    solution = None
    for i, (_t_dose, dose) in enumerate(doses):
        if solution:
            # Apply bolus directly to the central compartment.
            y = solution.current_state
            y[0] += dose
            solution.current_state = y

        next_t = doses[i + 1][0] if i + 1 < len(doses) else final_time
        solution = ode.solve(params, next_t, solution=solution)

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
