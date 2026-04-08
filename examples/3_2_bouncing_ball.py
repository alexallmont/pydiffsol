import numpy as np
import matplotlib.pyplot as plt
import pydiffsol as ds

def solve():
    restitution = 0.8
    final_time = 10.0
    points_per_segment = 20

    # Using a single-step method makes event restarts robust after resetting
    # state at each bounce.
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
            x,          // stop when height x reaches zero
        }
        """,
        matrix_type=ds.nalgebra_dense,
        method=ds.tsit45,
        linear_solver=ds.lu,
    )

    # [g, h]
    params = np.array([9.81, 10.0])
    solution = None
    bounce_times = []
    t_start = 0.0

    while True:
        t_eval = np.linspace(t_start, final_time, points_per_segment + 1)
        solution = ode.solve_dense(params, t_eval, solution=solution)
        t_end = float(solution.ts[-1])

        if t_end >= final_time - 1e-12:
            break

        bounce_times.append(t_end)

        y = solution.current_state
        y[1] *= -restitution
        y[0] = max(y[0], np.finfo(float).eps)
        solution.current_state = y
        t_start = t_end

    ts = solution.ts
    x, v = solution.ys

    fig, ax = plt.subplots()
    ax.plot(ts, x, label="x (height)")
    ax.plot(ts, v, label="v (velocity)")
    for t_bounce in bounce_times:
        ax.axvline(t_bounce, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.legend()
    fig.savefig("docs/images/bouncing_ball.svg")

if __name__ == "__main__":
    solve()
