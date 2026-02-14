import numpy as np
import pydiffsol as ds
import pytest

LOGISTIC_CODE = """
in_i { r = 1, k = 1, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""

def logistic_solution(r, k, y0, t):
    return k * y0 / (y0 + (k - y0) * np.exp(-r * t))


def make_ode(scalar_type=ds.f64):
    return ds.Ode(
        LOGISTIC_CODE,
        matrix_type=ds.nalgebra_dense,
        scalar_type=scalar_type,
        method=ds.bdf,
        linear_solver=ds.lu,
    )


def test_solution_is_reused_and_appended_in_place():
    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])
    t_eval_1 = np.array([0.0, 0.1, 0.2])
    t_eval_2 = np.array([0.3, 0.4])

    solution = ode.solve_dense(params, t_eval_1)
    ys_before = solution.ys.copy()
    ts_before = solution.ts.copy()

    solution_2 = ode.solve_dense(params, t_eval_2, solution)

    # Both Python wrappers should reflect the same underlying Rust solution.
    np.testing.assert_allclose(solution.ts, solution_2.ts)
    np.testing.assert_allclose(solution.ys, solution_2.ys)
    np.testing.assert_allclose(solution_2.ys[:, : ys_before.shape[1]], ys_before)
    np.testing.assert_allclose(solution_2.ts[: ts_before.shape[0]], ts_before)
    np.testing.assert_allclose(solution_2.ts, np.concatenate([t_eval_1, t_eval_2]))


def test_solution_current_state_round_trip():
    ode = ds.Ode(
        LOGISTIC_CODE,
        matrix_type=ds.nalgebra_dense,
        scalar_type=ds.f64,
        method=ds.tsit45,
        linear_solver=ds.lu,
    )
    r = 1.0
    k = 1.0
    y0 = 0.1
    params = np.array([r, k, y0])
    state_reset = 0.5
    t_eval_1 = np.array([0.0, 0.2])
    t_eval_2 = np.array([0.3, 0.4])

    solution = ode.solve_dense(params, t_eval_1)
    solution.current_state = np.array([state_reset])
    np.testing.assert_allclose(solution.current_state, np.array([state_reset]))

    solution = ode.solve_dense(params, t_eval_2, solution)

    expected_ts = np.concatenate([t_eval_1, t_eval_2])
    expected_ys_1 = logistic_solution(r, k, y0, t_eval_1)
    expected_ys_2 = logistic_solution(r, k, state_reset, t_eval_2 - t_eval_1[-1])
    expected_ys = np.concatenate([expected_ys_1, expected_ys_2])

    np.testing.assert_allclose(solution.ts, expected_ts)
    np.testing.assert_allclose(solution.ys[0], expected_ys, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(solution.current_state, solution.ys[:, -1])


def test_solution_rejects_incompatible_ode_instance():
    ode_f64 = make_ode(ds.f64)
    ode_f32 = make_ode(ds.f32)
    params_f64 = np.array([1.0, 1.0, 0.1], dtype=np.float64)

    solution = ode_f64.solve(params_f64, 0.2)

    with pytest.raises(Exception, match="incompatible with this Ode instance"):
        ode_f32.solve(params_f64, 0.4, solution)
