import numpy as np
import pydiffsol as ds
import pytest
import os

LOGISTIC_CODE = """
in_i { r = 1, k = 1, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""

def logistic_solution(r, k, y0, t):
    return k * y0 / (y0 + (k - y0) * np.exp(-r * t))


def make_ode(scalar_type=ds.f64, method=ds.bdf):
    return ds.Ode(
        LOGISTIC_CODE,
        matrix_type=ds.nalgebra_dense,
        scalar_type=scalar_type,
        method=method,
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


def test_solve_appends_into_existing_solution():
    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])

    solution = ode.solve(params, 0.2)
    ncols_before = solution.ys.shape[1]
    nt_before = len(solution.ts)
    ys_before = solution.ys.copy()
    ts_before = solution.ts.copy()

    solution2 = ode.solve(params, 0.4, solution)

    assert solution2.ys.shape[1] > ncols_before
    assert len(solution2.ts) > nt_before
    np.testing.assert_allclose(solution2.ys[:, :ncols_before], ys_before)
    np.testing.assert_allclose(solution2.ts[:nt_before], ts_before)


def test_solve_dense_appends_into_existing_solution():
    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])

    t_eval1 = np.array([0.0, 0.1, 0.2])
    t_eval2 = np.array([0.3, 0.4])

    solution = ode.solve_dense(params, t_eval1)
    solution = ode.solve_dense(params, t_eval2, solution)

    np.testing.assert_allclose(solution.ts, np.concatenate([t_eval1, t_eval2]))
    assert solution.ys.shape == (1, len(t_eval1) + len(t_eval2))


def test_solve_fwd_sens_appends_into_existing_solution():
    if os.name == "nt":
        return

    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])

    t_eval1 = np.array([0.0, 0.1, 0.2])
    t_eval2 = np.array([0.3, 0.4])

    solution = ode.solve_fwd_sens(params, t_eval1)
    solution = ode.solve_fwd_sens(params, t_eval2, solution)

    np.testing.assert_allclose(solution.ts, np.concatenate([t_eval1, t_eval2]))
    assert solution.ys.shape == (1, len(t_eval1) + len(t_eval2))
    assert len(solution.sens) == 3
    for sens_i in solution.sens:
        assert sens_i.shape == (1, len(t_eval1) + len(t_eval2))


def test_reject_append_when_existing_solution_has_sens_but_new_segment_does_not():
    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])

    t_eval1 = np.array([0.0, 0.1, 0.2])
    t_eval2 = np.array([0.3, 0.4])

    solution = ode.solve_fwd_sens(params, t_eval1)
    ys_before = solution.ys.copy()
    ts_before = solution.ts.copy()
    sens_before = [s.copy() for s in solution.sens]

    with pytest.raises(Exception, match="Cannot append solution with sensitivities"):
        ode.solve_dense(params, t_eval2, solution)

    np.testing.assert_allclose(solution.ys, ys_before)
    np.testing.assert_allclose(solution.ts, ts_before)
    for s_before, s_after in zip(sens_before, solution.sens):
        np.testing.assert_allclose(s_after, s_before)


@pytest.mark.parametrize("scalar_type", [ds.f64, ds.f32])
@pytest.mark.parametrize("method", [ds.bdf, ds.esdirk34, ds.tr_bdf2, ds.tsit45])
def test_solution_current_state_round_trip(scalar_type, method):
    ode = make_ode(scalar_type=scalar_type, method=method)
    r = 1.0
    k = 1.0
    y0 = 0.1
    # pyo3 bindings currently accept f64 parameter arrays for all scalar types.
    params = np.array([r, k, y0], dtype=np.float64)
    state_reset = 0.5
    t_split = 0.2
    t_final = 0.4

    solution = ode.solve(params, t_split)
    solution.current_state = np.array([state_reset])
    np.testing.assert_allclose(solution.current_state, np.array([state_reset]))

    n_before = solution.ys.shape[1]
    solution = ode.solve(params, t_final, solution)
    assert solution.ys.shape[1] > n_before
    assert solution.ts[-1] == pytest.approx(t_final, rel=1e-5, abs=1e-5)

    expected_final = logistic_solution(r, k, state_reset, t_final - t_split)
    np.testing.assert_allclose(solution.ys[0, -1], expected_final, rtol=3e-4, atol=2e-6)
    np.testing.assert_allclose(solution.current_state, solution.ys[:, -1], rtol=1e-6, atol=1e-8)


def test_solution_rejects_incompatible_ode_instance():
    ode_f64 = make_ode(ds.f64)
    ode_f32 = make_ode(ds.f32)
    params_f64 = np.array([1.0, 1.0, 0.1], dtype=np.float64)

    solution = ode_f64.solve(params_f64, 0.2)

    with pytest.raises(Exception, match="incompatible with this Ode instance"):
        ode_f32.solve(params_f64, 0.4, solution)
