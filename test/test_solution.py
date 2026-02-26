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


def logistic_sensitivity_with_respect_to_r(r, k, y0, t):
    exp_term = np.exp(-r * t)
    numerator = k * y0 * t * exp_term * (k - y0)
    denominator = (y0 + (k - y0) * exp_term) ** 2
    return numerator / denominator


def make_ode(scalar_type=ds.f64, method=ds.bdf):
    return ds.Ode(
        LOGISTIC_CODE,
        matrix_type=ds.nalgebra_dense,
        scalar_type=scalar_type,
        method=method,
        linear_solver=ds.lu,
    )


@pytest.mark.parametrize("scalar_type", [ds.f64, ds.f32])
@pytest.mark.parametrize("method", [ds.bdf, ds.esdirk34, ds.tr_bdf2, ds.tsit45])
def test_solution_can_be_updated_with_new_state(scalar_type, method):
    ode = make_ode(scalar_type=scalar_type, method=method)
    params = np.array([1.0, 1.0, 0.1])
    solution = ode.solve(params, 0.2)

    new_state = np.array([0.5])
    solution.current_state = new_state
    np.testing.assert_allclose(solution.current_state, new_state)
    
    solution = ode.solve(params, 0.4, solution)
    expected_final = logistic_solution(params[0], params[1], new_state[0], 0.2)
    np.testing.assert_allclose(solution.ys[0, -1], expected_final, rtol=3e-4, atol=2e-6)


def test_solution_is_consumed_and_appended_result_is_returned():
    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])
    t_eval_1 = np.array([0.0, 0.1, 0.2])
    t_eval_2 = np.array([0.3, 0.4])

    solution = ode.solve_dense(params, t_eval_1)
    ys_before = solution.ys.copy()
    ts_before = solution.ts.copy()

    solution_2 = ode.solve_dense(params, t_eval_2, solution)

    assert solution_2 is not solution
    with pytest.raises(RuntimeError, match="Solution payload missing"):
        _ = solution.ts
    with pytest.raises(RuntimeError, match="Solution payload missing"):
        _ = solution.ys

    np.testing.assert_allclose(solution_2.ys[:, : ys_before.shape[1]], ys_before)
    np.testing.assert_allclose(solution_2.ts[: ts_before.shape[0]], ts_before)
    np.testing.assert_allclose(solution_2.ts, np.concatenate([t_eval_1, t_eval_2]))


@pytest.mark.parametrize("scalar_type", [ds.f64, ds.f32])
@pytest.mark.parametrize("method", [ds.bdf, ds.esdirk34, ds.tr_bdf2, ds.tsit45])
def test_solution_split_solve_in_two(scalar_type, method):
    ode = make_ode(scalar_type=scalar_type, method=method)
    r = 1.0
    k = 1.0
    y0 = 0.1
    # pyo3 bindings currently accept f64 parameter arrays for all scalar types.
    params = np.array([r, k, y0], dtype=np.float64)
    t_split = 0.2
    t_final = 0.4

    solution = ode.solve(params, t_split)
    n_before = solution.ys.shape[1]
    solution = ode.solve(params, t_final, solution)
    assert solution.ys.shape[1] > n_before
    assert solution.ts[-1] == pytest.approx(t_final, rel=1e-5, abs=1e-5)

    expected_final = logistic_solution(r, k, y0, t_final)
    np.testing.assert_allclose(solution.ys[0, -1], expected_final, rtol=3e-4, atol=2e-6)
    np.testing.assert_allclose(solution.current_state, solution.ys[:, -1], rtol=1e-6, atol=1e-8)
   
    expected_trajectory = [logistic_solution(r, k, y0, t) for t in solution.ts]
    np.testing.assert_allclose(solution.ys[0], expected_trajectory, rtol=3e-4, atol=2e-6)
   
@pytest.mark.parametrize("scalar_type", [ds.f64, ds.f32])
@pytest.mark.parametrize("method", [ds.bdf, ds.esdirk34, ds.tr_bdf2, ds.tsit45])
def test_solution_split_solve_dense_in_two(scalar_type, method):
    ode = make_ode(scalar_type=scalar_type, method=method)
    r = 1.0
    k = 1.0
    y0 = 0.1
    # pyo3 bindings currently accept f64 parameter arrays for all scalar types.
    params = np.array([r, k, y0], dtype=np.float64)
    t_final = 0.4
    i_split = 3
    t_evals = np.array([0.0, 0.1, 0.2, 0.3, t_final])

    solution = ode.solve_dense(params, t_evals[:i_split])
    n_before = solution.ys.shape[1]
    solution = ode.solve_dense(params, t_evals[i_split:], solution)
    assert solution.ys.shape[1] > n_before
    assert solution.ts[-1] == pytest.approx(t_final, rel=1e-5, abs=1e-5)

    expected_trajectory = [logistic_solution(r, k, y0, t) for t in solution.ts]
    np.testing.assert_allclose(solution.ys[0], expected_trajectory, rtol=3e-4, atol=2e-6)
    
    
    
@pytest.mark.parametrize("scalar_type", [ds.f64, ds.f32])
@pytest.mark.parametrize("method", [ds.bdf, ds.esdirk34, ds.tr_bdf2, ds.tsit45])
def test_solution_split_solve_fwd_sens_in_two(scalar_type, method):
    if os.name == "nt":
        return
    ode = make_ode(scalar_type=scalar_type, method=method)
    r = 1.0
    k = 1.0
    y0 = 0.1
    # pyo3 bindings currently accept f64 parameter arrays for all scalar types.
    params = np.array([r, k, y0], dtype=np.float64)
    i_split = 3
    t_final = 0.4
    t_evals = np.array([0.0, 0.1, 0.2, 0.3, t_final])

    solution = ode.solve_fwd_sens(params, t_evals[:i_split])

    n_before = solution.ys.shape[1]
    solution = ode.solve_fwd_sens(params, t_evals[i_split:], solution)
    assert solution.ys.shape[1] > n_before
    assert len(solution.sens) == 3
    assert solution.sens[0].shape[1] == solution.ys.shape[1]
    assert solution.ts[-1] == pytest.approx(t_final, rel=1e-5, abs=1e-5)

    expected_final = logistic_solution(r, k, y0, t_final)
    np.testing.assert_allclose(solution.ys[0, -1], expected_final, rtol=3e-4, atol=2e-6)
    np.testing.assert_allclose(solution.current_state, solution.ys[:, -1], rtol=1e-6, atol=1e-8)
    
    expected_trajectory = [logistic_solution(r, k, y0, t) for t in solution.ts]
    np.testing.assert_allclose(solution.ys[0], expected_trajectory, rtol=3e-4, atol=2e-6)
    
    expected_r_sens = [logistic_sensitivity_with_respect_to_r(r, k, y0, t) for t in solution.ts]
    np.testing.assert_allclose(solution.sens[0][0], expected_r_sens, rtol=3e-4, atol=2e-6)


def test_reject_append_when_existing_solution_has_sens_but_new_segment_does_not():
    if os.name == "nt":
        return

    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])

    t_eval1 = np.array([0.0, 0.1, 0.2])
    t_eval2 = np.array([0.3, 0.4])

    solution = ode.solve_fwd_sens(params, t_eval1)
    ys_before = solution.ys.copy()
    ts_before = solution.ts.copy()
    sens_before = [s.copy() for s in solution.sens]
    state_before = solution.current_state.copy()

    with pytest.raises(Exception, match="Cannot append solution with sensitivities"):
        ode.solve_dense(params, t_eval2, solution)

    np.testing.assert_allclose(solution.ys, ys_before)
    np.testing.assert_allclose(solution.ts, ts_before)
    for s_before, s_after in zip(sens_before, solution.sens):
        np.testing.assert_allclose(s_after, s_before)
    np.testing.assert_allclose(solution.current_state, state_before)


def test_solution_current_state_rejects_wrong_length():
    ode = make_ode()
    params = np.array([1.0, 1.0, 0.1])
    solution = ode.solve(params, 0.2)
    state_before = solution.current_state.copy()

    with pytest.raises(ValueError, match="Expected current_state length 1 but got 2"):
        solution.current_state = np.array([0.2, 0.3])

    np.testing.assert_allclose(solution.current_state, state_before)


def test_solution_rejects_incompatible_ode_instance():
    ode_f64 = make_ode(ds.f64)
    ode_f32 = make_ode(ds.f32)
    params_f64 = np.array([1.0, 1.0, 0.1], dtype=np.float64)

    solution = ode_f64.solve(params_f64, 0.2)

    with pytest.raises(Exception, match="incompatible with this Ode instance"):
        ode_f32.solve(params_f64, 0.4, solution)


def test_solution_rejects_mixing_bdf_and_rk_state():
    ode_bdf = make_ode(method=ds.bdf)
    ode_rk = make_ode(method=ds.tsit45)
    params = np.array([1.0, 1.0, 0.1])

    solution_bdf = ode_bdf.solve(params, 0.2)

    with pytest.raises(Exception, match="Expected an RK state"):
        ode_rk.solve(params, 0.4, solution_bdf)
        
    solution_rk = ode_rk.solve(params, 0.2)
    with pytest.raises(Exception, match="Expected a BDF state"):
        ode_bdf.solve(params, 0.4, solution_rk)