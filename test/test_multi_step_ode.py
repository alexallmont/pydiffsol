import numpy as np
import pydiffsol as ds
import pydiffsol.multi_step_ode as mso
import pytest

STEP1_LOGISTIC = """
in_i { r = 1.0, k = 1.0, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""

MAPPER_DOUBLE = """
in_i { r = 1.0, k = 1.0, y0 = 0.1 }
u { y0 }
F { 2.0 * u }
"""

STEP2_DECAY = """
in_i { r = 1.0, k = 1.0, y0 = 0.1 }
u { y0 }
F { -r * u }
stop { u - 0.01 }
"""

def _make_multistep_ode():
    cfg = ds.MultiStepOdeConfig(
        models=[STEP1_LOGISTIC, MAPPER_DOUBLE, STEP2_DECAY],
        steps=[0, 2],
        mappers=[1],
        duration=[1.0, None],
    )
    return ds.MultiStepOde(
        cfg,
        matrix_type=ds.nalgebra_dense,
        scalar_type=ds.f64,
        method=ds.bdf,
        linear_solver=ds.lu,
    )


def _multistep_final_time_solution(r: float, k: float, y0: float):
    step_1_final = k * y0 / (y0 + (k - y0) * np.exp(-r * 1.0))
    step_2_initial = 2.0 * step_1_final
    step_2_final_time = np.log(step_2_initial / 0.01) / r + 1.0
    return step_2_final_time


def _multistep_solution(r: float, k: float, y0: float, t: float):
    step1 = k * y0 / (y0 + (k - y0) * np.exp(-r * t))
    step_1_final = k * y0 / (y0 + (k - y0) * np.exp(-r * 1.0))
    step_2_initial = 2.0 * step_1_final
    step2 = step_2_initial * np.exp(-r * (t - 1.0))

    if t < 1.0:
        return step1
    elif t != 1.0:
        assert step2 >= 0.01 - 1e-6, "Stop event should have triggered"
        return step2
    else:
        return (step1, step2)  # at t=1.0, both models are active


@pytest.mark.parametrize("final_time", [0.5, 1.0, 1.5, 2.0, 100.0])
def test_multistep_solve(final_time):
    ode = _make_multistep_ode()
    params = np.array([1.0, 1.0, 0.1])
    sol = ode.solve(params, final_time)
    model_final_time = _multistep_final_time_solution(params[0], params[1], params[2])
    np.testing.assert_allclose(sol.ts[-1], min(final_time, model_final_time), rtol=1e-4, atol=1e-6)
    expected = [_multistep_solution(params[0], params[1], params[2], t) for t in sol.ts]
    # at t = 1.0, there are two solutions, so we need to handle that case separately
    for i, (s, e) in enumerate(zip(sol.ys[0], expected)):
        if isinstance(e, tuple):
            # s can be either the step 1 or step 2 solution, as long as it is close to one of them
            assert s == pytest.approx(e[0], rel=1e-4, abs=1e-6) or s == pytest.approx(e[1], rel=1e-4, abs=1e-6)
            expected[i] = s
    print((sol.ys[0] - np.array(expected)) / np.abs(expected)) 
    np.testing.assert_allclose(sol.ys[0], expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("final_time", [0.5, 1.0, 1.5, 2.0, 100.0])
def test_multistep_solve_dense(final_time):
    ode = _make_multistep_ode()
    params = np.array([1.0, 1.0, 0.1])
    t_evals = np.linspace(0.0, final_time, 100)
    model_final_time = _multistep_final_time_solution(params[0], params[1], params[2])

    sol = ode.solve_dense(params, t_evals)
    expected_ts = t_evals[t_evals <= model_final_time]
    # if final_time > 1.0, we need to include this twice in expected_ts
    if final_time > 1.0:
        if 1.0 not in expected_ts:
            expected_ts = np.append(expected_ts, [1.0, 1.0])
            expected_ts.sort()
        else:
            expected_ts = np.append(expected_ts, 1.0)
            expected_ts.sort()
    # if final_time > model_final_time, we need to include model_final_time in expected_ts
    if final_time > model_final_time and model_final_time not in expected_ts:
        expected_ts = np.append(expected_ts, model_final_time)
        expected_ts.sort() 
    print(f"expected_ts: {expected_ts}")
    print(f"sol.ts: {sol.ts}")
    np.testing.assert_allclose(sol.ts, expected_ts, rtol=1e-4, atol=1e-6)
    expected = [_multistep_solution(params[0], params[1], params[2], t) for t in sol.ts]
    # at t = 1.0, there are two solutions, so we need to handle that case separately
    for i, (s, e) in enumerate(zip(sol.ys[0], expected)):
        if isinstance(e, tuple):
            # s can be either the step 1 or step 2 solution, as long as it is close to one of them
            assert s == pytest.approx(e[0], rel=1e-4, abs=1e-6) or s == pytest.approx(e[1], rel=1e-4, abs=1e-6)
            expected[i] = s
    error = np.abs(sol.ys[0] - np.array(expected)) / np.abs(expected)
    print(f"bad ts: {sol.ts[error > 1e-4]}")
    print(f"bad ys: {sol.ys[0][error > 1e-4]}")
    print(f"expected: {np.array(expected)[error > 1e-4]}")
    print(f"all ys: {sol.ys[0]}")
    np.testing.assert_allclose(sol.ys[0], expected, rtol=1e-4, atol=1e-6)


def test_multistep_options():
    ode = _make_multistep_ode()

    ode.rtol = 1e-5
    ode.atol = 1e-8
    ode.method = ds.tr_bdf2
    ode.options.max_error_test_failures = 11
    ode.ic_options.max_newton_iterations = 13

    assert ode.rtol == 1e-5
    assert ode.atol == 1e-8
    assert ode.method == ds.tr_bdf2
    assert ode.options.max_error_test_failures == 11
    assert ode.ic_options.max_newton_iterations == 13


def _discover_fields(obj):
    return sorted(
        name
        for name in dir(obj)
        if not name.startswith("_") and not callable(getattr(obj, name))
    )


def test_explicit_option_field_lists_match_runtime_fields():
    ode = ds.Ode(STEP1_LOGISTIC)

    runtime_ode_fields = _discover_fields(ode.options)
    runtime_ic_fields = _discover_fields(ode.ic_options)

    assert sorted(mso.ODE_OPTION_FIELDS) == runtime_ode_fields
    assert sorted(mso.IC_OPTION_FIELDS) == runtime_ic_fields
