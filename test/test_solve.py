import os

import numpy as np
import pydiffsol as ds
import pytest


LOGISTIC_CODE = """
in_i { r = 1, k = 1, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""

HYBRID_LOGISTIC_CODE = """
in_i { r = 1 }
u_i { y = 0.1 }
dudt_i { dydt = 0 }
F_i { (r * y) * (1 - y) }
stop_i { y - 0.9 }
reset_i { 0.1 }
out_i { y }
"""


def logistic_solution(r, k, y0, t):
    return k * y0 / (y0 + (k - y0) * np.exp(-r * t))


def logistic_sensitivity_with_respect_to_r(r, k, y0, t):
    exp_term = np.exp(-r * t)
    numerator = k * y0 * t * exp_term * (k - y0)
    denominator = (y0 + (k - y0) * exp_term) ** 2
    return numerator / denominator


def make_ode(
    jit_backend,
    *,
    code=LOGISTIC_CODE,
    matrix_type=ds.nalgebra_dense,
    scalar_type=ds.f64,
    ode_solver=ds.bdf,
    linear_solver=ds.lu,
):
    return ds.Ode(
        code,
        jit_backend=jit_backend,
        matrix_type=matrix_type,
        scalar_type=scalar_type,
        ode_solver=ode_solver,
        linear_solver=linear_solver,
    )


def test_solve(jit_backend):
    ode = make_ode(jit_backend, scalar_type=ds.f64, ode_solver=ds.bdf)
    params = np.array([1.0, 1.0, 0.1])

    assert ode.code == LOGISTIC_CODE
    assert ode.matrix_type == ds.nalgebra_dense
    assert ode.scalar_type == ds.f64
    assert ode.ode_solver == ds.bdf
    assert ode.linear_solver == ds.lu
    assert ode.nparams == 3
    assert ode.nstates == 1
    assert ode.nout == 1
    assert ode.has_stop() is False

    solution = ode.solve(params, 0.4)
    for i, t in enumerate(solution.ts):
        expect = logistic_solution(1.0, 1.0, 0.1, t)
        assert abs(solution.ys[0, i] - expect) < 1e-6

    y0 = ode.y0(params)
    rhs = ode.rhs(params, 0.0, np.array([0.25]))
    jac_mul = ode.rhs_jac_mul(params, 0.0, np.array([0.25]), np.array([3.0]))
    np.testing.assert_allclose(y0, np.array([0.1]))
    np.testing.assert_allclose(rhs, np.array([0.1875]))
    np.testing.assert_allclose(jac_mul, np.array([1.5]))


@pytest.mark.parametrize("final_time", [0.4, 1.0, 2.0])
@pytest.mark.parametrize("params", [[1.0, 1.0, 0.1], [2.0, 0.5, 0.2]])
def test_solve_f32_near_f64(jit_backend, final_time, params):
    results = []
    for scalar_type in [ds.f64, ds.f32]:
        ode = make_ode(jit_backend, scalar_type=scalar_type, ode_solver=ds.bdf)
        solution = ode.solve(np.array(params), final_time)
        results.append((solution.ys[0, -1], solution.ts[-1], solution.ys.dtype, solution.ts.dtype))

    assert results[0][0] == pytest.approx(results[1][0], abs=1e-4)
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-4)
    assert results[0][2] == np.float64
    assert results[0][3] == np.float64
    assert results[1][2] == np.float32
    assert results[1][3] == np.float32


def test_hybrid_metadata_and_solve_paths(jit_backend):
    ode = make_ode(
        jit_backend,
        code=HYBRID_LOGISTIC_CODE,
        scalar_type=ds.f64,
        ode_solver=ds.bdf,
        linear_solver=ds.default,
    )

    assert ode.nparams == 1
    assert ode.nstates == 1
    assert ode.nout == 1
    assert ode.has_stop() is True

    hybrid = ode.solve_hybrid(np.array([2.0]), 2.0)
    assert hybrid.ts[-1] == pytest.approx(2.0, rel=1e-5, abs=1e-5)

    t_eval = np.array([0.5, 1.0, 1.5, 2.0])
    hybrid_dense = ode.solve_hybrid_dense(np.array([2.0]), t_eval)
    assert hybrid_dense.ts.tolist() == pytest.approx(t_eval.tolist())

    if os.name != "nt" and hasattr(ds, "llvm"):
        sens_ode = make_ode(
            ds.llvm,
            code=HYBRID_LOGISTIC_CODE,
            scalar_type=ds.f64,
            ode_solver=ds.bdf,
            linear_solver=ds.default,
        )
        hybrid_sens = sens_ode.solve_hybrid_fwd_sens(np.array([2.0]), t_eval)
        assert len(hybrid_sens.sens) == 1
        assert hybrid_sens.sens[0].shape == hybrid_sens.ys.shape


def test_solve_fwd_sens(jit_backend):
    if not hasattr(ds, "llvm"):
        pytest.skip("Forward sensitivities require an LLVM JIT backend")

    ode = make_ode(ds.llvm, scalar_type=ds.f64, ode_solver=ds.bdf)
    params = np.array([1.0, 1.0, 0.1])
    t_eval = np.array([0.0, 0.1, 0.5])

    if os.name == "nt":
        with pytest.raises(Exception, match="Sensitivity analysis is not supported on Windows"):
            ode.solve_fwd_sens(params, t_eval)
        return

    solution = ode.solve_fwd_sens(params, t_eval)
    assert solution.ys.shape == (1, 3)
    assert len(solution.sens) == 3

    u = params[1] * params[2]
    v = params[2] + (params[1] - params[2]) * np.exp(-params[0] * t_eval)
    np.testing.assert_allclose(solution.ys[0], u / v, rtol=1e-4)

    expected_r = [logistic_sensitivity_with_respect_to_r(*params, t) for t in t_eval]
    np.testing.assert_allclose(solution.sens[0][0], expected_r, rtol=1e-4)


@pytest.mark.parametrize("scalar_type,data_dtype", [
    (ds.f64, np.float64),  # match:    f64 data → f64 ODE (borrow directly)
    (ds.f64, np.float32),  # mismatch: f32 data → f64 ODE (convert to f64)
    (ds.f32, np.float64),  # mismatch: f64 data → f32 ODE (convert to f32)
    (ds.f32, np.float32),  # match:    f32 data → f32 ODE (borrow directly)
])
def test_solve_sum_squares_adjoint(jit_backend, scalar_type, data_dtype):
    if not hasattr(ds, "llvm"):
        pytest.skip("Adjoint sensitivities require an LLVM JIT backend")

    params = np.array([1.0, 1.0, 0.1])
    t_eval = np.array([0.0, 0.1, 0.5])
    data_params = np.array([0.9, 0.9, 0.09])

    ode = make_ode(ds.llvm, scalar_type=scalar_type, ode_solver=ds.bdf)

    # Generate reference data using f64, then cast to the target data dtype
    ref_ode = make_ode(ds.llvm, scalar_type=ds.f64, ode_solver=ds.bdf)
    data = ref_ode.solve_dense(data_params, t_eval).ys.astype(data_dtype)
    assert data.dtype == data_dtype

    if os.name == "nt":
        with pytest.raises(Exception, match="Sensitivity analysis is not supported on Windows"):
            ode.solve_sum_squares_adj(params, data, t_eval)
        return

    value, sens = ode.solve_sum_squares_adj(params, data, t_eval)
    assert isinstance(value, float)
    assert sens.shape == (3,)
    assert np.isfinite(value)
    assert np.isfinite(sens).all()

    # For the f64/f64 case, also verify the value against the analytical solution
    if scalar_type == ds.f64 and data_dtype == np.float64:
        expected_y = np.array([logistic_solution(*params, t) for t in t_eval])
        np.testing.assert_allclose(value, np.sum((expected_y - data[0]) ** 2), rtol=1e-4)


def test_solve_sum_squares_adjoint_invalid_dtype(jit_backend):
    if not hasattr(ds, "llvm"):
        pytest.skip("Adjoint sensitivities require an LLVM JIT backend")

    ode = make_ode(ds.llvm, scalar_type=ds.f64, ode_solver=ds.bdf)
    params = np.array([1.0, 1.0, 0.1])
    t_eval = np.array([0.0, 0.1, 0.5])
    bad_data = np.zeros((1, 3), dtype=np.int32)

    with pytest.raises(Exception, match="float32 or float64"):
        ode.solve_sum_squares_adj(params, bad_data, t_eval)
