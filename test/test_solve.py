import numpy as np
import pydiffsol as ds
import pytest
import os

LOGISTIC_CODE = \
"""
in_i { r = 1, k = 1, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""


def test_solve():
    ode = ds.Ode(
        LOGISTIC_CODE,
        matrix_type=ds.nalgebra_dense,
        scalar_type=ds.f64,
        method=ds.bdf,
        linear_solver=ds.lu
    )

    r = 1.0
    k = 1.0
    y0 = 0.1
    params = np.array([r, k, y0])

    ys, ts = ode.solve(params, 0.4)

    assert len(ys) == 1
    assert len(ys[0]) == len(ts)

    for i, t in enumerate(ts):
       expect = k * y0 / (y0 + (k - y0) * np.exp(-r * t))
       err = np.abs(ys[0][i] - expect)
       assert err < 1e-6

    # Check that when re-running, that solve generates new arrays, i.e. that ts
    # and ys are new objects and not referring to mutated data.
    ys2, ts2 = ode.solve(params, 1.0)

    assert len(ys2[0]) == len(ts2)
    assert len(ys[0]) == len(ts)

    # Sanity check that the python objects are unique
    assert id(ys) != id(ys2)
    assert id(ts) != id(ts2)

    # Example using solve_dense to get results at particular times
    t_eval = np.array([0.0, 0.1, 0.5])
    ys = ode.solve_dense(params, t_eval)
    assert np.allclose(ys, [[0.1, 0.109366, 0.154828]], rtol=1e-4)

    # Check that code read back matches original
    assert ode.code == LOGISTIC_CODE


@pytest.mark.parametrize("final_time", [0.4, 1.0, 2.0])
@pytest.mark.parametrize("params", [[1.0, 1.0, 0.1], [2.0, 0.5, 0.2]])
def test_solve_f32_near_f64(final_time, params):
    last_y = []
    last_t = []

    for scalar_type in [ds.f64, ds.f32]:
        ode = ds.Ode(
            LOGISTIC_CODE,
            matrix_type=ds.nalgebra_dense,
            scalar_type=scalar_type,
            method=ds.bdf,
            linear_solver=ds.lu
        )
        ys, ts = ode.solve(np.array(params), final_time)
        last_y.append(ys[0][-1])
        last_t.append(ts[-1])

    assert last_y[0] == pytest.approx(last_y[1], abs=1e-4)
    assert last_y[0].dtype == np.float64
    assert last_y[1].dtype == np.float32

    assert last_t[0] == pytest.approx(last_t[1], abs=1e-4)
    assert last_t[0].dtype == np.float64
    assert last_t[1].dtype == np.float32


def test_solve_fwd_sens():
    ode = ds.Ode(
        LOGISTIC_CODE,
        matrix_type=ds.nalgebra_dense,
        scalar_type=ds.f64,
        method=ds.bdf,
        linear_solver=ds.lu
    )

    r = 1.0
    k = 1.0
    y0 = 0.1
    params = np.array([r, k, y0])
    t_eval = np.array([0.0, 0.1, 0.5])
    if os.name == 'nt':
        with pytest.raises(Exception, match="Sensitivity analysis is not supported on Windows"):
            ys, sens = ode.solve_fwd_sens(params, t_eval)
        return
    ys, sens = ode.solve_fwd_sens(params, t_eval)
    assert ys.shape == (1, 3)
    assert len(sens) == 3
    assert sens[0].shape == (1, 3)
    assert sens[1].shape == (1, 3)
    assert sens[2].shape == (1, 3)
    u = k * y0
    v = (y0 + (k - y0) * np.exp(-r * t_eval))
    expect = u / v
    np.testing.assert_allclose(ys[0], expect, rtol=1e-4)
    expect_sens = np.array([
        (v * 0.0 - u * -t_eval * (k - y0) * np.exp(-r * t_eval)) / v**2,
        (v * y0 - u * np.exp(-r * t_eval)) / v**2,
        (v * k - u * (1.0 - np.exp(-r * t_eval))) / v**2
    ])
    for sens_i, expect_i, param_name in zip(sens, expect_sens, ['r', 'k', 'y0']):
        np.testing.assert_allclose(sens_i[0], expect_i, rtol=1e-4, err_msg=f"Sensitivity mismatch for param {param_name}")


def test_solve_sum_squares_adjoint():
    ode = ds.Ode(
        LOGISTIC_CODE,
        matrix_type=ds.nalgebra_dense,
        scalar_type=ds.f64,
        method=ds.bdf,
        linear_solver=ds.lu
    )

    r = 1.0
    k = 1.0
    y0 = 0.1
    params = np.array([r, k, y0])
    t_eval = np.array([0.0, 0.1, 0.5])
    data_params = np.array([0.9 * r, 0.9 * k, 0.9 * y0])
    data = ode.solve_dense(data_params, t_eval)
    if os.name == 'nt':
        with pytest.raises(Exception, match="Sensitivity analysis is not supported on Windows"):
            ys, sens = ode.solve_sum_squares_adj(params, data, t_eval)
        return
    ys, sens = ode.solve_sum_squares_adj(params, data, t_eval)

    assert isinstance(ys, float)
    assert sens.shape == (3,)

    u = k * y0
    v = (y0 + (k - y0) * np.exp(-r * t_eval))
    expect_y = u / v
    expect_sum_squares = np.sum((expect_y - data)**2)
    np.testing.assert_allclose(ys, expect_sum_squares, rtol=1e-4)

    expect_sens = np.array([
        (v * 0.0 - u * -t_eval * (k - y0) * np.exp(-r * t_eval)) / v**2,
        (v * y0 - u * np.exp(-r * t_eval)) / v**2,
        (v * k - u * (1.0 - np.exp(-r * t_eval))) / v**2
    ])

    # l = sum((y - data)^2)
    # dl/dp = sum(2 * (y - data) * dy/dp)
    expect_dsum_squares_dp = np.array([
        np.sum(2.0 * (expect_y - data) * expect_sens[0]),
        np.sum(2.0 * (expect_y - data) * expect_sens[1]),
        np.sum(2.0 * (expect_y - data) * expect_sens[2]),
    ])
    np.testing.assert_allclose(sens, expect_dsum_squares_dp, rtol=1e-4, err_msg=f"Adjoint sensitivity mismatch")


if __name__ == "__main__":
    test_solve()
