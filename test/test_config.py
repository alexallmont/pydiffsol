import numpy as np
import pydiffsol as ds
import pytest

LOGISTIC_CODE = \
"""
in_i { r = 1, k = 1, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""

@pytest.mark.parametrize("rtol,atol", [(1e-3, 1e-6), (1e-6, 1e-9), (1e-9, 1e-12)])
def test_config(rtol, atol):
    ode = ds.Ode(LOGISTIC_CODE)
    ode.rtol = rtol
    assert ode.rtol == rtol
    ode.atol = atol
    assert ode.atol == atol
    ode.ic_use_linesearch = True
    assert ode.ic_use_linesearch is True
    ode.ic_max_linesearch_iterations = 5
    assert ode.ic_max_linesearch_iterations == 5
    ode.ic_max_newton_iterations = 15
    assert ode.ic_max_newton_iterations == 15
    ode.ic_step_reduction_factor = 0.5
    assert ode.ic_step_reduction_factor == 0.5
    ode.ic_armijo_constant = 1e-4
    assert ode.ic_armijo_constant == 1e-4
    ode.max_nonlinear_solver_iterations = 20
    assert ode.max_nonlinear_solver_iterations == 20
    ode.max_error_test_failures = 10
    assert ode.max_error_test_failures == 10
    ode.min_timestep = 1e-8
    assert ode.min_timestep == 1e-8

    r = 1.0
    k = 1.0
    y0 = 0.1
    params = np.array([r, k, y0])

    ys, ts = ode.solve(params, 0.4)

    expect = k * y0 / (y0 + (k - y0) * np.exp(-r * ts))
    np.testing.assert_allclose(ys[0], expect, rtol=rtol, atol=atol)
