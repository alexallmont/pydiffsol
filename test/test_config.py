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
def test_config_tol(rtol, atol):
    ode = ds.Ode(LOGISTIC_CODE)
    ode.rtol = rtol
    assert ode.rtol == rtol
    ode.atol = atol
    assert ode.atol == atol

    r = 1.0
    k = 1.0
    y0 = 0.1
    params = np.array([r, k, y0])

    ys, ts = ode.solve(params, 0.4)

    expect = k * y0 / (y0 + (k - y0) * np.exp(-r * ts))
    np.testing.assert_allclose(ys[0], expect, rtol=rtol, atol=atol)
    
def test_config_ic():
    ode = ds.Ode(LOGISTIC_CODE)
    ic_opts = ode.ic_options
    assert isinstance(ic_opts, ds.InitialConditionSolverOptions)

    ic_opts.use_linesearch = True
    assert ode.ic_options.use_linesearch is True
    ic_opts.max_linesearch_iterations = 10
    assert ode.ic_options.max_linesearch_iterations == 10
    ic_opts.max_newton_iterations = 20
    assert ode.ic_options.max_newton_iterations == 20
    ic_opts.max_linear_solver_setups = 5
    assert ode.ic_options.max_linear_solver_setups == 5
    ic_opts.step_reduction_factor = 0.4
    assert ode.ic_options.step_reduction_factor == 0.4
    ic_opts.armijo_constant = 1e-5
    assert ode.ic_options.armijo_constant == 1e-5
    

def test_config_ode():
    ode = ds.Ode(LOGISTIC_CODE)
    ode_opts = ode.options
    assert isinstance(ode_opts, ds.OdeSolverOptions)
    
    ode_opts.max_nonlinear_solver_iterations = 25
    assert ode.options.max_nonlinear_solver_iterations == 25
    ode_opts.max_error_test_failures = 12
    assert ode.options.max_error_test_failures == 12
    ode_opts.min_timestep = 1e-10
    assert ode.options.min_timestep == 1e-10
    ode_opts.update_jacobian_after_steps = 7
    assert ode.options.update_jacobian_after_steps == 7
    ode_opts.update_rhs_jacobian_after_steps = 9
    assert ode.options.update_rhs_jacobian_after_steps == 9
    ode_opts.threshold_to_update_jacobian = 1e-3
    assert ode.options.threshold_to_update_jacobian == 1e-3
    ode_opts.threshold_to_update_rhs_jacobian = 1e-4
    assert ode.options.threshold_to_update_rhs_jacobian == 1e-4
    
    # delete ode and make sure options persist
    del ode
    ode_opts.threshold_to_update_rhs_jacobian = 1e-4
    assert ode_opts.threshold_to_update_rhs_jacobian == 1e-4