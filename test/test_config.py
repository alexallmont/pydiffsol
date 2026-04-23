import numpy as np
import pydiffsol as ds
import pytest


LOGISTIC_CODE = """
in_i { r = 1, k = 1, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""


def make_ode(jit_backend):
    return ds.Ode(
        LOGISTIC_CODE,
        jit_backend=jit_backend,
        matrix_type=ds.nalgebra_dense,
        scalar_type=ds.f64,
        linear_solver=ds.lu,
        ode_solver=ds.bdf,
    )


@pytest.mark.parametrize("rtol, atol", [(1e-3, 1e-6), (1e-6, 1e-9), (1e-9, 1e-12)])
def test_config_tol(jit_backend, rtol, atol):
    ode = make_ode(jit_backend)
    ode.rtol = rtol
    ode.atol = atol

    params = np.array([1.0, 1.0, 0.1])
    solution = ode.solve(params, 0.4)
    expected = 0.1 / (0.1 + 0.9 * np.exp(-solution.ts))
    np.testing.assert_allclose(solution.ys[0], expected, rtol=rtol, atol=atol)


def test_config_ic(jit_backend):
    ode = make_ode(jit_backend)
    ic_opts = ode.ic_options
    assert isinstance(ic_opts, ds.InitialConditionSolverOptions)

    ic_opts.use_linesearch = True
    ic_opts.max_linesearch_iterations = 10
    ic_opts.max_newton_iterations = 20
    ic_opts.max_linear_solver_setups = 5
    ic_opts.step_reduction_factor = 0.4
    ic_opts.armijo_constant = 1e-5

    assert ode.ic_options.use_linesearch is True
    assert ode.ic_options.max_linesearch_iterations == 10
    assert ode.ic_options.max_newton_iterations == 20
    assert ode.ic_options.max_linear_solver_setups == 5
    assert ode.ic_options.step_reduction_factor == 0.4
    assert ode.ic_options.armijo_constant == 1e-5


def test_config_ode(jit_backend):
    ode = make_ode(jit_backend)
    ode_opts = ode.options
    assert isinstance(ode_opts, ds.OdeSolverOptions)

    ode_opts.max_nonlinear_solver_iterations = 25
    ode_opts.max_error_test_failures = 12
    ode_opts.min_timestep = 1e-10
    ode_opts.update_jacobian_after_steps = 7
    ode_opts.update_rhs_jacobian_after_steps = 9
    ode_opts.threshold_to_update_jacobian = 1e-3
    ode_opts.threshold_to_update_rhs_jacobian = 1e-4

    assert ode.options.max_nonlinear_solver_iterations == 25
    assert ode.options.max_error_test_failures == 12
    assert ode.options.min_timestep == 1e-10
    assert ode.options.update_jacobian_after_steps == 7
    assert ode.options.update_rhs_jacobian_after_steps == 9
    assert ode.options.threshold_to_update_jacobian == 1e-3
    assert ode.options.threshold_to_update_rhs_jacobian == 1e-4

    del ode
    ode_opts.threshold_to_update_rhs_jacobian = 2e-4
    assert ode_opts.threshold_to_update_rhs_jacobian == 2e-4
