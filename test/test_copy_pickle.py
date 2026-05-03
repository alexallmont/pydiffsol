import copy
import pickle

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


def configure_ode(ode):
    ode.rtol = 1e-7
    ode.atol = 1e-9
    ode.ode_solver = ds.tr_bdf2
    ode.linear_solver = ds.lu

    ic_options = ode.ic_options
    ic_options.use_linesearch = True
    ic_options.max_newton_iterations = 17
    ic_options.max_linesearch_iterations = 13
    ic_options.max_linear_solver_setups = 11

    ode_options = ode.options
    ode_options.max_nonlinear_solver_iterations = 23
    ode_options.max_error_test_failures = 19
    ode_options.update_jacobian_after_steps = 7
    ode_options.update_rhs_jacobian_after_steps = 9


def assert_matching_state(left, right):
    assert left.code == right.code
    assert left.jit_backend == right.jit_backend
    assert left.matrix_type == right.matrix_type
    assert left.scalar_type == right.scalar_type
    assert left.linear_solver == right.linear_solver
    assert left.ode_solver == right.ode_solver
    assert left.rtol == pytest.approx(right.rtol)
    assert left.atol == pytest.approx(right.atol)

    assert left.ic_options.use_linesearch is right.ic_options.use_linesearch
    assert left.ic_options.max_newton_iterations == right.ic_options.max_newton_iterations
    assert (
        left.ic_options.max_linesearch_iterations
        == right.ic_options.max_linesearch_iterations
    )
    assert (
        left.ic_options.max_linear_solver_setups
        == right.ic_options.max_linear_solver_setups
    )

    assert (
        left.options.max_nonlinear_solver_iterations
        == right.options.max_nonlinear_solver_iterations
    )
    assert left.options.max_error_test_failures == right.options.max_error_test_failures
    assert left.options.update_jacobian_after_steps == right.options.update_jacobian_after_steps
    assert (
        left.options.update_rhs_jacobian_after_steps
        == right.options.update_rhs_jacobian_after_steps
    )

    params = np.array([1.0, 1.0, 0.1])
    y = np.array([0.25])
    np.testing.assert_allclose(left.y0(params), right.y0(params))
    np.testing.assert_allclose(left.rhs(params, 0.0, y), right.rhs(params, 0.0, y))
    t_eval = np.array([0.0, 0.2, 0.4])
    np.testing.assert_allclose(
        left.solve_dense(params, t_eval).ys,
        right.solve_dense(params, t_eval).ys,
    )


def test_copy_shares_underlying_solver_state(jit_backend):
    ode = make_ode(jit_backend)
    configure_ode(ode)

    ode_copy = copy.copy(ode)

    assert ode_copy is not ode
    assert ode_copy.code == ode.code

    ode_copy.rtol = 1e-5
    ode_copy.atol = 1e-8
    ode_copy.options.max_nonlinear_solver_iterations = 31
    ode_copy.ic_options.max_newton_iterations = 29

    assert ode.rtol == pytest.approx(1e-5)
    assert ode.atol == pytest.approx(1e-8)
    assert ode.options.max_nonlinear_solver_iterations == 31
    assert ode.ic_options.max_newton_iterations == 29


@pytest.mark.skipif(not hasattr(ds, "llvm"), reason="LLVM backend not available")
def test_deepcopy_roundtrips_independent_llvm_solver():
    ode = make_ode(ds.llvm)
    configure_ode(ode)

    ode_copy = copy.deepcopy(ode)

    assert ode_copy is not ode
    assert_matching_state(ode, ode_copy)

    ode_copy.rtol = 1e-4
    ode_copy.atol = 1e-6
    ode_copy.options.max_nonlinear_solver_iterations = 41
    ode_copy.ic_options.max_newton_iterations = 37

    assert ode.rtol == pytest.approx(1e-7)
    assert ode.atol == pytest.approx(1e-9)
    assert ode.options.max_nonlinear_solver_iterations == 23
    assert ode.ic_options.max_newton_iterations == 17


@pytest.mark.skipif(not hasattr(ds, "llvm"), reason="LLVM backend not available")
def test_getstate_and_setstate_roundtrip_full_solver_state():
    ode = make_ode(ds.llvm)
    configure_ode(ode)

    state = ode.__getstate__()
    assert isinstance(state, bytes)

    restored = make_ode(ds.llvm)
    restored.rtol = 1e-3
    restored.atol = 1e-4
    restored.__setstate__(state)

    assert_matching_state(ode, restored)


@pytest.mark.skipif(not hasattr(ds, "llvm"), reason="LLVM backend not available")
def test_pickle_roundtrip_restores_llvm_solver_state():
    ode = make_ode(ds.llvm)
    configure_ode(ode)

    restored = pickle.loads(pickle.dumps(ode))

    assert restored is not ode
    assert_matching_state(ode, restored)


@pytest.mark.skipif(not hasattr(ds, "llvm"), reason="LLVM backend not available")
def test_setstate_rejects_invalid_state_bytes():
    ode = make_ode(ds.llvm)

    with pytest.raises(ValueError):
        ode.__setstate__(b"not valid serialized state")


@pytest.mark.skipif(not hasattr(ds, "cranelift"), reason="Cranelift backend not available")
def test_cranelift_rejects_deepcopy_and_pickle_protocols():
    ode = make_ode(ds.cranelift)

    with pytest.raises(RuntimeError):
        copy.deepcopy(ode)

    with pytest.raises(RuntimeError):
        ode.__getstate__()

    with pytest.raises(RuntimeError):
        pickle.dumps(ode)
