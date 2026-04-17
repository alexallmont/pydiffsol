import numpy as np
import pydiffsol as ds
import pytest


LOGISTIC_CODE = """
in_i { r = 1, k = 1, y0 = 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""


def make_ode(jit_backend, scalar_type=ds.f64, ode_solver=ds.bdf):
    return ds.Ode(
        LOGISTIC_CODE,
        jit_backend=jit_backend,
        matrix_type=ds.nalgebra_dense,
        scalar_type=scalar_type,
        ode_solver=ode_solver,
        linear_solver=ds.lu,
    )


@pytest.mark.parametrize("scalar_type", [ds.f64, ds.f32])
@pytest.mark.parametrize("ode_solver", [ds.bdf, ds.esdirk34, ds.tr_bdf2, ds.tsit45])
def test_solution_check_dims_and_dtypes(jit_backend, scalar_type, ode_solver):
    ode = make_ode(jit_backend, scalar_type=scalar_type, ode_solver=ode_solver)
    solution = ode.solve(np.array([1.0, 1.0, 0.1]), 0.4)

    assert solution.ys.ndim == 2
    assert solution.ts.ndim == 1
    assert solution.sens == []
    assert not hasattr(solution, "current_state")

    expected_dtype = np.float32 if scalar_type == ds.f32 else np.float64
    assert solution.ys.dtype == expected_dtype
    assert solution.ts.dtype == expected_dtype


def test_solution_objects_do_not_alias_between_solves(jit_backend):
    ode = make_ode(jit_backend)
    params = np.array([1.0, 1.0, 0.1])

    solution_1 = ode.solve(params, 0.2)
    ys_1 = solution_1.ys
    ts_1 = solution_1.ts

    solution_2 = ode.solve(params, 0.4)
    ys_2 = solution_2.ys
    ts_2 = solution_2.ts

    assert solution_1 is not solution_2
    assert id(ys_1) != id(ys_2)
    assert id(ts_1) != id(ts_2)


def test_solution_sensitivities_are_returned_for_forward_sensitivity_solves(jit_backend):
    if not hasattr(ds, "llvm"):
        pytest.skip("Forward sensitivities require an LLVM JIT backend")

    ode = make_ode(ds.llvm, scalar_type=ds.f64, ode_solver=ds.bdf)
    solution = ode.solve_fwd_sens(np.array([1.0, 1.0, 0.1]), np.array([0.0, 0.1, 0.5]))

    assert len(solution.sens) == 3
    assert solution.sens[0].shape == solution.ys.shape
