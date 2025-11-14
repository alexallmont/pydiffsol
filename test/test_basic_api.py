import numpy as np
import pydiffsol as ds
import pytest

LOGISTIC_CODE = \
"""
in = [r, k, y0]
r { 1.0 }
k { 1.0 }
y0 { 0.1 }
u { y0 }
F { r * u * (1.0 - u / k) }
"""

def test_basic_api():
    ode = ds.Ode(LOGISTIC_CODE, matrix_type=ds.nalgebra_dense, method=ds.bdf, linear_solver=ds.lu)

    r = 1.0
    k = 1.0
    y0 = 0.1
    params = np.array([r, k, y0])

    ys, ts = ode.solve(params, 0.4)

    assert len(ys) == 1
    assert len(ys[0]) == 28
    assert len(ts) == 28

    for i, t in enumerate(ts):
       expect = k * y0 / (y0 + (k - y0) * np.exp(-r * t))
       err = np.abs(ys[0][i] - expect)
       assert err < 1e-6

    # Check that when re-running, that solve generates new arrays, i.e. that ts
    # and ys are new objects and not referring to mutated data.
    ys2, ts2 = ode.solve(params, 1.0)

    # New solve generates more data due to larger t of 1.0
    assert len(ys2[0]) == 34
    assert len(ts2) == 34

    # Old arrays refer to original data for t = 0.4
    assert len(ys[0]) == 28
    assert len(ts) == 28

    # Sanity check that the python objects are unique
    assert id(ys) != id(ys2)
    assert id(ts) != id(ts2)

    # Example using solve_dense to get results at particular times
    t_eval = np.array([0.0, 0.1, 0.5])
    ys = ode.solve_dense(params, t_eval)
    assert np.allclose(ys, [[0.1, 0.109366, 0.154828]], rtol=1e-4)

    # Check that code read back matches original
    assert ode.code == LOGISTIC_CODE

if __name__ == "__main__":
    test_basic_api()
