import numpy as np
import py_diffsol_new as ds


def test_basic_api():
    ode = ds.Ode(
        """
        in = [r, k, y0]
        r { 1.0 }
        k { 1.0 }
        y0 { 0.1 }
        u { y0 }
        F { r * u * (1.0 - u / k) }
        """,
        matrix_type=ds.nalgebra_dense_f64
    )

    config = ds.Config()
    config.method = ds.bdf
    config.linear_solver = ds.lu
    config.rtol = 1e-6

    r = 1.0
    k = 1.0
    y0 = 0.1
    p = np.array([r, k, y0])
    ys, ts = ode.solve(p, 0.4, config)

    for i, t in enumerate(ts):
       expect = k * y0 / (y0 + (k - y0) * np.exp(-r * t))
       err = np.abs(ys[0][i] - expect)
       assert err < 1e-6


if __name__ == "__main__":
    test_basic_api()
