import py_diffsol_new as ds

def test_api_concept():
    ode = ds.Ode(
        """
        in = [r, k, y0]
        r { 1 } k { 1 } y0 { 1 }
        u { y0 }
        F { r*u*(1 - u/k) }
        out { u }
        """,
        matrix_type=ds.nalgebra_dense_f64
    )

    config = ds.Config()
    config.method = ds.bdf
    config.linear_solver = ds.lu
    config.rtol = 1e-6

    p = "FIXME" # should be np.array([r, k, y0])
    bad_value = ode.solve(p)
    assert bad_value == "numpy_array"

    #ts, ys = ode.solve(p, config) # use ds.Config() if None
    #config.rtol = 1e-8
    #ts2, ys2 = ode.solve(p, config)

    #for t, y in zip(ts, ys):  # Note ts = 1d array, ys = 2d array
    #    expect = k*y0/(y0 + (k - y0)*np.exp(-r*t))
    #    err = np.abs(y[0] - expect)
    #    assert err < 1e-6

if __name__ == "__main__":
    test_api_concept()
