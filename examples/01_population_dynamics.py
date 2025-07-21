import pydiffsol as ds
import plotly.subplots as psp
import numpy as np

PLOTLY_ARGS = {
    "full_html": False,
    "include_plotlyjs": False
}

def solve():
    ode = ds.Ode(
        """
        a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
        u_i {
            y1 = 1,
            y2 = 1,
        }
        F_i {
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        }
        """,
        ds.nalgebra_dense_f64
    )

    config = ds.Config()
    config.method = ds.bdf
    config.linear_solver = ds.lu
    config.rtol = 1e-6

    params = np.array([])
    ys, ts = ode.solve(params, 40.0, config)

    fig = psp.make_subplots(rows=1, cols=1)
    fig.add_scatter(x=ts, y=ys[0], name="prey")
    fig.add_scatter(x=ts, y=ys[1], name="predator")
    fig.write_html("book/src/primer/images/prey-predator.html", **PLOTLY_ARGS)


def phase_plane():
    ode = ds.Ode(
        """
        in = [ y0 ]
        y0 { 1.0 }
        a { 2.0/3.0 } b { 4.0/3.0 } c { 1.0 } d { 1.0 }
        u_i {
            y1 = y0,
            y2 = y0,
        }
        F_i {
            a * y1 - b * y1 * y2,
            c * y1 * y2 - d * y2,
        }
        """,
        ds.nalgebra_dense_f64
    )

    config = ds.Config()
    config.method = ds.bdf
    config.linear_solver = ds.lu
    config.rtol = 1e-6

    fig = psp.make_subplots(rows=1, cols=1)
    for i in range(5):
        y0 = float(i + 1)
        params = np.array([y0])
        [prey, predator], _ = ode.solve(params, 40.0, config)
        fig.add_scatter(x=prey, y=predator, name=f"y0 = {y0}")

    fig.write_html("book/src/primer/images/prey-predator2.html", **PLOTLY_ARGS)


if __name__ == "__main__":
    solve()
    phase_plane()
