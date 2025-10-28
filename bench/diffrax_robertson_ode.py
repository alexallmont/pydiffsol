from functools import partial
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision in JAX, required solving problems
# with tolerances of 1e-8
# (see https://docs.kidger.site/diffrax/examples/stiff_ode/)
jax.config.update("jax_enable_x64", True)


class RobertsonOde(eqx.Module):
    ngroups: int

    def __call__(self, t, y, args):
        k1 = 0.04
        k2 = 30000000.0
        k3 = 10000.0

        xs = slice(0, self.ngroups)
        ys = slice(self.ngroups, 2 * self.ngroups)
        zs = slice(2 * self.ngroups, 3 * self.ngroups)
        f0 = -k1 * y[xs] + k3 * y[ys] * y[zs]
        f1 = k1 * y[xs] - k2 * y[ys] ** 2 - k3 * y[ys] * y[zs]
        f2 = k2 * y[ys] ** 2
        return jnp.vstack([f0, f1, f2]).flatten()


def setup(ngroups: int):
    robertson = RobertsonOde(ngroups=ngroups)
    return robertson


@partial(jax.jit, static_argnames=["model", "ngroups", "tol"])
def bench(model, ngroups: int, tol: float, t_final: float) -> jnp.ndarray:
    terms = diffrax.ODETerm(model)
    stepsize_controller = diffrax.PIDController(rtol=tol, atol=tol)

    t0 = 0.0
    t1 = t_final
    y0 = jnp.concatenate([jnp.ones(ngroups), jnp.zeros(2 * ngroups)])
    dt0 = None
    solver = diffrax.Kvaerno5()
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
    )
    return sol.ys[-1]
