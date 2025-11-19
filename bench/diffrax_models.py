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


class LotkaVolterra(eqx.Module):
    ngroups: int

    def __call__(self, t, y, args):
        a = 2.0 / 3.0
        b = 4.0 / 3.0
        c = 1.0
        d = 1.0

        f0 = a * y[0] - b * y[0] * y[1]
        f1 = -c * y[1] + d * y[0] * y[1]
        return jnp.vstack([f0, f1]).flatten()


def setup(ngroups: int, tol: float, method: str, problem: str):
    if problem == "robertson_ode":
        t_final = 1e10
        y0 = jnp.concatenate([jnp.ones(ngroups), jnp.zeros(2 * ngroups)])
        problem = RobertsonOde(ngroups=ngroups)
    elif problem == "lotka_volterra_ode":
        y0 = jnp.ones(2)
        t_final = 10.0
        problem = LotkaVolterra(ngroups=1)
    else:
        raise ValueError(f"Unknown problem: {problem}")
    if method == "kvaerno5":
        solver = diffrax.Kvaerno5()
    elif method == "tsit5":
        solver = diffrax.Tsit5()
    else:
        raise ValueError(f"Unknown method: {method}")
    return (problem, tol, t_final, solver, HashableArrayWrapper(y0))


# https://github.com/jax-ml/jax/issues/4572#issuecomment-709809897
def some_hash_function(x):
    return int(jnp.sum(x))


class HashableArrayWrapper:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return some_hash_function(self.val)

    def __eq__(self, other):
        return (isinstance(other, HashableArrayWrapper) and jnp.all(jnp.equal(self.val, other.val)))


@partial(jax.jit, static_argnames=["model"])
def bench(model) -> jnp.ndarray:
    (model, tol, t_final, solver, y0) = model
    terms = diffrax.ODETerm(model)
    stepsize_controller = diffrax.PIDController(rtol=tol, atol=tol)

    t0 = 0.0
    t1 = t_final
    dt0 = None
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0.val,
        stepsize_controller=stepsize_controller,
    )
    return sol.ys[-1]
