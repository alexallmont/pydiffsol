from functools import partial
import diffrax
import jax
import jax.numpy as jnp
from diffrax_lotka_volterra import LotkaVolterra
from diffrax_robertson import RobertsonOde

# Enable 64-bit precision in JAX, required solving problems
# with tolerances of 1e-8
# (see https://docs.kidger.site/diffrax/examples/stiff_ode/)
jax.config.update("jax_enable_x64", True)

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
