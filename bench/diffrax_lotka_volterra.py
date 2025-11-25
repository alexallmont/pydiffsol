import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.numpy as jnp

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