import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.numpy as jnp

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