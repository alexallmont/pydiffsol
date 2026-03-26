# pydiffsol

Python bindings for [diffsol](https://github.com/martinjrobins/diffsol)

- **Documentation:** [https://pydiffsol.readthedocs.io/en/latest](https://pydiffsol.readthedocs.io/en/latest)
- **Source code:** [https://github.com/alexallmont/pydiffsol](https://github.com/alexallmont/pydiffsol)

## Example usage

```py
import pydiffsol as ds
import numpy as np

# DiffSl code and matrix type specified in constructor
# Defaults to f64 BDF solver unless specified
ode = ds.Ode(
    """
    in { r = 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
    """,
    ds.nalgebra_dense
)

# Solve up to t = 0.4, overriding r input param = 2.0
params = np.array([2.0])
solution = ode.solve(params, 0.4)
print(solution.ys, solution.ts)

# Above defaults to bdf. Try esdirk34 instead
ode.method = ds.esdirk34
solution = ode.solve(params, 0.4)
print(solution.ys, solution.ts)
```