# pydiffsol

Python bindings for [diffsol](https://github.com/martinjrobins/diffsol)

- **Documentation:** https://pydiffsol.readthedocs.io/en/latest/
- **Source code:** https://github.com/alexallmont/pydiffsol/

## Example usage

```py
import pydiffsol as ds
import numpy as np

ode = ds.Ode(
    """
    r { 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
    """,
    ds.nalgebra_dense
)

# Solve with default solver for nalgebra_dense (bdf)
p = np.array([])
print(ode.solve(p, 0.4))

# Above defaults to bdf. Try esdirk34 instead
ode.method = ds.esdirk34
print(ode.solve(p, 0.4))
```
