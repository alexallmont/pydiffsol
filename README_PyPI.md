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
    ds.nalgebra_dense_f64
)
p = np.array([])
print(ode.solve(p, 0.4))
```

## Licenses

This wheel bundles `libunwind.1.dylib` from LLVM, licensed under the Apache 2.0
License with LLVM exceptions, and `libzstd.1.dylib` from the Zstandard project,
licensed under the BSD 3-Clause License. See `LICENSE.libunwind` and
`LICENSE.zstd` for details.
