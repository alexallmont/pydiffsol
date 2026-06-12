# pydiffsol

Pydiffsol provides python bindings for [diffsol](https://github.com/martinjrobins/diffsol), a Rust library for solving ordinary differential equations (ODEs) or semi-explicit differential algebraic equations (DAEs).

- **Documentation:** [https://pydiffsol.readthedocs.io/en/latest](https://pydiffsol.readthedocs.io/en/latest)
- **Source code:** [https://github.com/alexallmont/pydiffsol](https://github.com/alexallmont/pydiffsol)

Equations are specified with [diffsl](https://github.com/martinjrobins/diffsl), a domain specific language (DSL) that uses automatic differentiation to calculate the necessary jacobians, and JIT compilation using [LLVM](https://llvm.org/) or [Cranelift](https://cranelift.dev/) to generate efficient native code at runtime.

This provides the performance of Rust with the flexibility of Python. Users create a Python `Ode` object with DiffSL code, specifying the diffsol solver, matrix, linear solver and scalar types. All standard solver configuration settings such as tolerances, min step size, max newton steps etc. can be set through the `Ode` instance.

Currently supported solver types are BDF, ESDIRK34, TRBDF2 and TSIT45.

Wheels are built for linux, windows and macos.

## Example usage

```py
import pydiffsol as ds
import numpy as np

ode = ds.Ode(
    """
    in { r = 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
    """,
    matrix_type=ds.nalgebra_dense,
)

# Solve up to t = 0.4, overriding r input param = 2.0
params = np.array([2.0])
solution = ode.solve(params, 0.4)
print(solution.ys, solution.ts)

# Above defaults to bdf. Try esdirk34 instead
ode.ode_solver = ds.esdirk34
solution = ode.solve(params, 0.4)
print(solution.ys, solution.ts)
```
