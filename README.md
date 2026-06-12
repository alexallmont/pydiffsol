# pydiffsol

Pydiffsol provides python bindings for [diffsol](https://github.com/martinjrobins/diffsol), a Rust library for solving ordinary differential equations (ODEs) or semi-explicit differential algebraic equations (DAEs).

- **PyPI**: [https://pypi.org/project/pydiffsol](https://pypi.org/project/pydiffsol)
- **Documentation**: [https://pydiffsol.readthedocs.io/en/latest](https://pydiffsol.readthedocs.io/en/latest)

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

## Known issues

- Instability for BDF with FaerSparse KLU. We are investigating a segfault in
underlying diffsol. In the meantime, unit tests are disabled for this combination.

## Local development

To build locally, create a venv and use [maturin](https://www.maturin.rs/installation.html)
to set up your environment, optionally setting `diffsol-llvm` to your installed
LLVM and enable `suitesparse` if you have it installed (required for sparse
matrix types). Also specify `dev` extras for pytest, running examples and docs
image generation. For example:

```sh
maturin develop --extras dev --features diffsol-llvm17 --features suitesparse
```

The `.vscode` setup includes examples for running tests and examples in python
via lldb so underlying rust can be debugged. The build task in `tasks.json` runs
with `diffsol-llvm17` and `suitesparse` and assumes that you have these
installed, for example on macos with `brew install llvm@17 suite-sparse` or for
debian-flavoured linux `apt install llvm-17 libsuitesparse-dev`. If you have a
different configuration, you may need to edit `tasks.json` and `settings.json`.

The python path is hard-coded in `launch.json` to `.venv/bin/activate` (this is
the default when running `uv` in macos or Linux). If you have pip-installed
to a different location or running on Windows, you need to edit `launch.json`.

## Local wheel builds

To replicate CI wheel builds, specify CIBW_ environment variables equivalent to
those in `.github/workflows/CI.yml`. Using `cibuildwheel` performs the repair
step required to generate the .pyi autocomplete stubs.

For example, building alternative macos deployment target with python 3.11:

```sh
    CIBW_BUILD=cp311-macosx_arm64 \
    MACOSX_DEPLOYMENT_TARGET=15.0 \
    MATURIN_PEP517_ARGS="--features diffsol-llvm17" \
    cibuildwheel --platform macos .
```
