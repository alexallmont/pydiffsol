# pydiffsol

Python bindings for [diffsol](https://github.com/martinjrobins/diffsol)

- **PyPI**: https://pypi.org/project/pydiffsol/
- **Documentation**: https://pydiffsol.readthedocs.io/en/latest/

## Example usage

```py
import pydiffsol as ds
import numpy as np

# DiffSl code and matrix type specified in constructor
ode = ds.Ode(
    """
    in = [r]
    r { 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
    """,
    ds.nalgebra_dense
)

# Example overriding r input param with 2.0
params = np.array([2.0])
print(ode.solve(params, 0.4))

# Above defaults to bdf. Try esdirk34 instead
ode.method = ds.esdirk34
print(ode.solve(params, 0.4))
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
