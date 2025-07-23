# pydiffsol

Python bindings for [diffsol](https://github.com/martinjrobins/diffsol)

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

## Local development

To build locally, use [maturin](https://www.maturin.rs/installation.html) and
set `diffsol-llvm` feature to your installed LLVM. Also specify `dev` extras for
pytest, running examples and docs image generation. For example:

```sh
maturin develop --extras dev --features diffsol-llvm17
```

The included `.vscode` include examples for running tests and examples in
python and rust debuggers. The config works with `diffsol-llvm17` by default and
assumes that you have it already installed, for example on macos with
`brew install llvm@17` or for debian-flavoured linux `apt install llvm-17`.

## Licenses

This wheel bundles `libunwind.1.dylib` from LLVM, licensed under the Apache 2.0
License with LLVM exceptions, and `libzstd.1.dylib` from the Zstandard project,
licensed under the BSD 3-Clause License. See `LICENSE.libunwind` and
`LICENSE.zstd` for details.
