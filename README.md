# pydiffsol

Python bindings for [diffsol](https://github.com/martinjrobins/diffsol)

## Local development

Specify the `diffsol-llvm` version in `maturin develop`. Also add `dev` extras
for pytest along with plotly and pandas for docs image generation.

```sh
maturin develop --extras dev --features diffsol-llvm17
```

The included `.vscode` IDE config works with `diffsol-llvm17` by default. This
assumes that you have installed on macos with `brew install llvm@17` or for
debian-flavoured linux `apt install llvm-17`.

## Licenses

This wheel bundles `libunwind.1.dylib` from LLVM, licensed under the Apache 2.0
License with LLVM exceptions, and `libzstd.1.dylib` from the Zstandard project,
licensed under the BSD 3-Clause License. See `LICENSE.libunwind` and
`LICENSE.zstd` for details.
