# pydiffsol

Python bindings for [diffsol](https://github.com/martinjrobins/diffsol)

## Local development

Specify the `diffsl-llvm` version in `maturin develop` so along with the `dev`
extras set to allow debugging and testing locally.

```sh
maturin develop --extras dev --features diffsol/diffsl-llvm16
```

The included `.vscode` IDE config works with `diffsl-llvm16` by default.