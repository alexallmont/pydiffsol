# pydiffsol

Python bindings for [diffsol](https://github.com/martinjrobins/diffsol)

## Local development

Specify the `diffsol-llvm` version in `maturin develop` so along with the `dev`
extras set to allow debugging and testing locally.

```sh
maturin develop --extras dev --features diffsol-llvm17
```

The included `.vscode` IDE config works with `diffsol-llvm17` by default.