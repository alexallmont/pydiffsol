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