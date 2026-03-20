# Contributing to pydiffsol

Contributions to `pydiffsol` are very welcome, whether that means bug reports, features, tests, or documentation improvements.

`pydiffsol` is the Python-facing sibling of [`diffsol`](https://github.com/martinjrobins/diffsol). The numerical work lives in Rust; this repository wraps it in a Python API using PyO3 and packages it with `maturin`.

## Overview

- `src/` Rust extension code
- `test/` Python tests
- `examples/` example scripts
- `docs/` Sphinx docs
- `.vscode/` local developer tooling

One of the main design challenges is bridging Rust's static types with Python's dynamic API. In practice, that means much of the codebase is wrapper, conversion, and API-shaping code rather than solver logic itself. Effectively, `pydiffsol` builds preset combinations of static `matrix_type`, `scalar_type`, `linear_solver` and `method` types, and provides a friendly mechanism to select them dynamically.

## Getting set up

You will usually want:

- Rust via [`rustup`](https://rustup.rs/)
- Python 3.9+
- a virtual environment
- [`maturin`](https://www.maturin.rs/)

Depending on the backend, you may also need LLVM and optionally SuiteSparse. Typical local builds are:

```sh
maturin develop --extras dev --features diffsol-llvm17
maturin develop --extras dev --features diffsol-llvm17 --features suitesparse
maturin develop --extras dev --features diffsol-cranelift
```

For VSCode developers, `.vscode/` contains a `launch.json` for common debug scenarios, including debugging Python for stepping through tests and debugging Rust code that has been launched from tests. The latter is useful for drilling into Rust bugs found via `pydiffsol`. Scenarios are set up for launching both unit tests and example code.

When you launch one of the scenarios it will automatically build for you using the `maturin develop` task, which uses the default configuration for your platform as in the released wheel.

## Typical workflow

For most changes:

1. Build locally with `maturin develop`.
2. Run the relevant tests with `pytest`.
3. Format with `cargo fmt`.
4. If needed, run `cargo clippy` with the same feature flags as your build.
5. Rebuild docs or examples if the change is user-facing.

Useful commands:

```sh
pytest
cargo fmt
cargo clippy --features diffsol-llvm17
```

To build the docs locally:

```sh
cd docs
pip install -r requirements.txt
make html
```

If the docs are not showing local Rust changes, rerun `maturin develop` first.

## Style

The main rule is to optimise for readability at the Rust/Python boundary.

- Start each source file with a short plain-English comment stating it's purpose.
- Prefer clear, direct code over heavy abstraction.
- Avoid duplication where it helps, but do not hide important control flow behind macros.
- Mark any unsafe code clearly and explain why it is sound.
- Keep Python-facing names, errors, shapes, and dtypes unsurprising.

## Tests and docs

- Add or update tests for user-visible behaviour changes.
- Prefer testing through the public Python API unless the change is purely internal.
- Update docs, examples, or README files when behaviour visible to users changes.

## Scope

Small, focused pull requests are easiest to review.

- Avoid mixing refactors with behaviour changes unless they are tightly related.
- If a bug belongs in `diffsol` rather than `pydiffsol`, it may be better fixed upstream first.
- If you are unsure, open a small draft PR with context.

## Pull requests

Please include:

- the problem being solved
- the approach taken
- any relevant feature flags or platform notes
- tests or examples updated as part of the change
