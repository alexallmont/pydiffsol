# Contributing to pydiffsol

Contributions to `pydiffsol` are very welcome. You can contribute by reporting bugs, or helping with new features and documentation.

This guide is primarily written for developers. If you have any questions or problems getting set up, please get in touch so we can improve this page.

## Overview

`pydiffsol` is the Python-facing sibling of [`diffsol`](https://github.com/martinjrobins/diffsol). The core numerical work lives in Rust and this crate exposes a friendly Python API through PyO3. This guide intends to help you get set up, building and running code, and for developers an orientation and code style aid.

This `pydiffsol` repo is primarily:

- A Rust extension module built with PyO3 and built with `maturin`.
- A Python API layer that exposes diffsol functionality in a NumPy-friendly way.
- A Python test suite covering the public interface.
- Sphinx documentation and runnable Python examples.

The key technical hurdle in this library is that, being a Rust library, `diffsol` is statically-typed but Python is a dynamically-typed language, so `pydiffsol` builds preset static types and makes them available in a dynamic way. This is presented through the `Ode` class which combines the `diffsl` code to JIT-compile along with a `matrix_type` (e.g. nalgebra or faer matrix libraries) and `scalar_type` (f32, f64) for `diffsol`. After construction those three things are immutable, but other `diffsol` traits are left mutable at runtime for experimentation, in particular the ODE `linear_solver` and `method`.

Locally we build with `maturin` but in CI we use `cibuildwheels` which is better at resolving shared library issues.

As a rough map:

- `src/` contains the Rust extension code
- `test/` contains Python tests
- `examples/` contains simple example scripts
- `docs/` contains the Sphinx documentation
- `.vscode/` developer tooling

## Getting set up

### Prerequisites

- Rust toolchain via [`rustup`](https://rustup.rs/)
- Python 3.9+
- A virtual environment
- [`maturin`](https://www.maturin.rs/)
- LLVM if not using Cranelift as a backend, e.g. for LLVM 17
  - on macos use `brew install llvm@17 suite-sparse`
  - for debian-flavoured linux use `apt install llvm-17 libsuitesparse-dev`

### First time build

1. Go to [alexallmont/pydiffsol](https://github.com/alexallmont/pydiffsol) and click on the top right "fork" button to create your own copy of the project.
2. Clone locally with `git clone https://github.com/your-username/pydiffsol.git`
3. Create a virtual environment with `pip` or `uv`: `python3 -m venv .venv`
4. Activate the environment `source .venv/bin/activate`
5. Install maturin `pip install maturin`
6. Build for your platform, e.g. `maturin develop --extras dev --features diffsol-llvm17`

### Local development in VSCode

For VSCode developers, the `.vscode/` folder contains a `launch.json` for common debug scenarios, including debugging Python for stepping through tests and debugging Rust code that has been launched from tests. The latter is useful for drilling into Rust bugs found via `pydiffsol`. Scenarios are set up for launching both unit tests and example code.

When you launch one of the scenarios it will automatically build for you using the `maturin develop` task, which uses the default configuration for your platform as in the released wheel. If you have different versions of LLVM/Cranelift or features, edit `tasks.json` for your platform, e.g. to add `suitesparse` for osx add:

### Features

Optional dependencies depend on the backend you want to use:

- `diffsol-llvm*` to select LLVM version
- `diffsol-cranelift` for cranelift, mandatory if LLVM is not specified
- `suitesparse` enables sparse matrix support.

The current release wheel builds with LLVM 17 on macos and linux, with `suitesparse` enabled for linux. At present, Windows uses Cranelift as a backend instead of LLVM. See `.vscode/tasks.json` for examples of various features being enabled, e.g.

- `maturin develop --extras dev --features diffsol-llvm17 --features suitesparse`
- `maturin develop --extras dev --features diffsol-llvm17`
- `maturin develop --extras dev --features diffsol-cranelift`

If your machine uses a different LLVM version or local setup, adjust the feature flags accordingly.

## Typical workflow

For most changes, this is a good loop:

1. Build and install locally with `maturin develop`.
2. Run the relevant Python tests with `pytest`.
3. Run `cargo fmt`.
4. If you touched Rust logic substantially, run `cargo clippy` with correct feature flags.
5. Rebuild docs or examples if your change affects user-facing behaviour.

Useful commands:

```sh
pytest
cargo fmt
cargo clippy --features diffsol-llvm17
```

To build the docs into `docs/_build/html`:

```sh
cd docs
pip install -r requirements.txt
make html
```

If docs are not reflecting your local Rust changes, rerun `maturin develop` before rebuilding them.

## Project style

The most important rule is to optimise for readability at the Rust/Python boundary. A contributor who understands one side better than the other should still be able to follow the file.

### File-level guidance

- Start each source file with a short plain-English description saying what the file is for. This is to help both new readers and maintainers.
- Avoid code duplication, but at the same time don't overuse macros. An example of this trade-off is in methods like `solve` and `solve_dense` where there is a lot of repeated code, but we prefer not wrapping in macros to aid reading and debugging.
- All unsafe code must clearly be marked why it is unsafe and how the risk is mitigated.

### Python-facing API conventions

- Favour simple, unsurprising Python method names and behaviour.
- Preserve consistency with the existing API, even if the internal Rust implementation changes.
- Error messages should be understandable to a Python user, to a Rust developer.
- NumPy shape and dtype expectations should be clear in both code and tests.

### Tests and examples

- Add or update tests for user-visible behaviour changes.
- Prefer testing through the public Python API unless the change is purely internal.
- When a new feature is something users will likely copy, consider adding or updating an example as well as a test.

## Writing docs

User-facing changes should usually come with user-facing documentation updates.

That might mean updating:

- `README.md`
- `README_PyPI.md`
- `docs/`
- `examples/`

The docs are aimed at Python users first, so keep them practical and example-driven where possible.

## Scope of changes

Small, focused pull requests are easiest to review and maintain. In particular:

- Avoid mixing refactors with behavioural changes unless they are tightly related.
- If a change requires touching both Rust and Python-facing docs/tests, that is usually a good sign.
- If a bug really originates in `diffsol` rather than `pydiffsol`, it may be better fixed upstream first.

## Pull requests

When opening a pull request, please include:

- A short description of the problem being solved.
- A summary of the approach taken.
- Notes about feature flags, platform limitations, or optional dependencies if relevant.
- Tests or examples updated as part of the change.

If something is still exploratory or incomplete, that is fine too; just say so clearly in the PR description.

## If you are unsure

If you are not sure where a change belongs, or how much documentation or testing is appropriate, err on the side of a small draft PR with context. This is a compact codebase, and a short explanation of your intent is often enough to get a useful review moving.
