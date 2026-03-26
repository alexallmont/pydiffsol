# Contributing to pydiffsol

Contributions to `pydiffsol` are welcome, whether that means bug reports, features, tests, or documentation improvements.

`pydiffsol` is the Python-facing sibling of [`diffsol`](https://github.com/martinjrobins/diffsol). The numerical work lives in Rust and this repository wraps it in a Python API.

## Overview

The main design challenge in `pydiffsol` is bridging Rust's static types with Python's dynamic API. In practice, `pydiffsol` builds preset combinations of static `matrix_type`, `scalar_type`, `linear_solver` and `method` types, and provides a friendly mechanism to select them dynamically and query results with a numpy interface.

## Getting set up

You will need:

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

For VSCode developers, `.vscode/` contains a `launch.json` for common run scenarios, including debugging Python or Rust code. The latter is useful for drilling into Rust bugs found via `pydiffsol`. Scenarios are set up for launching both unit tests and example code.

When you launch one of the scenarios it will automatically build for you using the `maturin develop` task, which uses the default configuration for your platform as in the released wheel.

## Typical workflow

For most changes:

1. Implement feature/bug code changes
1. Build locally with `maturin develop`.
1. Run the relevant tests with `pytest`.
1. Format with `cargo fmt`.
1. If needed, run `cargo clippy` with the same feature flags as your build.
1. Rebuild docs or examples if the change is user-facing.
1. Update `CHANGELOG.md`

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

## Project management

Releases are drafted in [issues](https://github.com/alexallmont/pydiffsol/issues) and delivered by [milestone](https://github.com/alexallmont/pydiffsol/milestones). The [project board](https://github.com/users/alexallmont/projects/3/views/2) summarises work progress.

## Releases

To release v0.x.y, e.g. x and y from next unreleased draft [issue](https://github.com/alexallmont/pydiffsol/issues):

1. Ensure all PRs are merged to `main` and there are no build issues.
1. Ensure `Cargo.toml` is correct, e.g. `version = "0.x.y"`.
1. Ensure `CHANGELOG.md` details latest changes up to v0.x.y.
1. Edit `docs/requirements.txt` and change `pydiffsol==0.x.y` to the latest version.
1. Go to the [releases](https://github.com/alexallmont/pydiffsol/releases), click **Draft a new release**.
1. Click on 'Select tag' then 'Create new tag'.
1. Set the tag and release name to v0.x.y, note the leading 'v'.
1. Click 'Generate release notes' to get a changes digest and diff range.
1. Replace the text under `## What's Changed` with latest `CHANGELOG.md` range. Keep the `**Full Changelog**` at bottom.
1. Click 'Publish release'
1. Check build action completes and the new package is on [PyPI](https://pypi.org/project/pydiffsol/).
1. Log in to [readthedocs](https://app.readthedocs.org/dashboard/), rebuild `latest`, `stable` and add a new version for `v0.x.y`

The final step - to manually rebuild readthedocs - is needed because the docs are built through introspection and require the latest PyPI version of pydiffsol, but readthedocs is triggered by watching GitHub and not PyPI, so presently there is a bug where the docs try to build before the package is available.
