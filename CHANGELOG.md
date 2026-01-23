# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1]

### Added

- Get/set initial conditions, solver options, rhs and y0 (#30)
- Electrical circuits example (#39)
- Heat equation example (#36)
- Update to diffsol 0.10.2 (#41)

## [0.3.0]

### Added

- 32-bit float support (#27)
- Robertson_ode benchmark (#16)
- Forward sensitivity and adjoint sum of squares solves (#21)
- Generalise benchmarks for different models and add lotka-volterra benchmark (#19)

## [0.2.0]

### Added

- Wheels built using `cibuildwheel` rather than maturin for neater build matrix and resolving dylib issues.
- Config class removed. This is now rolled into Ode class for friendlier interface.
- Support for `tr_bdf2` and `tsit45` solver methods.
- Build with SuiteSparse for KLU support for Linux. Use `is_klu_available` method to check if available on running platform.
- Support for `faer_dense_f64` matrix_type if KLU available.
- `solve_dense` method to return solution at specified times.
- Runtime reflection utilities for `SolverMethod`, `MatrixType` and `MatrixType`: get all enums with `all`; `__str__` and `__hash__` dunder methods.
- `default` solver type automatically selects `lu` or `klu` based on matrix type, except for `tsit45` which does not need solver type.
- `solver_for_matrix_type` config method to check how default will be resolved at point of solve call.
- Get original `code` specified in Ode when constructed.
- Spring mass systems example from original Diffsol ported to Python.

### Fixed

- Ensure that problem is compiled when Ode is constructed, not on every call to solve.
- SuiteSparse added to build so `fear_sparse_f64` now working correctly (except for BDF on macos, known issue internal to diffsol)

## [0.1.3]

### Added

- Build wheel for macos and Linux (LLVM backend), and Windows (CraneLift backend).
- `bdf` and `esdirk34` solvers.
- `nalgebra_dense_f64` matrix type and in-progress `faer_sparse_f64` matrix type.
- Population dynamics example from original Diffsol ported to Python.
