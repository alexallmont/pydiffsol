Robertson ODE Benchmark
=======================

This is a benchmark on a stiff ODE system, using the classic test case, the Robertson (1966) problem,
which models the kinetics of an autocatalytic reaction.
This test case can be written as an ODE given by the following set of equations:

.. math::

   \begin{align}
   \frac{dx}{dt} &= -0.04x + 10^4 y z \\
   \frac{dy}{dt} &= 0.04x - 3 \cdot 10^7 y^2 - 10^4 y z \\
   \frac{dz}{dt} &= 3 \cdot 10^7 y^2 \\
   \end{align}

with initial conditions:

.. math::

   \begin{align}
   x(0) &= 1 \\
   y(0) &= 0 \\
   z(0) &= 0 \\
   \end{align}

This problem is known to be stiff due to the presence of widely varying timescales in the solution.

We can extend this problem to a larger system by creating multiple groups of the Robertson equations,
where each group is independent of the others. This allows us to benchmark the performance of different
ODE solvers as we increase the size of the system.

Solvers
-------

We benchmark the following solvers:

- Diffsol's BDF, ESDIRK34 & TR-BDF2 methods
- CasADi's CVODE solver
- Diffrax's Kvaerno5 & Tsit5 solvers
- Julia's DifferentialEquations.jl FBDF, KenCarp3, & TRBDF2 methods

The following solvers are similar so should be compared against each other:

- Diffsol BDF, CasADi CVODE & DifferentialEquations.jl FBDF methods.
- Diffsol's TR-BDF2 and DifferentialEquations.jl TRBDF2.
- Diffsol's ESDIRK34, DifferentialEquations.jl KenCarp3 and Diffrax's Tsit5 methods are different methods, but are all SDIRK implicit Runge-Kutta methods of similar order.

Benchmark Setup
---------------

For each solver, we perform as much initial setup as possible outside of the timing loop to ensure a fair comparison, using a function called `setup`.
The actual benchmark is performed in a function called `bench`.

The code for the Diffsol solvers is shown below:

.. literalinclude:: ../../bench/diffsol_robertson.py
   :encoding: latin-1
   :language: python

.. literalinclude:: ../../bench/diffsol_models.py
   :encoding: latin-1
   :language: python

Note that for `ngroup < 20`` it uses the nalgebra dense matrix and LU solver, and for `ngroups >= 20` the faer sparse matrix and LU solver are used.

The code for the CasADi solver is shown below:

.. literalinclude:: ../../bench/casadi_robertson.py
   :encoding: latin-1
   :language: python

.. literalinclude:: ../../bench/casadi_models.py
   :encoding: latin-1
   :language: python

The code for the Diffrax solver is shown below:

.. literalinclude:: ../../bench/diffrax_robertson.py
   :encoding: latin-1
   :language: python

.. literalinclude:: ../../bench/diffrax_models.py
   :encoding: latin-1
   :language: python

The code for the DifferentialEquations.jl solvers is shown below:

.. literalinclude:: ../../bench/diffeq_robertson.jl
   :language: julia

.. literalinclude:: ../../bench/diffeq_models.jl
   :language: julia

Differences between implementations
-----------------------------------

There are a few key differences between the Diffrax, Casadi, Diffsol and DifferentialEquations.jl implementations that may affect the performance of the solvers. The main differences are:

- **Sparse vs Dense matrices:** The Casadi implementation uses sparse matrices, whereas the Diffsol and DifferentialEquations.jl implementations use dense matrices for ngroups < 20,
  and sparse matrices for ngroups >= 20. This will provide an advantage for Diffsol for smaller problems. The Diffrax implementation uses dense matrices.
  Treating the Jacobian as dense will be a disadvantage for Diffrax for larger problems as the Jacobian is very sparse for larger problem sizes.
- **Multithreading**: For the Macbook M2 Pro run, each library was free to use multiple threads according to their default settings.
  For the rack server, each library was limited to using 20 threads (using `RAYON_NUM_THREADS=20 OMP_NUM_THREADS=20 JULIA_NUM_THREADS=20`).
  The only part of the Diffsol implementation that takes advantage of multiple threads is the faer sparse LU solver and matrix.
  Both the nalgebra LU solver, matrix, and the DiffSL generated code are all single-threaded.
  Diffrax uses JAX, which takes advantage of multiple threads (CPU only, no GPUs were used in these benchmarks). Casadi uses multithreading via OpenMP and the Sundials solver.
  It is unclear if DifferentialEquations.jl uses multithreading for single ODE runs, although it supports use multiple threads for ensemble runs.

Results
-------

The benchmarks were run on:
   - A Dell PowerEdge R7525 2U rack server, with dual AMD EPYC 7343 3.2Ghz 16C CPU and 128GB Memory
   - A Macbook M2 Pro (14-inch, 2023) with 16GB Memory, 12 (8 performance and 4 efficiency) cores

The results are shown below:

.. image:: ../images/benchmark_robertson_ode_rack_server.svg
  :width: 800
  :height: 600
  :alt: benchmark_robertson_ode_rack_server.svg

.. image:: ../images/benchmark_robertson_ode_macbook.svg
  :width: 800
  :height: 600
  :alt: benchmark_robertson_ode_macbook.svg

The Diffsol implementation outperforms the other implementations significantly for smaller problem sizes, especially when using the rack server.
At these small problem sizes, the dense matrix and solver used by Diffsol provide an advantage over the sparse solver used by Casadi.
Casadi also has additional overhead to evaluate each function evaluation, as it needs to traverse a graph of operations to calculate each rhs or jacobian evaluation,
whereas the DiffSL JIT compiler will compile to native code using the LLVM backend, along with low-level optimisations that are not available to Casadi.
Diffrax is also significantly slower than Diffsol for smaller problems, this might be due to
(a) Diffrax being a ML library and not optimised for solving stiff ODEs,
or (b) double precision is used, which again is not a common use case for ML libraries.

As the problem sizes get larger the dense solver used by Diffrax becomes less efficient and the runtime starts to diverge from the other methods
The performance of Casadi improve rapidly relative to Diffsol as the problem size increases, and for `n > 256` it becomes faster than the Diffsol BDF method for the rack server.
For the macbook the Casadi solver never becomes faster than the Diffsol BDF method, instead the two methods converge in performance. This is likely due to the better multi-threading
performance of the CVODE solver used by Casadi on the rack server, which has more CPU cores available.

The DifferentialEquations.jl implementations are slower than the Diffsol impolementation across all problem sizes, and slower than Casadi at larger problem sizes.
Anthough the DifferentialEquations.jl FBDF method is faster than Casadi for smaller problems.
