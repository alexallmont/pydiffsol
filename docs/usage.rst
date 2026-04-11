Usage
=====

.. _installation:

Installation
------------

To use pydiffsol, install using pip:

.. code-block:: console

   (.venv) $ pip install pydiffsol

Basic Usage
-----------

.. code-block:: python

   import pydiffsol as ds
   import numpy as np

   def select_jit_backend():
      backend = ds.default_enabled_jit_backend()
      if backend is not None:
         return backend
      if hasattr(ds, "cranelift"):
         return ds.cranelift
      if hasattr(ds, "llvm"):
         return ds.llvm
      raise RuntimeError("No JIT backend available")

   ode = ds.Ode(
      """
      in { r = 1.0 }
      k { 1.0 }
      u { 0.1 }
      F { r * u * (1.0 - u / k) }
      """,
      jit_backend=select_jit_backend(),
      matrix_type=ds.nalgebra_dense,
   )

   # Solve up to t = 0.4, overriding r input param = 2.0
   params = np.array([2.0])
   solution = ode.solve(params, 0.4)
   print(solution.ys, solution.ts)

   # Above defaults to bdf. Try esdirk34 instead
   ode.ode_solver = ds.esdirk34
   solution = ode.solve(params, 0.4)
   print(solution.ys, solution.ts)
