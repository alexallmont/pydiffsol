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

   # DiffSl code and matrix type specified in constructor
   # Defaults to f64 BDF solver unless specified
   ode = ds.Ode(
      """
      in_i { r = 1.0 }
      k_i { 1.0 }
      u_i { 0.1 }
      F_i { r * u * (1.0 - u / k) }
      """,
      ds.nalgebra_dense
   )

   # Solve up to t = 0.4, overriding r input param = 2.0
   params = np.array([2.0])
   solution = ode.solve(params, 0.4)
   print(solution.ys, solution.ts)

   # Above defaults to bdf. Try esdirk34 instead
   ode.method = ds.esdirk34
   solution = ode.solve(params, 0.4)
   print(solution.ys, solution.ts)
