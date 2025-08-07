Usage
=====

.. _installation:

Installation
------------

To use pydiffsol, install using pip:

.. code-block:: console

   (.venv) $ pip install pydiffsol

Minimal example
---------------

.. code-block:: python

   import pydiffsol as ds
   import numpy as np

   ode = ds.Ode(
      """
      r { 1.0 }
      k { 1.0 }
      u { 0.1 }
      F { r * u * (1.0 - u / k) }
      """,
      ds.nalgebra_dense_f64
   )
   p = np.array([])
   print(ode.solve(p, 0.4))
