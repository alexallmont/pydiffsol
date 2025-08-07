pydiffsol documentation
=======================

`Diffsol <https://github.com/martinjrobins/diffsol/>`_ is a library for solving
ordinary differential equations (ODEs) or semi-explicit differential algebraic
equations (DAEs) in Rust. It can solve equations in the following form:

.. math:: M \frac{dy}{dt} = f(t, y, p)

where \\(M\\) is a (possibly singular and optional) mass matrix, \\(y\\) is the
state vector, \\(t\\) is the time and \\(p\\) is a vector of parameters.

The equations are specified using the `DiffSL <https://martinjrobins.github.io/diffsl/>`_
Domain Specific Language (DSL) which uses automatic differentiation to calculate
the necessary jacobians, and JIT compilation (using either LLVM or Cranelift) to
generate efficient native code at runtime. The DSL allows Diffsol to be used in
higher-level languages like Python while maintaining similar performance to
pure rust.

Contents
--------

.. toctree::

   Home <self>
   usage
   primer
   api

* :ref:`genindex`

* :ref:`search`
