Example: Heat Equation
======================

We will now consider the heat equation as an example for a simple PDE. The heat equation is a PDE that describes how the temperature of a material changes over time. In one dimension, the heat equation is

\\[
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}
\\]

where \\(u(x, t)\\) is the temperature of the material at position \\(x\\) and time \\(t\\), and \\(D\\) is the thermal diffusivity of the material. Please have a look at https://martinjrobins.github.io/diffsol/primer/heat_equation.html for the derivation of the diffsol model.

We can solve the one-dimensional heat equation using Diffsol with the following code:

.. literalinclude:: ../../examples/03_heat_equation.py
   :encoding: latin-1
   :lines: 4,6-9,11-33
   :language: python

The plot of the solution is shown below:
.. image:: ../images/spring_mass_system.svg
  :width: 640
  :height: 480
  :alt: spring_mass_system.svg
