Example: Bouncing Ball
======================

A bouncing ball is a simple hybrid system with continuous dynamics between
impacts and a discrete state update at each impact.

The continuous model is:

.. math::

  \frac{d^2x}{dt^2} = -g

where \\(x\\) is height and \\(g\\) is gravitational acceleration. Introducing
velocity \\(v = dx/dt\\), we write:

.. math::

  \frac{dx}{dt} = v

.. math::

  \frac{dv}{dt} = -g

with initial conditions \\(x(0)=h\\), \\(v(0)=0\\).

At each ground contact, we would ideally apply a discrete reset:

.. math::

  v^+ = -e v^-

where \\(e\\) is the coefficient of restitution.

In DiffSL we define an event (root) function ``stop { x }`` so integration
halts when \\(x = 0\\). This Python example therefore plots the trajectory up to
the first impact using the current ``diffsol-c``-aligned API surface.

.. literalinclude:: ../../examples/3_2_bouncing_ball.py
  :encoding: latin-1
  :language: python

.. image:: ../images/bouncing_ball.svg
  :width: 640
  :height: 480
  :alt: bouncing_ball.svg
