PyBerny Documentation
=====================

This Python package can optimize molecular structures (with
experimental support for crystals) with respect to total energy, using
nuclear gradient information.

In each step, it takes energy and Cartesian gradients as an input, and
returns a new structure estimate.

The algorithm is an amalgam of several techniques, comprising redundant
internal coordinates, iterative Hessian estimate, trust region, line
search, and coordinate weighting, mostly inspired by the optimizer in
the `Gaussian <http://gaussian.com>`_ program.

.. toctree::

   getting-started
   algorithm
   standard_method
   api
