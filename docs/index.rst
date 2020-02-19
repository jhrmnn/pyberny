``berny`` — Molecular optimizer
===============================

This Python package can optimize molecular and crystal structures
with respect to total energy, using nuclear gradient information.

In each step, it takes energy and Cartesian gradients as an input, and
returns a new structure estimate.

The algorithm is an amalgam of several techniques, comprising redundant
internal coordinates, iterative Hessian estimate, trust region, line
search, and coordinate weighing, mostly inspired by the optimizer in the
`Gaussian <http://gaussian.com>`_ program.

.. toctree::

   getting-started
   algorithm
   api
