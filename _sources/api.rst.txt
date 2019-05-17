API
===

This covers all supported public API.

.. module:: berny

.. autoclass:: Berny
   :members:
   :exclude-members: Point, send, throw

.. autodata:: berny.berny.defaults

.. autodata:: berny.coords.angstrom

    Can be imported directly as :py:data:`berny.angstrom`.

.. autofunction:: optimize

.. autoclass:: Logger
   :members:

Geometry operations
-------------------

.. autoclass:: Geometry
   :members:

.. module:: berny.geomlib

.. autofunction:: load

.. autofunction:: loads

.. autofunction:: readfile

Solvers
-------

All functions in this module are coroutines that satisfy the solver interface
expected by :py:func:`~berny.optimize`.

.. module:: berny.solvers

.. autofunction:: MopacSolver
