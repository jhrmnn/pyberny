API
===

This covers all supported public API.

.. module:: berny

.. autofunction:: Berny

.. autodata:: berny.core.defaults

.. autodata:: berny.coords.bohr

    Can be imported directly as :py:data:`berny.bohr`.

.. autofunction:: optimize

.. autoclass:: Logger
   :members: __call__

Geometry operations
-------------------

.. autoclass:: Molecule
   :members:

.. module:: berny.geomlib

.. autoclass:: Crystal
   :members:

.. autofunction:: load

.. autofunction:: loads

.. autofunction:: readfile

Solvers
-------

All functions in this module are coroutines that satisfy the solver interface
expected by :py:func:`~berny.optimize`.

.. module:: berny.solvers

.. autofunction:: MopacSolver
