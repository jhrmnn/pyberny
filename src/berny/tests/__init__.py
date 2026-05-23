# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Reusable end-to-end optimizer tests built on analytic model potentials.

This subpackage ships with :mod:`berny` so that *any* geometry optimizer --
not only :class:`berny.Berny` -- can be exercised against model potential
energy surfaces whose minima are known in closed form. Each potential exposes

* :meth:`~base.ModelPotential.start` -- a deliberately awkward starting
  geometry,
* :meth:`~base.ModelPotential.energy` / :meth:`~base.ModelPotential.gradient`
  -- the energy surface and its analytic gradient (energy in arbitrary but
  consistent units, length in angstrom, gradient per angstrom),
* :meth:`~base.ModelPotential.assert_at_minimum` -- a check that a converged
  geometry sits at the known minimum, up to rigid motion and any soft
  (zero-curvature) modes.

The framework lives in :mod:`berny.tests.base`; each model potential has its
own module (e.g. :mod:`berny.tests.linear_bend`). All public names are
re-exported here.

A third-party optimizer is tested by supplying a ``minimize`` adapter::

    from berny.tests import LinearBendCrossover, run_and_check

    def minimize(species, coords, energy, gradient):
        # drive your optimizer to convergence and return the final
        # (N, 3) coordinates in angstrom
        ...

    run_and_check(LinearBendCrossover(), minimize)

``run_and_check`` raises :class:`AssertionError` if the optimizer fails to
reach the minimum, so it drops straight into any test runner.
"""

from .base import ModelPotential, run_and_check
from .linear_bend import LinearBendCrossover

__all__ = ['ModelPotential', 'LinearBendCrossover', 'run_and_check']
