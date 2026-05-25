# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Framework shared by the :mod:`berny.tests` model potentials.

Defines the :class:`ModelPotential` interface that each potential implements
and the :func:`run_and_check` driver that exercises an optimizer against one.
"""

import math
from typing import ClassVar

import numpy as np
from numpy import dot
from numpy.linalg import norm


def _cos_with_grads(p, v, q):
    # Cosine of the angle at vertex ``v`` (arms to ``p`` and ``q``) and its
    # derivatives w.r.t. each of the three points.
    u = p - v
    w = q - v
    nu = norm(u)
    nw = norm(w)
    cos = dot(u, w) / (nu * nw)
    dcos_dp = w / (nu * nw) - cos * u / nu**2
    dcos_dq = u / (nu * nw) - cos * w / nw**2
    dcos_dv = -(dcos_dp + dcos_dq)
    return cos, dcos_dp, dcos_dv, dcos_dq


def _angle(p, v, q):
    u = p - v
    w = q - v
    cos = dot(u, w) / (norm(u) * norm(w))
    return math.acos(min(1.0, max(-1.0, cos)))


class ModelPotential:
    """Base class for analytic model potentials with a known minimum.

    Subclasses set :attr:`species` and implement :meth:`start`,
    :meth:`energy`, :meth:`gradient` and :meth:`assert_at_minimum`. Energies
    are in arbitrary but consistent units and coordinates in angstrom; the
    gradient is ``dE/dr`` in those units per angstrom.
    """

    species: ClassVar[list[str]]

    def start(self):
        """Return the starting geometry as an ``(N, 3)`` array in angstrom."""
        raise NotImplementedError

    def energy(self, coords):
        """Return the potential energy at ``coords`` (``(N, 3)``, angstrom)."""
        raise NotImplementedError

    def gradient(self, coords):
        """Return the analytic gradient ``dE/dr`` at ``coords`` (per angstrom)."""
        raise NotImplementedError

    def assert_at_minimum(self, coords, **tols):
        """Assert that ``coords`` sit at the known minimum of this potential."""
        raise NotImplementedError

    def numerical_gradient(self, coords, step=1e-5):
        """Return a central-difference gradient, to cross-check :meth:`gradient`."""
        coords = np.asarray(coords, dtype=float)
        g = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            for j in range(3):
                up = coords.copy()
                up[i, j] += step
                dn = coords.copy()
                dn[i, j] -= step
                g[i, j] = (self.energy(up) - self.energy(dn)) / (2 * step)
        return g


def run_and_check(potential, minimize, **tols):
    """Run ``minimize`` against ``potential`` and assert it reaches the minimum.

    ``minimize`` is the adapter for the optimizer under test, called as
    ``minimize(species, start_coords, energy, gradient)`` and expected to
    return the converged geometry as an ``(N, 3)`` array in angstrom. Extra
    keyword arguments are forwarded to
    :meth:`ModelPotential.assert_at_minimum` as tolerances.
    """
    species = list(potential.species)
    expected = (len(species), 3)
    final = minimize(species, potential.start(), potential.energy, potential.gradient)
    try:
        final = np.asarray(final, dtype=float)
    except (TypeError, ValueError) as e:
        raise AssertionError(
            f'minimize must return a float-convertible array, got {final!r}'
        ) from e
    assert (
        final.shape == expected
    ), f'minimize must return coordinates of shape {expected}, got {final.shape}'
    potential.assert_at_minimum(final, **tols)
    return final
