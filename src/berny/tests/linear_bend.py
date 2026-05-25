# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Linear-bend / dihedral-crossover model potential."""

import math
from typing import ClassVar

import numpy as np
from numpy.linalg import norm

from .base import ModelPotential, _angle, _cos_with_grads


class LinearBendCrossover(ModelPotential):
    """A 4-atom chain whose minimum is linear ``a-b-c`` with a bent terminal ``d``.

    At the minimum ``a-b-c`` is collinear (an sp-like centre at ``b``) and ``d``
    is bent off ``c`` at ``127 deg``. The :meth:`start` geometry has *both* the
    ``a-b-c`` and ``b-c-d`` angles well away from linear, so an optimizer using
    redundant internal coordinates has to cross the linear-bend threshold on the
    way in -- exercising the dummy-atom / dihedral handoff.

    The torsion about the ``b-c`` axis is a free (zero-curvature) mode at the
    minimum, so :meth:`assert_at_minimum` checks only the bond lengths and the
    two angles, which are invariant under rigid motion and that soft rotation.
    """

    species: ClassVar[list[str]] = ['C', 'C', 'C', 'C']
    r_ab = 1.20
    r_bc = 1.30
    r_cd = 1.30
    theta_bcd = math.radians(127.0)
    k_bond = 0.6
    k_linear = 0.25
    k_bend = 0.25

    def start(self):
        """Return a starting geometry with both bond angles far from linear."""
        a = np.zeros(3)
        b = a + np.array([self.r_ab, 0.0, 0.0])
        phi = math.acos(-math.cos(math.radians(150.0)))
        bc = np.array([math.cos(phi), math.sin(phi), 0.0])
        c = b + 1.34 * bc
        cb = (b - c) / norm(b - c)
        perp = np.cross(cb, np.array([0.0, 0.0, 1.0]))
        perp /= norm(perp)
        th = math.radians(110.0)
        cd = math.cos(th) * cb + math.sin(th) * perp
        d = c + 1.28 * cd
        return np.array([a, b, c, d])

    def energy(self, coords):
        """Return the model energy at ``coords`` (``(4, 3)`` array, angstrom)."""
        a, b, c, d = np.asarray(coords, dtype=float)
        e = (
            0.5
            * self.k_bond
            * (
                (norm(b - a) - self.r_ab) ** 2
                + (norm(c - b) - self.r_bc) ** 2
                + (norm(d - c) - self.r_cd) ** 2
            )
        )
        e += self.k_linear * (1.0 + _cos_with_grads(a, b, c)[0])
        e += 0.5 * self.k_bend * (_angle(b, c, d) - self.theta_bcd) ** 2
        return float(e)

    def gradient(self, coords):
        """Return the analytic gradient ``dE/dr`` (``(4, 3)``, per angstrom)."""
        coords = np.asarray(coords, dtype=float)
        a, b, c, d = coords
        g = np.zeros((4, 3))
        for i, j, r0 in [(0, 1, self.r_ab), (1, 2, self.r_bc), (2, 3, self.r_cd)]:
            v = coords[j] - coords[i]
            r = norm(v)
            f = self.k_bond * (r - r0) * v / r
            g[j] += f
            g[i] -= f
        _, dp, dv, dq = _cos_with_grads(a, b, c)
        g[0] += self.k_linear * dp
        g[1] += self.k_linear * dv
        g[2] += self.k_linear * dq
        cos2, dp2, dv2, dq2 = _cos_with_grads(b, c, d)
        sin = math.sqrt(max(1.0 - cos2**2, 1e-12))
        theta = math.acos(min(1.0, max(-1.0, cos2)))
        coef = self.k_bend * (theta - self.theta_bcd) * (-1.0 / sin)
        g[1] += coef * dp2
        g[2] += coef * dv2
        g[3] += coef * dq2
        return g

    def assert_at_minimum(self, coords, tol_dist=1e-2, tol_angle_deg=1.0):
        """Assert ``coords`` match the known minimum (bond lengths and angles)."""
        coords = np.asarray(coords, dtype=float)
        a, b, c, d = coords
        residuals = {
            'r_ab': norm(b - a) - self.r_ab,
            'r_bc': norm(c - b) - self.r_bc,
            'r_cd': norm(d - c) - self.r_cd,
            'angle_abc_deg': math.degrees(_angle(a, b, c)) - 180.0,
            'angle_bcd_deg': (
                math.degrees(_angle(b, c, d)) - math.degrees(self.theta_bcd)
            ),
        }
        bad = []
        for key, val in residuals.items():
            tol = tol_angle_deg if key.startswith('angle') else tol_dist
            if abs(val) > tol:
                bad.append(f'{key}: residual {val:+.3g} exceeds tol {tol:g}')
        assert not bad, 'geometry not at minimum:\n  ' + '\n  '.join(bad)
