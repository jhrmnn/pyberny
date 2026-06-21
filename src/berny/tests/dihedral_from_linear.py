# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Reverse-direction model potential: a bent dihedral reached from linear."""

import math
from typing import Any, ClassVar

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray

from .base import ModelPotential, _angle, _cos_with_grads

FloatArray = NDArray[np.floating[Any]]


class DihedralFromLinear(ModelPotential):
    """A 4-atom chain whose minimum is an ordinary bent dihedral, started linear.

    This is the mirror image of :class:`~berny.tests.linear_bend.LinearBendCrossover`.
    The minimum is a perfectly ordinary geometry -- ``a-b-c`` bent at ``120 deg``
    and ``b-c-d`` at ``113 deg`` -- so an optimizer on redundant internals
    describes it with a genuine ``a-b-c-d`` dihedral and *no* dummy atoms. The
    :meth:`start` geometry, however, has ``a-b-c`` near-linear (``178 deg``,
    above the 175 deg threshold), so the optimizer *begins* with two dummy bends
    and no dihedral, and must drop below the 170 deg exit threshold on the way
    in -- forcing the dummy-to-dihedral handoff in the opposite direction.

    Both bends use a ``(cos theta - cos theta0)**2`` form rather than an
    angle-harmonic, so the energy and its gradient stay smooth through the
    near-linear start (no ``1 / sin theta`` singularity at 180 deg). The torsion
    about ``b-c`` is a free mode at the minimum, so :meth:`assert_at_minimum`
    checks only the bond lengths and the two angles.
    """

    species: ClassVar[list[str]] = ['C', 'C', 'C', 'C']
    r_ab = 1.45
    r_bc = 1.50
    r_cd = 1.45
    theta_abc = math.radians(120.0)
    theta_bcd = math.radians(113.0)
    k_bond = 0.6
    k_bend = 0.25

    @property
    def _bends(self) -> list[tuple[int, int, int, float]]:
        return [
            (0, 1, 2, math.cos(self.theta_abc)),
            (1, 2, 3, math.cos(self.theta_bcd)),
        ]

    def start(self) -> FloatArray:
        """Return a starting geometry with ``a-b-c`` near-linear (178 deg)."""
        a = np.zeros(3)
        b = a + np.array([self.r_ab, 0.0, 0.0])
        phi = math.radians(180.0 - 178.0)
        c = b + self.r_bc * np.array([math.cos(phi), math.sin(phi), 0.0])
        cb = (b - c) / norm(b - c)
        perp = np.cross(cb, np.array([0.0, 0.0, 1.0]))
        perp /= norm(perp)
        th = math.radians(100.0)
        cd = math.cos(th) * cb + math.sin(th) * perp
        d = c + self.r_cd * cd
        return np.array([a, b, c, d])

    def energy(self, coords: FloatArray) -> float:
        """Return the model energy at ``coords`` (``(4, 3)`` array, angstrom)."""
        coords = np.asarray(coords, dtype=float)
        e = 0.0
        for i, j, r0 in [(0, 1, self.r_ab), (1, 2, self.r_bc), (2, 3, self.r_cd)]:
            e += 0.5 * self.k_bond * float(norm(coords[j] - coords[i]) - r0) ** 2
        for i, j, k, cos0 in self._bends:
            cos = _cos_with_grads(coords[i], coords[j], coords[k])[0]
            e += 0.5 * self.k_bend * (cos - cos0) ** 2
        return float(e)

    def gradient(self, coords: FloatArray) -> FloatArray:
        """Return the analytic gradient ``dE/dr`` (``(4, 3)``, per angstrom)."""
        coords = np.asarray(coords, dtype=float)
        g = np.zeros((4, 3))
        for i, j, r0 in [(0, 1, self.r_ab), (1, 2, self.r_bc), (2, 3, self.r_cd)]:
            v = coords[j] - coords[i]
            r = norm(v)
            f = self.k_bond * (r - r0) * v / r
            g[j] += f
            g[i] -= f
        for i, j, k, cos0 in self._bends:
            cos, dp, dv, dq = _cos_with_grads(coords[i], coords[j], coords[k])
            coef = self.k_bend * (cos - cos0)
            g[i] += coef * dp
            g[j] += coef * dv
            g[k] += coef * dq
        return g

    def assert_at_minimum(  # type: ignore[override]
        self, coords: FloatArray, tol_dist: float = 1e-2, tol_angle_deg: float = 1.0
    ) -> None:
        """Assert ``coords`` match the known minimum (bond lengths and angles)."""
        coords = np.asarray(coords, dtype=float)
        a, b, c, d = coords
        residuals: dict[str, float] = {
            'r_ab': float(norm(b - a)) - self.r_ab,
            'r_bc': float(norm(c - b)) - self.r_bc,
            'r_cd': float(norm(d - c)) - self.r_cd,
            'angle_abc_deg': math.degrees(_angle(a, b, c) - self.theta_abc),
            'angle_bcd_deg': math.degrees(_angle(b, c, d) - self.theta_bcd),
        }
        bad = []
        for key, val in residuals.items():
            tol = tol_angle_deg if key.startswith('angle') else tol_dist
            if abs(val) > tol:
                bad.append(f'{key}: residual {val:+.3g} exceeds tol {tol:g}')
        assert not bad, 'geometry not at minimum:\n  ' + '\n  '.join(bad)
