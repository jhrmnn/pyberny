# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Point-group detection and symmetry-breaking of start geometries.

A gradient-following optimizer cannot leave the symmetric subspace of an exactly
symmetric start geometry: the gradient along every non-totally-symmetric mode is
zero by symmetry, so the run can converge to a symmetric *saddle* rather than a
minimum (issue #148). This module detects the point group and can apply a small,
deterministic symmetry-breaking displacement confined to the
non-totally-symmetric subspace, so the optimizer is no longer seeded exactly on a
symmetry element. Both use the ``molsym`` package.
"""

from __future__ import annotations

import logging
from typing import Any

import molsym
import numpy as np
from molsym.salcs.cartesian_coordinates import CartesianCoordinates
from molsym.salcs.projection_op import ProjectionOp

from .geomlib import Geometry
from .species_data import get_property

__all__ = ['SYMMETRY_EPS', 'break_symmetry', 'detect_point_group']

log = logging.getLogger(__name__)

#: Default RMS amplitude (Å, per Cartesian component) of the symmetry-breaking
#: displacement. Calibrated as roughly the smallest amplitude that lets
#: ``symmetry='break'`` escape every symmetric Baker saddle (#148); the binding
#: case is caffeine, whose shallow saddle re-symmetrizes below ~0.018 Å. 0.02
#: keeps a small safety margin (every other case is resolved by <=0.005 Å).
SYMMETRY_EPS = 0.02


def _symtext(geom: Geometry) -> Any:
    """Build a MolSym ``Symtext`` (point group + symmetry operations) for ``geom``."""
    species = list(geom.species)
    # NB: a copy, not np.asarray -- MolSym recenters/reorients the molecule in
    # place, which would otherwise mutate the caller's geometry.
    coords = np.array(geom.coords, dtype=float)
    masses = np.array([float(get_property(sp, 'mass')) for sp in species])
    mol = molsym.Molecule(species, coords, masses)
    return molsym.Symtext.from_molecule(mol)


def _detect(geom: Geometry) -> tuple[str, Any]:
    """Return ``(point_group_symbol, symtext)`` for ``geom``.

    ``symtext`` is the reusable MolSym object behind the detection -- a caller
    that goes on to break the symmetry can pass it back to :func:`break_symmetry`
    instead of rebuilding it. It is ``None`` when detection does not apply
    (periodic geometry) or fails; in both cases the symbol is ``'C1'``. Detection
    runs by default on every optimization, so a failure degrades to ``'C1'``
    rather than breaking the run.
    """
    if geom.lattice is not None:
        return 'C1', None
    try:
        symtext = _symtext(geom)
    except Exception as e:  # pragma: no cover - defensive, never break a run
        log.debug('symmetry detection failed (%s); assuming C1', e)
        return 'C1', None
    return str(symtext.pg), symtext


def detect_point_group(geom: Geometry) -> str:
    """Return the Schoenflies point-group symbol of a molecular ``geom``.

    Periodic geometries (those carrying lattice vectors) always return ``'C1'``
    -- point groups apply to finite molecules. Detection uses MolSym; should it
    fail for any reason ``'C1'`` is returned rather than propagating the error.

    :param geom: geometry to classify
    """
    return _detect(geom)[0]


def break_symmetry(
    geom: Geometry, eps: float = SYMMETRY_EPS, *, symtext: Any = None
) -> Geometry:
    """Return a copy of ``geom`` displaced off its symmetry elements.

    The displacement is **deterministic** and **targeted**: it is the
    equal-weight sum of the non-totally-symmetric symmetry-adapted Cartesian
    displacement coordinates (SALCs) of ``geom``, rescaled to a per-component RMS
    of ``eps`` (Å). Confining the kick to the non-totally-symmetric subspace
    guarantees an order-one overlap with the energy-lowering imaginary mode that
    a gradient optimizer is otherwise blind to (issue #148), unlike an isotropic
    random kick whose overlap is a seed-dependent lottery. Translations and
    rotations are excluded via MolSym's Eckart projection. The input ``geom`` is
    not modified; a geometry MolSym finds to have no non-symmetric modes (e.g.
    already ``C1``) is returned unchanged.

    :param geom: geometry to perturb
    :param float eps: RMS displacement per Cartesian component (Å)
    :param symtext: precomputed MolSym ``Symtext`` for ``geom``; built when omitted
    """
    if symtext is None:
        symtext = _symtext(geom)
    salcs = ProjectionOp(symtext, CartesianCoordinates(symtext), project_Eckart=True)
    totally_symmetric = symtext.irreps[0].symbol
    nonsym = [salc.coeffs for salc in salcs if salc.irrep.symbol != totally_symmetric]
    if not nonsym:
        return geom
    direction = np.asarray(nonsym, dtype=float).sum(axis=0)
    rms = float(np.sqrt(np.mean(direction**2)))
    if rms == 0:  # pragma: no cover - a nonzero SALC basis sum always has rms > 0
        return geom
    displacement = (direction * (eps / rms)).reshape(-1, 3)
    coords = np.asarray(geom.coords, dtype=float)
    return Geometry(list(geom.species), coords + displacement, geom.lattice)
