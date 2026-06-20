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
symmetry element. Both use the optional ``molsym`` package
(``pip install 'pyberny[symmetry]'``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

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
    """Build a MolSym ``Symtext`` (point group + symmetry operations) for ``geom``.

    Imports ``molsym`` lazily and propagates :class:`ImportError` when it is
    missing; the caller decides how to degrade.
    """
    import molsym

    species = list(geom.species)
    # NB: a copy, not np.asarray -- MolSym recenters/reorients the molecule in
    # place, which would otherwise mutate the caller's geometry.
    coords = np.array(geom.coords, dtype=float)
    masses = np.array([float(get_property(sp, 'mass')) for sp in species])
    mol = molsym.Molecule(species, coords, masses)
    return molsym.Symtext.from_molecule(mol)


def detect_point_group(geom: Geometry) -> str:
    """Return the Schoenflies point-group symbol of a molecular ``geom``.

    Periodic geometries (those carrying lattice vectors) always return ``'C1'``
    -- point groups apply to finite molecules. Detection uses the optional
    ``molsym`` package; if it is not installed (or detection fails for any
    reason) ``'C1'`` is returned so an optimization never breaks for want of it.

    :param geom: geometry to classify
    """
    if geom.lattice is not None:
        return 'C1'
    try:
        import molsym  # noqa: F401
    except ImportError:
        log.info('molsym is not installed; skipping symmetry detection')
        return 'C1'
    try:
        return str(_symtext(geom).pg)
    except Exception as e:  # pragma: no cover - defensive, detection is optional
        log.debug('symmetry detection failed (%s); assuming C1', e)
        return 'C1'


def break_symmetry(geom: Geometry, eps: float = SYMMETRY_EPS) -> Geometry:
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

    Requires the optional ``molsym`` package (``pip install 'pyberny[symmetry]'``).

    :param geom: geometry to perturb
    :param float eps: RMS displacement per Cartesian component (Å)
    """
    try:
        from molsym.salcs.cartesian_coordinates import CartesianCoordinates
        from molsym.salcs.projection_op import ProjectionOp

        symtext = _symtext(geom)
    except ImportError as e:
        raise ImportError(
            "symmetry='break' requires the molsym package; install it with "
            f"`pip install 'pyberny[symmetry]'` (underlying import error: {e})"
        ) from e
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
