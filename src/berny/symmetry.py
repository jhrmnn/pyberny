# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Point-group detection and symmetry-breaking of start geometries.

A gradient-following optimizer cannot leave the symmetric subspace of an exactly
symmetric start geometry: the gradient along every non-totally-symmetric mode is
zero by symmetry, so the run can converge to a symmetric *saddle* rather than a
minimum (issue #148). This module detects the point group (via the optional
``pointgroup`` package) and can apply a small, deterministic symmetry-breaking
displacement (via the optional ``molsym`` package) confined to the
non-totally-symmetric subspace, so the optimizer is no longer seeded exactly on a
symmetry element.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geomlib import Geometry
from .species_data import get_property

__all__ = ['SYMMETRY_EPS', 'break_symmetry', 'detect_point_group']

log = logging.getLogger(__name__)

FloatArray = NDArray[np.floating[Any]]

#: Default RMS amplitude (Å, per Cartesian component) of the symmetry-breaking
#: displacement. Calibrated as roughly the smallest amplitude that lets
#: ``symmetry='break'`` escape every symmetric Baker saddle (#148); the binding
#: case is caffeine, whose shallow saddle re-symmetrizes below ~0.018 Å. 0.02
#: keeps a small safety margin (every other case is resolved by <=0.005 Å).
SYMMETRY_EPS = 0.02


def detect_point_group(
    geom: Geometry,
    tolerance_eig: float = 0.01,
    tolerance_ang: float = 4.0,
) -> str:
    """Return the Schoenflies point-group symbol of a molecular ``geom``.

    Periodic geometries (those carrying lattice vectors) always return ``'C1'``
    -- point groups apply to finite molecules. Requires the optional
    ``pointgroup`` package; if it cannot be imported the detection is skipped and
    ``'C1'`` is returned so an optimization never fails for want of it.

    :param geom: geometry to classify
    :param float tolerance_eig: ``pointgroup`` inertia-eigenvalue tolerance
    :param float tolerance_ang: ``pointgroup`` angular tolerance (degrees)
    """
    if geom.lattice is not None:
        return 'C1'
    try:
        from pointgroup import PointGroup
    except ImportError:
        log.info('pointgroup is not installed; skipping symmetry detection')
        return 'C1'
    pg = PointGroup(
        positions=np.asarray(geom.coords, dtype=float),
        symbols=list(geom.species),
        tolerance_eig=tolerance_eig,
        tolerance_ang=tolerance_ang,
    )
    return str(pg.get_point_group())


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
    not modified; for a geometry MolSym finds to have no non-symmetric modes
    (e.g. already ``C1``) it is returned unchanged.

    Requires the optional ``molsym`` package (``pip install 'pyberny[symmetry]'``).

    :param geom: geometry to perturb
    :param float eps: RMS displacement per Cartesian component (Å)
    """
    try:
        import molsym
        from molsym.salcs.cartesian_coordinates import CartesianCoordinates
        from molsym.salcs.projection_op import ProjectionOp
    except ImportError as e:
        raise ImportError(
            "symmetry='break' requires the molsym package; install it with "
            f"`pip install 'pyberny[symmetry]'` (underlying import error: {e})"
        ) from e
    species = list(geom.species)
    coords = np.asarray(geom.coords, dtype=float)
    masses = np.array([float(get_property(sp, 'mass')) for sp in species])
    mol = molsym.Molecule(species, coords, masses)
    symtext = molsym.Symtext.from_molecule(mol)
    salcs = ProjectionOp(symtext, CartesianCoordinates(symtext), project_Eckart=True)
    totally_symmetric = symtext.irreps[0].symbol
    nonsym = [salc.coeffs for salc in salcs if salc.irrep.symbol != totally_symmetric]
    if not nonsym:
        return geom
    direction = np.asarray(nonsym, dtype=float).sum(axis=0)
    rms = float(np.sqrt(np.mean(direction**2)))
    if rms == 0:
        return geom
    displacement = (direction * (eps / rms)).reshape(-1, 3)
    return Geometry(species, coords + displacement, geom.lattice)
