"""End-to-end optimizer tests built on the :mod:`berny.tests` model potentials.

This is the designated home for tests that drive :class:`berny.Berny` against
the analytic model potentials shipped in :mod:`berny.tests`; add a new block
here as further potentials are introduced.

The two cases exercise the linear-bend / dihedral coordinate handoff in
opposite directions:

* :class:`~berny.tests.LinearBendCrossover` starts bent (a dihedral, no
  dummies) and relaxes to a linear ``a-b-c``, so the optimizer must cross the
  175 deg threshold and rebuild -- dropping the dihedral for two dummy bends.
* :class:`~berny.tests.DihedralFromLinear` runs the reverse: it starts with
  ``a-b-c`` near-linear (two dummy bends, no dihedral) and relaxes to an
  ordinary bent minimum, crossing back below the 170 deg exit threshold so the
  rebuild drops the dummies and introduces a genuine dihedral.
"""

import logging

import numpy as np
import pytest

from berny import Berny, optimize
from berny.coords import Dihedral, InternalCoords, angstrom
from berny.geomlib import Geometry
from berny.tests import (
    DihedralFromLinear,
    LinearBendCrossover,
    ModelPotential,
    run_and_check,
)

_POTENTIALS = [LinearBendCrossover(), DihedralFromLinear()]
_POTENTIAL_IDS = ['linear_bend', 'dihedral_from_linear']


def _solver(energy, gradient):
    atoms, _ = yield
    while True:
        xyz = np.array([coord for _, coord in atoms])
        atoms, _ = yield energy(xyz), gradient(xyz) / angstrom


class _RebuildCatcher(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        if 'rebuilding internal coordinates' in record.getMessage():
            self.sink.append(record.getMessage())


def _run_through_berny(pot):
    """Drive ``Berny`` on ``pot`` via ``run_and_check``.

    Returns ``(final_coords, converged, n_rebuilds)``, capturing how many times
    the optimizer rebuilt its internal coordinates mid-run.
    """
    rebuilds = []
    captured = {}
    handler = _RebuildCatcher(rebuilds)
    logger = logging.getLogger('berny.berny')
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:

        def minimize(species, coords, energy, gradient):
            opt = Berny(Geometry(species, coords), maxsteps=100)
            captured['opt'] = opt
            return optimize(opt, _solver(energy, gradient)).coords

        final = run_and_check(pot, minimize)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    return final, captured['opt'].converged, len(rebuilds)


def test_model_potential_base_is_abstract():
    base = ModelPotential()
    arr = np.zeros((4, 3))
    with pytest.raises(NotImplementedError):
        base.start()
    with pytest.raises(NotImplementedError):
        base.energy(arr)
    with pytest.raises(NotImplementedError):
        base.gradient(arr)
    with pytest.raises(NotImplementedError):
        base.assert_at_minimum(arr)


@pytest.mark.parametrize('pot', _POTENTIALS, ids=_POTENTIAL_IDS)
def test_assert_at_minimum_rejects_start(pot):
    # Each start geometry is deliberately far from the minimum, so the check
    # must reject it -- guarding against a vacuous assertion.
    with pytest.raises(AssertionError, match='not at minimum'):
        pot.assert_at_minimum(pot.start())


@pytest.mark.parametrize('pot', _POTENTIALS, ids=_POTENTIAL_IDS)
def test_potential_gradient_matches_numeric(pot):
    rng = np.random.default_rng(0)
    coords = pot.start() + rng.standard_normal((4, 3)) * 0.1
    assert np.allclose(pot.gradient(coords), pot.numerical_gradient(coords), atol=1e-6)


def test_linear_bend_crossover_end_to_end():
    pot = LinearBendCrossover()

    # Start: both component angles are non-linear, so the coordinate set has a
    # dihedral and no dummy atoms.
    ic0 = InternalCoords(Geometry(list(pot.species), pot.start()))
    assert any(isinstance(c, Dihedral) for c in ic0)
    assert ic0.dummy_atoms.shape[0] == 0

    final, converged, n_rebuilds = _run_through_berny(pot)

    assert converged
    # Exactly one dihedral -> dummy crossover on the way to the linear minimum.
    assert n_rebuilds == 1

    # After the rebuild the dihedral is gone and two dummy atoms carry the bend
    # at the linear centre.
    ic_final = InternalCoords(Geometry(list(pot.species), final))
    assert not any(isinstance(c, Dihedral) for c in ic_final)
    assert ic_final.dummy_atoms.shape[0] == 2


def test_dihedral_from_linear_end_to_end():
    pot = DihedralFromLinear()

    # Start: a-b-c is near-linear, so the coordinate set has two dummy bends and
    # no dihedral -- the mirror image of the linear-bend case.
    ic0 = InternalCoords(Geometry(list(pot.species), pot.start()))
    assert not any(isinstance(c, Dihedral) for c in ic0)
    assert ic0.dummy_atoms.shape[0] == 2

    final, converged, n_rebuilds = _run_through_berny(pot)

    assert converged
    # Exactly one dummy -> dihedral crossover on the way to the bent minimum.
    assert n_rebuilds == 1

    # After the rebuild the dummies are gone and an ordinary dihedral describes
    # the now-bent chain.
    ic_final = InternalCoords(Geometry(list(pot.species), final))
    assert any(isinstance(c, Dihedral) for c in ic_final)
    assert ic_final.dummy_atoms.shape[0] == 0
