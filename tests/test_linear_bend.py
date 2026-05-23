"""End-to-end test of the linear-bend / dihedral handoff during optimization.

Drives :class:`berny.Berny` on the :class:`berny.tests.LinearBendCrossover`
model potential from a start where *both* bond angles are non-linear. The
optimizer therefore begins with a dihedral and no dummy atoms, must cross the
175 deg linear-bend threshold on the way to the linear minimum, and at that
point rebuilds its internal coordinates -- dropping the dihedral and switching
to the dummy-atom bend representation.
"""

import logging

import numpy as np
import pytest

from berny import Berny, optimize
from berny.coords import Dihedral, InternalCoords, angstrom
from berny.geomlib import Geometry
from berny.tests import LinearBendCrossover, ModelPotential, run_and_check


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


def test_assert_at_minimum_rejects_wrong_geometry():
    pot = LinearBendCrossover()
    # The start geometry is deliberately far from the minimum (a-b-c at 150 deg),
    # so the check must reject it -- guarding against a vacuous assertion.
    with pytest.raises(AssertionError, match='not at minimum'):
        pot.assert_at_minimum(pot.start())


def test_linear_bend_potential_gradient_matches_numeric():
    pot = LinearBendCrossover()
    rng = np.random.default_rng(0)
    coords = pot.start() + rng.standard_normal((4, 3)) * 0.1
    assert np.allclose(pot.gradient(coords), pot.numerical_gradient(coords), atol=1e-6)


def test_linear_bend_crossover_end_to_end():
    pot = LinearBendCrossover()

    # Start: both component angles are non-linear, so the coordinate set has a
    # dihedral and no dummy atoms.
    ic0 = InternalCoords(Geometry(list(pot.species), pot.start()))
    assert any(isinstance(c, Dihedral) for c in ic0)
    assert len(ic0._dummy_specs) == 0

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

    assert captured['opt'].converged
    # Exactly one dihedral -> dummy crossover on the way to the linear minimum.
    assert len(rebuilds) == 1

    # After the rebuild the dihedral is gone and two dummy atoms carry the bend
    # at the linear centre.
    ic_final = InternalCoords(Geometry(list(pot.species), final))
    assert not any(isinstance(c, Dihedral) for c in ic_final)
    assert len(ic_final._dummy_specs) == 2
