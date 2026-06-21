import importlib.util
import logging

import numpy as np
import pytest

from berny import Berny, Geometry, break_symmetry
from berny.benchmarks import iter_molecules
from berny.solvers import XTBSolver
from berny.symmetry import SYMMETRY_EPS, _detect

xtb_required = pytest.mark.skipif(
    importlib.util.find_spec('tblite') is None, reason='tblite not installed'
)


def water():
    # Exact C2v geometry.
    return Geometry(
        ['O', 'H', 'H'],
        [[0.0, 0.117, 0.0], [0.0, -0.469, 0.757], [0.0, -0.469, -0.757]],
    )


def c1_geom():
    # Four distinct elements at generic, non-coplanar positions: no nontrivial
    # point-group operation. (Three atoms would always be at least Cs, since any
    # three points are coplanar.)
    return Geometry(
        ['C', 'H', 'O', 'N'],
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.2], [-0.3, 1.1, 0.05], [0.2, -0.4, 1.0]],
    )


def methylamine():
    ((_, geom, _),) = list(iter_molecules('baker', ['methylamine']))
    return geom


# -- detection (_detect) -----------------------------------------------------


def test_detect_water_is_c2v():
    assert _detect(water())[0] == 'C2v'


def test_detect_methylamine_start_is_cs():
    assert _detect(methylamine())[0] == 'Cs'


def test_detect_asymmetric_is_c1():
    assert _detect(c1_geom())[0] == 'C1'


def test_detect_periodic_short_circuits_to_c1():
    # Point groups apply to finite molecules; a geometry with a lattice must
    # report C1 without ever invoking the detector.
    geom = water()
    geom.lattice = np.eye(3) * 10.0
    assert _detect(geom)[0] == 'C1'


# -- break_symmetry (targeted, deterministic) --------------------------------


def test_break_symmetry_does_not_mutate_input():
    geom = methylamine()
    before = geom.coords.copy()
    break_symmetry(geom, eps=0.05)
    assert np.array_equal(geom.coords, before)


def test_break_symmetry_changes_geometry_and_lowers_symmetry():
    geom = methylamine()
    broken = break_symmetry(geom, eps=0.05)
    assert not np.allclose(broken.coords, geom.coords)
    assert _detect(broken)[0] == 'C1'


def test_break_symmetry_is_deterministic():
    # No seed: repeated calls must be bit-for-bit identical.
    geom = methylamine()
    a = break_symmetry(geom, eps=0.05)
    b = break_symmetry(geom, eps=0.05)
    assert np.array_equal(a.coords, b.coords)


def test_break_symmetry_realizes_requested_rms():
    geom = methylamine()
    eps = 0.05
    disp = break_symmetry(geom, eps=eps).coords - geom.coords
    assert np.sqrt(np.mean(disp**2)) == pytest.approx(eps)


def test_break_symmetry_is_noop_for_asymmetric():
    # A C1 geometry has no non-symmetric modes, so breaking it returns the very
    # same object unchanged.
    geom = c1_geom()
    assert break_symmetry(geom) is geom


# -- Berny symmetry policy ---------------------------------------------------


def _warning_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_default_logs_warning_on_symmetric_start(caplog):
    # The default notice is a logging warning (not a Python UserWarning), so it
    # never trips downstream test suites run with -W error.
    with caplog.at_level(logging.INFO, logger='berny.berny'):
        Berny(water())
    assert any('C2v' in r.getMessage() for r in _warning_records(caplog))


def test_nowarn_is_silent_but_still_checks(caplog):
    with caplog.at_level(logging.INFO, logger='berny.berny'):
        b = Berny(water(), symmetry='nowarn')
    # No warning-level record, and the geometry is left untouched.
    assert not _warning_records(caplog)
    assert np.array_equal(b._state.geom.coords, water().coords)


def test_no_warning_for_asymmetric_start(caplog):
    with caplog.at_level(logging.INFO, logger='berny.berny'):
        Berny(c1_geom())
    assert not _warning_records(caplog)


def test_invalid_symmetry_mode_rejected():
    with pytest.raises(ValueError, match='symmetry must be one of'):
        Berny(water(), symmetry='bogus')


def test_break_perturbs_the_optimized_geometry():
    geom = water()
    b = Berny(geom, symmetry='break')
    expected = break_symmetry(geom, SYMMETRY_EPS)
    assert np.array_equal(b._state.geom.coords, expected.coords)


def test_break_is_noop_for_asymmetric_start():
    # symmetry='break' on a C1 start leaves the geometry untouched.
    geom = c1_geom()
    b = Berny(geom, symmetry='break')
    assert np.array_equal(b._state.geom.coords, geom.coords)


# -- end-to-end --------------------------------------------------------------


@xtb_required
def test_break_escapes_methylamine_saddle():
    # The planar (Cs) methylamine start is the umbrella-inversion saddle; the
    # default symmetry handling leaves it there, while symmetry='break' falls to
    # the ~6.2 kcal/mol-lower pyramidal minimum (issue #148).
    geom = methylamine()

    def run(**kw):
        opt = Berny(geom, maxsteps=150, **kw)
        solver = XTBSolver()
        next(solver)
        energy = None
        for g in opt:
            energy, gradients = solver.send((list(g), g.lattice))
            opt.send((energy, gradients))
        assert opt.converged
        return energy

    saddle = run(symmetry='nowarn')
    minimum = run(symmetry='break')
    assert (minimum - saddle) * 627.509 < -5.0
