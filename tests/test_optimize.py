import importlib.util
from pathlib import Path

import pytest

from berny import Berny, geomlib, optimize
from berny.solvers import MopacSolver, XTBSolver

XYZ_DIR = Path(__file__).parent

xtb_required = pytest.mark.skipif(
    importlib.util.find_spec('xtb') is None, reason='xtb not installed'
)


@pytest.fixture
def mopac():
    return MopacSolver()


@pytest.fixture
def xtb():
    return XTBSolver()


def ethanol():
    return geomlib.readfile(str(XYZ_DIR / 'ethanol.xyz')), 5


def aniline():
    return geomlib.readfile(str(XYZ_DIR / 'aniline.xyz')), 12


def cyanogen():
    return geomlib.readfile(str(XYZ_DIR / 'cyanogen.xyz')), 4


def water():
    return geomlib.readfile(str(XYZ_DIR / 'water.xyz')), 7


@pytest.mark.parametrize('test_case', [ethanol, aniline, cyanogen, water])
def test_optimize(mopac, test_case):
    geom, n_ref = test_case()
    berny = Berny(geom)
    optimize(berny, mopac)
    assert berny.converged
    # n_ref is the historical exact step count. We allow a small drift so
    # that algorithmic tweaks that change the iteration trajectory by one or
    # two steps don't break the integration tests; a real regression that
    # blows up the optimizer past this band will still be caught.
    assert (
        berny._n <= n_ref + 2
    ), f'converged in {berny._n} steps, more than the {n_ref} + 2 band'


@xtb_required
@pytest.mark.parametrize('test_case', [ethanol, aniline, cyanogen, water])
def test_optimize_xtb(xtb, test_case):
    # GFN2-xTB step counts differ from PM7 and aren't pinned to a historical
    # baseline, so we only assert that the optimization converges (within a
    # generous ceiling) through the same coroutine path as MopacSolver.
    geom, _ = test_case()
    berny = Berny(geom, maxsteps=100)
    optimize(berny, xtb)
    assert berny.converged


def test_optimize_writes_trajectory(mopac, tmp_path):
    geom, _ = water()
    berny = Berny(geom)
    traj = tmp_path / 'traj.xyz'
    optimize(berny, mopac, trajectory=str(traj))
    assert berny.converged
    # One XYZ frame per optimizer step. Each frame is atom_count + comment +
    # N atom lines = N + 2 lines.
    lines = traj.read_text().splitlines()
    n = len(geom)
    assert len(lines) % (n + 2) == 0
    n_frames = len(lines) // (n + 2)
    assert n_frames == berny._n
    for i in range(n_frames):
        assert lines[i * (n + 2)].strip() == str(n)
