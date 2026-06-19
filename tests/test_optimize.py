import importlib.util
from pathlib import Path

import pytest

from berny import Berny, geomlib, optimize
from berny.solvers import XTBSolver

XYZ_DIR = Path(__file__).parent

xtb_required = pytest.mark.skipif(
    importlib.util.find_spec('tblite') is None, reason='tblite not installed'
)


@pytest.fixture
def xtb():
    return XTBSolver()


def ethanol():
    return geomlib.readfile(str(XYZ_DIR / 'ethanol.xyz'))


def aniline():
    return geomlib.readfile(str(XYZ_DIR / 'aniline.xyz'))


def cyanogen():
    return geomlib.readfile(str(XYZ_DIR / 'cyanogen.xyz'))


def water():
    return geomlib.readfile(str(XYZ_DIR / 'water.xyz'))


@xtb_required
@pytest.mark.parametrize('test_case', [ethanol, aniline, cyanogen, water])
def test_optimize(xtb, test_case):
    # GFN2-xTB step counts aren't pinned to a historical baseline, so we only
    # assert that the optimization converges within a generous ceiling.
    geom = test_case()
    berny = Berny(geom, maxsteps=100)
    optimize(berny, xtb)
    assert berny.converged


@xtb_required
def test_optimize_writes_trajectory(xtb, tmp_path):
    geom = water()
    berny = Berny(geom, maxsteps=100)
    traj = tmp_path / 'traj.xyz'
    optimize(berny, xtb, trajectory=str(traj))
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
