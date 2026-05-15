from pathlib import Path

import pytest

from berny import Berny, geomlib, optimize
from berny.solvers import MopacSolver

XYZ_DIR = Path(__file__).parent


@pytest.fixture
def mopac(scope='session'):
    return MopacSolver()


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
    assert berny._n == n_ref
