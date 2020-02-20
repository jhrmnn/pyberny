import pytest
from pkg_resources import resource_filename

from berny import Berny, geomlib, optimize
from berny.solvers import MopacSolver


@pytest.fixture
def mopac(scope='session'):
    return MopacSolver()


def ethanol():
    return geomlib.readfile(resource_filename('tests', 'ethanol.xyz')), 5


def aniline():
    return geomlib.readfile(resource_filename('tests', 'aniline.xyz')), 8


def cyanogen():
    return geomlib.readfile(resource_filename('tests', 'cyanogen.xyz')), 4


def water():
    return geomlib.readfile(resource_filename('tests', 'water.xyz')), 6


@pytest.mark.parametrize('test_case', [ethanol, aniline, cyanogen, water])
def test_optimize(mopac, test_case):
    geom, n_ref = test_case()
    berny = Berny(geom)
    optimize(berny, mopac)
    assert berny.converged
    assert berny._n == n_ref
