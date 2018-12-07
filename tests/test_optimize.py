import numpy as np
from pkg_resources import resource_filename

import pytest
from pytest import approx

from berny import Berny, optimize, geomlib
from berny.solvers import MopacSolver


@pytest.fixture
def mopac(scope='session'):
    return MopacSolver()


@pytest.fixture
def ethanol():
    return geomlib.readfile(resource_filename('tests', 'ethanol.xyz'))


@pytest.fixture
def aniline():
    return geomlib.readfile(resource_filename('tests', 'aniline.xyz'))


@pytest.fixture
def cyanogen():
    return geomlib.readfile(resource_filename('tests', 'cyanogen.xyz'))


def test_ethanol(mopac, ethanol):
    berny = Berny(ethanol, steprms=0.01, stepmax=0.05, maxsteps=5)
    final = optimize(berny, mopac)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([14.95, 52.58, 61.10], rel=1e-3)


def test_aniline(mopac, aniline):
    berny = Berny(aniline, steprms=0.01, stepmax=0.05, maxsteps=8)
    final = optimize(berny, mopac)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([90.94, 193.1, 283.9], rel=1e-3)


def test_cyanogen(mopac, cyanogen):
    berny = Berny(cyanogen, steprms=0.01, stepmax=0.05, maxsteps=4)
    final = optimize(berny, mopac)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([0, 107.5, 107.5], rel=1e-3, abs=1e-3)
