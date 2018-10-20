import numpy as np
from pkg_resources import resource_filename

import pytest
from pytest import approx

from berny import optimize, geomlib
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


def test_ethanol(mopac, ethanol):
    final = optimize(mopac, ethanol, steprms=0.01, stepmax=0.05, maxsteps=5)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([14.95, 52.58, 61.10], rel=1e-3)


def test_aniline(mopac, aniline):
    final = optimize(mopac, aniline, steprms=0.01, stepmax=0.05, maxsteps=8)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([90.94, 193.1, 283.9], rel=1e-3)
