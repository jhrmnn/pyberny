# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
from __future__ import division

import os
import shutil
import subprocess
import tempfile

import numpy as np

from .coords import angstrom

__all__ = ['MopacSolver']


def MopacSolver(cmd='mopac', method='PM7', workdir=None):
    """
    Crate a solver that wraps `MOPAC <http://openmopac.net>`_.

    Mopac needs to be installed on the system.

    :param str cmd: MOPAC executable
    :param str method: model to calculate energy
    """
    kcal = 1 / 627.503
    tmpdir = workdir or tempfile.mkdtemp()
    try:
        atoms, lattice = yield
        while True:
            mopac_input = '{} 1SCF GRADIENTS\n\n\n'.format(method) + '\n'.join(
                '{} {} 1 {} 1 {} 1'.format(el, *coord) for el, coord in atoms
            )
            if lattice is not None:
                mopac_input += '\n' + '\n'.join(
                    'Tv {} 1 {} 1 {} 1'.format(*vec) for vec in lattice
                )
            input_file = os.path.join(tmpdir, 'job.mop')
            with open(input_file, 'w') as f:
                f.write(mopac_input)
            subprocess.check_call([cmd, input_file])
            with open(os.path.join(tmpdir, 'job.out')) as f:
                energy = float(
                    next(l for l in f if 'FINAL HEAT OF FORMATION' in l).split()[5]
                )
                next(l for l in f if 'FINAL  POINT  AND  DERIVATIVES' in l)
                next(f)
                next(f)
                gradients = np.array(
                    [
                        [float(next(f).split()[6]) for _ in range(3)]
                        for _ in range(len(atoms) + (0 if lattice is None else 3))
                    ]
                )
            atoms, lattice = yield energy * kcal, gradients * kcal / angstrom
    finally:
        if tmpdir != workdir:
            shutil.rmtree(tmpdir)


def GenericSolver(f, *args, **kwargs):
    delta = kwargs.pop('delta', 1e-3)
    atoms, lattice = yield
    while True:
        energy = f(atoms, lattice, *args, **kwargs)
        coords = np.array([coord for _, coord in atoms])
        gradients = np.zeros(coords.shape)
        for i_atom in range(coords.shape[0]):
            for i_xyz in range(3):
                ene = {}
                for step in [-2, -1, 1, 2]:
                    coords_diff = coords.copy()
                    coords_diff[i_atom, i_xyz] += step * delta
                    atoms_diff = list(zip([sp for sp, _, in atoms], coords_diff))
                    ene[step] = f(atoms_diff, lattice, *args, **kwargs)
                gradients[i_atom, i_xyz] = _diff5(ene, delta)
        if lattice is not None:
            lattice_grads = np.zeros((3, 3))
            for i_vec in range(3):
                for i_xyz in range(3):
                    ene = {}
                    for step in [-2, -1, 1, 2]:
                        lattice_diff = lattice.copy()
                        lattice_diff[i_vec, i_xyz] += step * delta
                        ene[step] = f(atoms, lattice_diff, *args, **kwargs)
                    lattice_grads[i_vec, i_xyz] = _diff5(ene, delta)
            gradients = np.vstack((gradients, lattice_grads))
        atoms, lattice = yield energy, gradients / angstrom


def _diff5(x, delta):
    return (1 / 12 * x[-2] - 2 / 3 * x[-1] + 2 / 3 * x[1] - 1 / 12 * x[2]) / delta
