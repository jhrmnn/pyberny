# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .coords import angstrom

__all__ = ['MopacSolver']

FloatArray = NDArray[np.floating[Any]]

#: Geometry sent to a solver: per-atom ``(symbol, xyz)`` pairs and optional
#: lattice vectors (``None`` for a molecule).
SolverInput = tuple[list[tuple[str, FloatArray]], FloatArray | None]
#: Energy and gradients yielded by a solver (gradients in atomic units).
SolverOutput = tuple[float, FloatArray]
#: Generator type of a solver — yields ``None`` once before the first send.
Solver = Generator[SolverOutput | None, SolverInput, None]


_MOPAC_MULT_KEYWORDS = {
    1: '',
    2: 'DOUBLET',
    3: 'TRIPLET',
    4: 'QUARTET',
    5: 'QUINTET',
    6: 'SEXTET',
    7: 'SEPTET',
}


def _mopac_keyword_line(method: str, charge: int, mult: int) -> str:
    """Build the MOPAC keyword line for a single-point gradient run."""
    try:
        mult_kw = _MOPAC_MULT_KEYWORDS[mult]
    except KeyError as e:
        raise ValueError(f'unsupported MOPAC multiplicity: {mult}') from e
    keywords = [method, '1SCF', 'GRADIENTS']
    if charge:
        keywords.append(f'CHARGE={charge}')
    if mult_kw:
        keywords += [mult_kw, 'UHF']
    return ' '.join(keywords)


def MopacSolver(
    cmd: str = 'mopac',
    method: str = 'PM7',
    workdir: str | None = None,
    *,
    charge: int = 0,
    mult: int = 1,
) -> Solver:
    """
    Crate a solver that wraps `MOPAC <http://openmopac.net>`_.

    Mopac needs to be installed on the system.

    :param str cmd: MOPAC executable
    :param str method: model to calculate energy
    :param workdir: directory for MOPAC scratch files (default: a tempdir)
    :param int charge: total charge (keyword-only)
    :param int mult: spin multiplicity, keyword-only (1 = singlet, 2 = doublet,
        ...); values > 1 also switch MOPAC to UHF
    """
    keyword_line = _mopac_keyword_line(method, charge, mult)
    kcal = 1 / 627.503
    tmpdir = workdir or tempfile.mkdtemp()
    try:
        atoms, lattice = yield None
        while True:
            mopac_input = f'{keyword_line}\n\n\n' + '\n'.join(
                f'{el} {x} 1 {y} 1 {z} 1' for el, (x, y, z) in atoms
            )
            if lattice is not None:
                mopac_input += '\n' + '\n'.join(
                    f'Tv {x} 1 {y} 1 {z} 1' for x, y, z in lattice
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


def GenericSolver(f: Callable[..., float], *args: Any, **kwargs: Any) -> Solver:
    delta: float = kwargs.pop('delta', 1e-3)
    atoms, lattice = yield None
    while True:
        energy = f(atoms, lattice, *args, **kwargs)
        coords = np.array([coord for _, coord in atoms])
        gradients = np.zeros(coords.shape)
        for i_atom in range(coords.shape[0]):
            for i_xyz in range(3):
                ene: dict[int, float] = {}
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


def _diff5(x: dict[int, float], delta: float) -> float:
    return (1 / 12 * x[-2] - 2 / 3 * x[-1] + 2 / 3 * x[1] - 1 / 12 * x[2]) / delta
