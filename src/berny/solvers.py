# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Generator
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .coords import angstrom

__all__ = ['MopacSolver']

FloatArray = NDArray[np.floating[Any]]

# NB: ``Optional[X]`` is used instead of ``X | None`` because these aliases
# are evaluated at import time, and Sphinx's autodoc mocks numpy (see
# ``doc/conf.py``), making ``FloatArray`` a Mock whose ``__or__`` raises.
#: Geometry sent to a solver: per-atom ``(symbol, xyz)`` pairs and optional
#: lattice vectors (``None`` for a molecule).
SolverInput = tuple[list[tuple[str, FloatArray]], Optional[FloatArray]]  # noqa: UP045
#: Energy and gradients yielded by a solver (gradients in atomic units).
SolverOutput = tuple[float, FloatArray]
#: Generator type of a solver — yields ``None`` once before the first send.
Solver = Generator[Optional[SolverOutput], SolverInput, None]  # noqa: UP045


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
    keywords = [method, '1SCF', 'GRADIENTS', 'AUX(PRECISION=9)']
    if charge:
        keywords.append(f'CHARGE={charge}')
    if mult_kw:
        keywords += [mult_kw, 'UHF']
    return ' '.join(keywords)


def _parse_mopac_aux(path: str, n_grad: int) -> tuple[float, FloatArray]:
    """Parse the energy and gradients from a MOPAC ``AUX`` file.

    The ``AUX`` file carries the heat of formation to 15 significant figures
    (vs. ``1e-5 kcal/mol`` print quantization in the ``.out`` file) and the
    gradients to higher precision, both as flat whitespace-separated lists in
    a Fortran ``D`` exponent format. ``n_grad`` is the expected number of
    gradient components (``3`` per atom and lattice vector).
    """
    energy: float | None = None
    with open(path) as f:
        lines = iter(f)
        for line in lines:
            if line.startswith(' HEAT_OF_FORMATION:KCAL/MOL='):
                energy = float(line.split('=', 1)[1].replace('D', 'E'))
            elif line.startswith(' GRADIENTS:KCAL/MOL/ANGSTROM'):
                values: list[float] = []
                while len(values) < n_grad:
                    values += [float(t.replace('D', 'E')) for t in next(lines).split()]
                if energy is None:
                    raise ValueError('no HEAT_OF_FORMATION found in MOPAC AUX file')
                return energy, np.array(values).reshape(-1, 3)
    raise ValueError('no GRADIENTS found in MOPAC AUX file')


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
            n_grad = 3 * (len(atoms) + (0 if lattice is None else 3))
            energy, gradients = _parse_mopac_aux(
                os.path.join(tmpdir, 'job.aux'), n_grad
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
