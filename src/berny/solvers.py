# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Generator, Iterable, Iterator
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
    keywords = [method, '1SCF', 'GRADIENTS']
    if charge:
        keywords.append(f'CHARGE={charge}')
    if mult_kw:
        keywords += [mult_kw, 'UHF']
    return ' '.join(keywords)


def _parse_float_token(token: str) -> float:
    return float(token.replace('D', 'E').replace('d', 'e'))


def _parse_mopac_energy(line: str) -> float:
    tokens = line.replace('=', ' = ').split()
    if '=' in tokens:
        tokens = tokens[tokens.index('=') + 1 :]
    for token in tokens:
        try:
            return _parse_float_token(token)
        except ValueError:
            pass
    raise ValueError(f'could not parse MOPAC energy line: {line.strip()}')


def _parse_mopac_gradient_value(line: str) -> float:
    tokens = line.split()
    if 'KCAL/ANGSTROM' in tokens:
        tokens = tokens[: tokens.index('KCAL/ANGSTROM')]
    for token in reversed(tokens):
        try:
            return _parse_float_token(token)
        except ValueError:
            pass
    raise ValueError(f'could not parse MOPAC gradient line: {line.strip()}')


def _read_mopac_gradient_value(
    lines: Iterator[str], component_number: int, total_components: int
) -> float:
    for line in lines:
        if 'ATOM' in line and 'CHEMICAL' in line:
            break
        if 'CARTESIAN' not in line:
            continue
        return _parse_mopac_gradient_value(line)
    raise ValueError(
        'MOPAC output ended before gradient component '
        f'{component_number} of {total_components}'
    )


def _parse_mopac_output(
    lines: Iterable[str], n_gradient_rows: int
) -> tuple[float, FloatArray]:
    line_iter = iter(lines)
    energy_line = next(
        (line for line in line_iter if 'FINAL HEAT OF FORMATION' in line), None
    )
    if energy_line is None:
        raise ValueError('MOPAC output is missing FINAL HEAT OF FORMATION')
    energy = _parse_mopac_energy(energy_line)
    derivatives_line = next(
        (
            line
            for line in line_iter
            if 'FINAL' in line and 'POINT' in line and 'DERIVATIVES' in line
        ),
        None,
    )
    if derivatives_line is None:
        raise ValueError('MOPAC output is missing FINAL POINT AND DERIVATIVES block')
    n_components = 3 * n_gradient_rows
    gradient_values = [
        _read_mopac_gradient_value(line_iter, i + 1, n_components)
        for i in range(n_components)
    ]
    gradients: FloatArray = np.array(gradient_values, dtype=float).reshape(
        (n_gradient_rows, 3)
    )
    return energy, gradients


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
                energy, gradients = _parse_mopac_output(
                    f, len(atoms) + (0 if lattice is None else 3)
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
