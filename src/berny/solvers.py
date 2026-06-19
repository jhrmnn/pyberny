# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .coords import angstrom
from .species_data import get_property

__all__ = ['XTBSolver']

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


#: Maps an ``XTBSolver`` method name to the corresponding ``tblite`` method.
_TBLITE_METHODS = {
    'gfn1': 'GFN1-xTB',
    'gfn2': 'GFN2-xTB',
    'ipea1': 'IPEA1-xTB',
}


def _tblite_method(method: str) -> str:
    """Normalise an ``XTBSolver`` method name to a ``tblite`` method string."""
    key = str(method).lower().replace('-', '').replace('_', '').replace(' ', '')
    if key in ('1', '2'):
        key = f'gfn{key}'
    try:
        return _TBLITE_METHODS[key]
    except KeyError as e:
        raise ValueError(
            f'unsupported xtb method: {method!r} '
            f'(choose from {", ".join(sorted(_TBLITE_METHODS))})'
        ) from e


def _tblite_geometry(
    atoms: list[tuple[str, FloatArray]],
) -> tuple[FloatArray, FloatArray]:
    """Convert ``(symbol, xyz_in_angstrom)`` atoms to the inputs tblite expects.

    Returns integer atomic numbers and Cartesian positions in bohr (tblite works
    in atomic units, unlike the Angstrom geometry pyberny uses).
    """
    numbers = np.array([int(get_property(sp, 'number')) for sp, _ in atoms])
    positions = np.array([coord for _, coord in atoms]) * angstrom
    return numbers, positions


def _tblite_singlepoint(
    method: str,
    atoms: list[tuple[str, FloatArray]],
    charge: int,
    mult: int,
    accuracy: float | None,
) -> SolverOutput:
    """Run a single tblite energy+gradient evaluation via the Python bindings.

    Energy (Hartree) and gradient (Hartree/bohr) come back in atomic units and
    are returned unchanged -- no unit conversion needed.
    """
    try:
        from tblite.interface import Calculator
    except ImportError as e:
        raise ImportError(
            'XTBSolver requires the tblite package; install it with '
            f'`pip install pyberny[benchmark]` (underlying import error: {e})'
        ) from e
    numbers, positions = _tblite_geometry(atoms)
    calc = Calculator(method, numbers, positions, charge=float(charge), uhf=mult - 1)
    calc.set('verbosity', 0)
    if accuracy is not None:
        calc.set('accuracy', accuracy)
    res = calc.singlepoint()
    # tblite returns the energy as a 0-d array; coerce to a plain float so the
    # SolverOutput contract holds (and the value stays JSON-serialisable).
    return float(res.get('energy')), res.get('gradient')


def XTBSolver(
    method: str = 'gfn2',
    *,
    charge: int = 0,
    mult: int = 1,
    accuracy: float | None = None,
) -> Solver:
    """
    Create a solver for the `xTB <https://tblite.readthedocs.io>`_ family of
    semiempirical tight-binding methods, evaluated through the `tblite
    <https://tblite.readthedocs.io>`_ library.

    The ``tblite`` package must be installed (``pip install pyberny[benchmark]``).
    GFN2-xTB has a smooth potential-energy surface, which makes it well behaved
    even near flat minima.

    :param str method: xTB parametrisation -- ``'gfn2'`` (default), ``'gfn1'``
        or ``'ipea1'``
    :param int charge: total charge (keyword-only)
    :param int mult: spin multiplicity, keyword-only (1 = singlet, 2 = doublet,
        ...); passed to tblite as ``mult - 1`` unpaired electrons
    :param accuracy: tblite numerical accuracy (smaller is tighter); the tblite
        default is used when ``None`` (keyword-only)
    """
    if mult < 1:
        raise ValueError(f'multiplicity must be >= 1, got {mult}')
    tblite_method = _tblite_method(method)
    atoms, lattice = yield None
    while True:
        if lattice is not None:
            raise NotImplementedError(
                'XTBSolver does not support periodic systems (lattice vectors)'
            )
        energy, gradients = _tblite_singlepoint(
            tblite_method, atoms, charge, mult, accuracy
        )
        atoms, lattice = yield energy, gradients


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
