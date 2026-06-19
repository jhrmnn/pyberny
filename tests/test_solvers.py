import sys
import types

import numpy as np
import pytest

from berny.coords import angstrom
from berny.solvers import (
    GenericSolver,
    XTBSolver,
    _diff5,
    _tblite_geometry,
    _tblite_method,
)


@pytest.mark.parametrize(
    ('method', 'expected'),
    [
        ('gfn2', 'GFN2-xTB'),
        ('2', 'GFN2-xTB'),
        ('GFN1', 'GFN1-xTB'),
        ('ipea1', 'IPEA1-xTB'),
        ('IPEA1', 'IPEA1-xTB'),
    ],
)
def test_tblite_method(method, expected):
    assert _tblite_method(method) == expected


def test_tblite_method_unsupported():
    with pytest.raises(ValueError, match='unsupported xtb method'):
        _tblite_method('bogus')


def test_tblite_geometry_numbers_and_bohr_positions():
    # Atomic numbers come from the species table and coordinates are converted
    # from Angstrom to bohr (the atomic units tblite expects).
    atoms = [
        ('O', np.array([0.0, 0.0, 0.0])),
        ('H', np.array([0.0, 0.0, 0.96])),
    ]
    numbers, positions = _tblite_geometry(atoms)
    np.testing.assert_array_equal(numbers, [8, 1])
    np.testing.assert_allclose(positions, [[0, 0, 0], [0, 0, 0.96 * angstrom]])


def test_xtb_solver_rejects_nonpositive_multiplicity():
    # Validation happens when the generator is primed (before any tblite import),
    # so this raises even when tblite is not installed.
    with pytest.raises(ValueError, match='multiplicity must be >= 1'):
        next(XTBSolver(mult=0))


def test_xtb_solver_rejects_periodic_system():
    # GFN2-xTB here is molecule-only; a non-None lattice must raise. The lattice
    # check runs before the lazy tblite import, so it works without tblite.
    solver = XTBSolver()
    next(solver)
    atoms = [('H', np.array([0.0, 0.0, 0.0]))]
    lattice = np.eye(3)
    with pytest.raises(NotImplementedError, match='periodic'):
        solver.send((atoms, lattice))


def _install_fake_tblite(monkeypatch):
    """Inject a minimal fake ``tblite`` package into ``sys.modules``.

    This lets the XTBSolver single-point path (import, Calculator construction,
    verbosity/accuracy, singlepoint, result unpacking) be exercised
    deterministically without the real ``tblite`` package. ``calls`` captures
    what the solver passed in.
    """
    calls: dict = {'settings': {}}

    class _Result:
        def __init__(self, n_atoms):
            self._n = n_atoms

        def get(self, key):
            return {'energy': -1.5, 'gradient': np.zeros((self._n, 3))}[key]

    class Calculator:
        def __init__(self, method, numbers, positions, charge=0.0, uhf=0):
            calls.update(
                method=method,
                numbers=numbers,
                positions=positions,
                charge=charge,
                uhf=uhf,
            )
            self._n = len(numbers)

        def set(self, key, value):
            calls['settings'][key] = value

        def singlepoint(self):
            return _Result(self._n)

    interface = types.ModuleType('tblite.interface')
    interface.Calculator = Calculator
    monkeypatch.setitem(sys.modules, 'tblite', types.ModuleType('tblite'))
    monkeypatch.setitem(sys.modules, 'tblite.interface', interface)
    return calls


def test_xtb_singlepoint_passes_inputs_and_returns_atomic_units(monkeypatch):
    calls = _install_fake_tblite(monkeypatch)
    solver = XTBSolver(method='gfn2', charge=-1, mult=2, accuracy=0.5)
    next(solver)
    atoms = [
        ('O', np.array([0.0, 0.0, 0.0])),
        ('H', np.array([0.0, 0.0, 0.96])),
    ]
    energy, gradients = solver.send((atoms, None))

    # Energy (Hartree) and gradient (Hartree/bohr) are returned unchanged.
    assert energy == -1.5
    assert gradients.shape == (2, 3)
    # Method, charge and uhf = mult - 1 are forwarded; coordinates reach tblite
    # in bohr and elements as atomic numbers.
    assert calls['method'] == 'GFN2-xTB'
    assert calls['charge'] == -1.0
    assert calls['uhf'] == 1
    assert calls['settings']['accuracy'] == 0.5
    np.testing.assert_array_equal(calls['numbers'], [8, 1])
    np.testing.assert_allclose(calls['positions'][1], [0.0, 0.0, 0.96 * angstrom])


def test_xtb_singlepoint_skips_accuracy_when_unset(monkeypatch):
    calls = _install_fake_tblite(monkeypatch)
    solver = XTBSolver()
    next(solver)
    energy, _ = solver.send(([('H', np.zeros(3))], None))
    assert energy == -1.5
    # accuracy defaults to None, so it is never passed to tblite.
    assert 'accuracy' not in calls['settings']


def test_xtb_solver_missing_bindings_raises_helpful_error(monkeypatch):
    # Force the lazy tblite import to fail (covers the ImportError branch even
    # when the real package is installed).
    monkeypatch.setitem(sys.modules, 'tblite', None)
    monkeypatch.setitem(sys.modules, 'tblite.interface', None)
    solver = XTBSolver()
    next(solver)
    with pytest.raises(ImportError, match='tblite'):
        solver.send(([('H', np.zeros(3))], None))


def test_diff5_recovers_derivative_of_cubic():
    # 5-point stencil is exact (to numerical noise) for polynomials up to
    # degree 4, so x^3 derivative should come out spot-on at any x0.
    x0, delta = 1.7, 1e-3
    samples = {step: (x0 + step * delta) ** 3 for step in (-2, -1, 1, 2)}
    assert _diff5(samples, delta) == pytest.approx(3 * x0**2, rel=1e-6)


def _quadratic_factory(target):
    target = np.asarray(target, dtype=float)

    def f(atoms, lattice):
        coords = np.array([c for _, c in atoms])
        return float(np.sum((coords - target) ** 2))

    return f


def test_generic_solver_matches_analytical_gradient():
    target = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    f = _quadratic_factory(target)

    solver = GenericSolver(f)
    next(solver)
    atoms = [
        ('H', np.array([0.1, 0.0, 0.0])),
        ('H', np.array([0.0, 1.2, 0.0])),
    ]
    energy, gradients = solver.send((atoms, None))

    coords = np.array([c for _, c in atoms])
    assert energy == pytest.approx(float(np.sum((coords - target) ** 2)))
    # Analytic ∂E/∂x_i = 2(x_i − target_i); GenericSolver divides the
    # finite-difference gradient by `angstrom` to convert per-Å → per-bohr.
    expected = 2 * (coords - target) / angstrom
    np.testing.assert_allclose(gradients, expected, atol=1e-6)


def test_generic_solver_handles_lattice():
    # Energy depends on both atoms and lattice; the solver must finite-
    # difference both blocks and stack them ((N+3, 3) gradient).
    def f(atoms, lattice):
        coords = np.array([c for _, c in atoms])
        return float(np.sum(coords**2) + np.sum(lattice**2))

    solver = GenericSolver(f)
    next(solver)
    atoms = [('H', np.array([0.5, 0.0, 0.0]))]
    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    energy, gradients = solver.send((atoms, lattice))

    assert energy == pytest.approx(0.25 + 27.0)
    assert gradients.shape == (4, 3)
    np.testing.assert_allclose(gradients[0], 2 * atoms[0][1] / angstrom, atol=1e-6)
    np.testing.assert_allclose(gradients[1:], 2 * lattice / angstrom, atol=1e-6)


def test_generic_solver_forwards_extra_args_and_kwargs():
    def f(atoms, lattice, scale, *, offset):
        coords = np.array([c for _, c in atoms])
        return scale * float(np.sum((coords - offset) ** 2))

    solver = GenericSolver(f, 3.0, offset=np.array([1.0, 0.0, 0.0]))
    next(solver)
    atoms = [('H', np.array([0.0, 0.0, 0.0]))]
    energy, gradients = solver.send((atoms, None))

    assert energy == pytest.approx(3.0)
    np.testing.assert_allclose(
        gradients[0],
        3.0 * 2 * (atoms[0][1] - np.array([1.0, 0.0, 0.0])) / angstrom,
        atol=1e-6,
    )


def test_generic_solver_accepts_custom_delta():
    # Pop the `delta` kwarg out of kwargs so it's not forwarded to f.
    captured = {}

    def f(atoms, lattice):
        captured['called'] = True
        coords = np.array([c for _, c in atoms])
        return float(np.sum(coords**2))

    solver = GenericSolver(f, delta=1e-2)
    next(solver)
    atoms = [('H', np.array([0.3, 0.0, 0.0]))]
    energy, gradients = solver.send((atoms, None))

    assert captured['called']
    assert energy == pytest.approx(0.09)
    np.testing.assert_allclose(gradients[0], 2 * atoms[0][1] / angstrom, atol=1e-4)
