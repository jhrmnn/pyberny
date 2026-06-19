import numpy as np
import pytest

from berny.coords import angstrom
from berny.solvers import (
    GenericSolver,
    _diff5,
    _mopac_keyword_line,
    _parse_mopac_aux,
)

# A trimmed MOPAC AUX file for water (PM7 1SCF GRADIENTS AUX(PRECISION=9)).
# Energy and gradients are printed in Fortran ``D`` exponent format and to
# 15 significant figures -- far finer than the ``1e-5 kcal/mol`` quantization
# of the ``.out`` file the solver used to parse.
_WATER_AUX = """\
 HEAT_OF_FORMATION:KCAL/MOL=-0.577822806548975D+02
 GRADIENT_NORM:KCAL/MOL/ANGSTROM=+0.546567360177825D+01
 ATOM_X_OPT:ANGSTROMS[09]=
    0.0000000000000    0.0000000000000    0.0000000000000
 GRADIENTS:KCAL/MOL/ANGSTROM[09]=
  -3.4312143657544   0.0000000507512  -2.3676270992310   0.7963817277085 \
   0.0000000252344   2.2122353896105   2.6348324481675   0.0000000251599 \
   0.1553914772902
 CPU_TIME:SECONDS[1]=        0.01
"""


def test_parse_mopac_aux(tmp_path):
    aux = tmp_path / 'job.aux'
    aux.write_text(_WATER_AUX)
    energy, gradients = _parse_mopac_aux(str(aux), 9)
    assert energy == pytest.approx(-57.7822806548975, abs=0)
    assert gradients.shape == (3, 3)
    np.testing.assert_allclose(
        gradients,
        [
            [-3.4312143657544, 0.0000000507512, -2.3676270992310],
            [0.7963817277085, 0.0000000252344, 2.2122353896105],
            [2.6348324481675, 0.0000000251599, 0.1553914772902],
        ],
    )


def test_parse_mopac_aux_missing_gradients(tmp_path):
    aux = tmp_path / 'job.aux'
    aux.write_text(' HEAT_OF_FORMATION:KCAL/MOL=-0.577822806548975D+02\n')
    with pytest.raises(ValueError, match='no GRADIENTS'):
        _parse_mopac_aux(str(aux), 9)


def test_parse_mopac_aux_truncated_gradients(tmp_path):
    # GRADIENTS block present but cut short (MOPAC died mid-write): must raise
    # a clear, contextual error rather than letting ``next`` exhaust the file
    # and surface an opaque ``StopIteration``/``RuntimeError`` (see issue #130).
    aux = tmp_path / 'job.aux'
    aux.write_text(
        ' HEAT_OF_FORMATION:KCAL/MOL=-0.577822806548975D+02\n'
        ' GRADIENTS:KCAL/MOL/ANGSTROM[09]=\n'
        '  -3.4312143657544   0.0000000507512  -2.3676270992310\n'
    )
    with pytest.raises(ValueError, match='truncated'):
        _parse_mopac_aux(str(aux), 9)


def test_parse_mopac_aux_gradients_before_energy(tmp_path):
    # A GRADIENTS block with no preceding HEAT_OF_FORMATION is malformed.
    aux = tmp_path / 'job.aux'
    aux.write_text(' GRADIENTS:KCAL/MOL/ANGSTROM[03]=\n  1.0  2.0  3.0\n')
    with pytest.raises(ValueError, match='no HEAT_OF_FORMATION'):
        _parse_mopac_aux(str(aux), 3)


def test_mopac_neutral_singlet():
    assert _mopac_keyword_line('PM7', 0, 1) == 'PM7 1SCF GRADIENTS AUX(PRECISION=9)'


def test_mopac_cation():
    assert (
        _mopac_keyword_line('PM7', 1, 1)
        == 'PM7 1SCF GRADIENTS AUX(PRECISION=9) CHARGE=1'
    )


def test_mopac_dianion():
    assert (
        _mopac_keyword_line('PM7', -2, 1)
        == 'PM7 1SCF GRADIENTS AUX(PRECISION=9) CHARGE=-2'
    )


def test_mopac_doublet_cation():
    assert (
        _mopac_keyword_line('PM7', 1, 2)
        == 'PM7 1SCF GRADIENTS AUX(PRECISION=9) CHARGE=1 DOUBLET UHF'
    )


def test_mopac_unsupported_multiplicity():
    with pytest.raises(ValueError, match='unsupported MOPAC multiplicity'):
        _mopac_keyword_line('PM7', 0, 99)


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
