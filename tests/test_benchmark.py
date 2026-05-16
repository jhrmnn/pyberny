"""Opt-in benchmark: reproduce Birkholz & Schlegel, Theor. Chem. Acc. 135, 84 (2016).

Deselected from the default ``pytest`` run by ``pyproject.toml``; run with
``pytest -m benchmark``. PySCF tests drive the optimization through PySCF's
own ``pyscf.geomopt.berny_solver`` bridge (so this also exercises the
upstream integration); they require ``pyberny[benchmark]``. MOPAC tests use
:func:`berny.solvers.MopacSolver` and require a ``mopac`` binary on ``$PATH``.
"""

import json
import shutil
from pathlib import Path

import pytest

from berny import Berny, geomlib, optimize

DATA = Path(__file__).parent / 'data' / 'birkholz_schlegel'
REF = json.loads((DATA / 'reference.json').read_text())

PYBERNY_STEP_MARGIN = 3
MOPAC_STEP_MARGIN = 3


@pytest.mark.benchmark
@pytest.mark.parametrize('name', sorted(REF))
def test_pyberny_vs_paper(name):
    pytest.importorskip('pyscf')
    from pyscf import dft, gto, scf
    from pyscf.geomopt import berny_solver

    ref = REF[name]
    expected = ref['paper_steps']
    if expected is None:
        pytest.skip(f'{name}: no paper_steps in reference')
    mol = gto.M(
        atom=str(DATA / f'{name}.xyz'),
        basis=ref['paper_steps_basis'],
        charge=ref['charge'],
        spin=ref['mult'] - 1,
        verbose=0,
    )
    method = ref['paper_steps_method']
    if method == 'HF':
        mf = scf.RHF(mol) if ref['mult'] == 1 else scf.UHF(mol)
    elif method == 'B3LYP':
        mf = dft.RKS(mol) if ref['mult'] == 1 else dft.UKS(mol)
        mf.xc = 'b3lyp'
    else:
        pytest.skip(f'{name}: unsupported paper method {method!r}')
    state = {'n': 0}
    converged, _ = berny_solver.kernel(
        mf, callback=lambda loc: state.update(n=loc['cycle'] + 1)
    )
    assert converged, f'{name}: did not converge'
    assert abs(state['n'] - expected) <= PYBERNY_STEP_MARGIN, (
        f'{name}: {state["n"]} steps vs paper {expected} ' f'(±{PYBERNY_STEP_MARGIN})'
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('name', sorted(REF))
def test_mopac_pm7(name):
    if not shutil.which('mopac'):
        pytest.skip('mopac not on PATH')
    from berny.solvers import MopacSolver

    ref = REF[name]
    expected = ref['mopac_pm7_steps']
    if expected is None:
        pytest.skip(f'{name}: no committed mopac_pm7_steps reference')
    geom = geomlib.readfile(str(DATA / f'{name}.xyz'))
    berny = Berny(geom)
    optimize(berny, MopacSolver(charge=ref['charge'], mult=ref['mult']))
    assert berny.converged, f'{name}: did not converge'
    # MOPAC PM7 is not bitwise-reproducible across hosts: BLAS/LAPACK
    # summation order and threading produce small gradient differences
    # that propagate over the optimizer's 30-100 iterations. The reference
    # numbers come from ubuntu-latest CI; tolerate a small drift.
    assert (
        abs(berny._n - expected) <= MOPAC_STEP_MARGIN
    ), f'{name}: {berny._n} steps vs reference {expected} (±{MOPAC_STEP_MARGIN})'
