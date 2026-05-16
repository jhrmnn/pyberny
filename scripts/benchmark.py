#!/usr/bin/env python3
"""Run the Birkholz-Schlegel 2016 benchmark and print a markdown summary.

Usage::

    scripts/benchmark.py --solver pyscf [--molecules NAME ...] [--out PATH]
    scripts/benchmark.py --solver mopac

PySCF mode drives the optimization through ``pyscf.geomopt.berny_solver``
and requires ``pip install pyberny[benchmark]``; MOPAC mode uses
:func:`berny.solvers.MopacSolver` and requires a ``mopac`` binary on
``$PATH``. Non-neutral / open-shell molecules are skipped in MOPAC mode
because :func:`MopacSolver` does not currently expose charge/multiplicity.
"""

# Pin numeric-library thread counts to physical cores before importing
# numpy / pyscf below. Cloud VMs (and GitHub Actions runners) typically
# expose SMT-doubled vCPUs; BLAS / MKL / OpenMP default to one thread per
# logical CPU and contend for cache. Honor user overrides.
import glob
import os


def _physical_cores():
    try:
        ids = set()
        for path in glob.glob('/sys/devices/system/cpu/cpu*/topology/core_id'):
            with open(path) as f:
                ids.add(f.read().strip())
        if ids:
            return len(ids)
    except OSError:
        pass
    return os.cpu_count() or 1


_THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
if not any(v in os.environ for v in _THREAD_VARS):
    _n = str(_physical_cores())
    for _v in _THREAD_VARS:
        os.environ[_v] = _n
    del _n, _v
del _THREAD_VARS

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from berny import Berny, geomlib, optimize  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'birkholz_schlegel'


def run_pyscf(name, ref):
    from pyscf import dft, gto, scf
    from pyscf.geomopt import berny_solver

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
        raise ValueError(f'unsupported paper method {method!r}')
    state = {'n': 0}
    converged, _ = berny_solver.kernel(
        mf, callback=lambda loc: state.update(n=loc['cycle'] + 1)
    )
    return converged, state['n']


def run_mopac(name, ref):
    from berny.solvers import MopacSolver

    geom = geomlib.readfile(str(DATA / f'{name}.xyz'))
    berny = Berny(geom)
    optimize(berny, MopacSolver(charge=ref['charge'], mult=ref['mult']))
    return berny.converged, berny._n


def run_one(name, ref, kind):
    runner = run_pyscf if kind == 'pyscf' else run_mopac
    t0 = time.perf_counter()
    try:
        converged, n = runner(name, ref)
    except Exception as e:  # noqa: BLE001
        return {
            'name': name,
            'converged': False,
            'steps': None,
            'wall': time.perf_counter() - t0,
            'error': f'{type(e).__name__}: {e}',
        }
    return {
        'name': name,
        'converged': converged,
        'steps': n,
        'wall': time.perf_counter() - t0,
        'error': None,
    }


def format_table(rows, kind, reference):
    out = [
        f'| Molecule | Atoms | Paper | pyberny ({kind}) | Converged | Wall (s) |',
        '|---|---:|---:|---:|---|---:|',
    ]
    for row in rows:
        ref = reference[row['name']]
        paper = ref.get('paper_steps')
        out.append(
            f"| {row['name']} | {ref['atoms']} "
            f"| {'-' if paper is None else paper} "
            f"| {'-' if row['steps'] is None else row['steps']} "
            f"| {'yes' if row['converged'] else 'no'} "
            f"| {row['wall']:.1f} |"
        )
    return '\n'.join(out) + '\n'


def format_errors(rows):
    bad = [r for r in rows if r['error']]
    if not bad:
        return ''
    lines = ['', '### Failures', '']
    for row in bad:
        lines.append(f"- **{row['name']}**: {row['error']}")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--solver', choices=['pyscf', 'mopac'], required=True)
    ap.add_argument(
        '--molecules',
        nargs='*',
        default=None,
        help='subset of molecule names (default: all in reference.json)',
    )
    ap.add_argument('--out', type=Path, default=None, help='write markdown table here')
    args = ap.parse_args(argv)

    if args.solver == 'mopac' and not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')

    reference = json.loads((DATA / 'reference.json').read_text())
    names = args.molecules or sorted(reference)
    missing = [n for n in names if n not in reference]
    if missing:
        raise SystemExit(f'unknown molecules: {missing}')

    rows = []
    for name in names:
        print(f'==> {name}', flush=True)
        rows.append(run_one(name, reference[name], args.solver))

    table = format_table(rows, args.solver, reference)
    errors = format_errors(rows)
    if args.out:
        args.out.write_text(table + errors)
    print()
    print(table, end='')
    if errors:
        print(errors, end='')

    return 0 if all(r['converged'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
