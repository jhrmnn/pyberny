#!/usr/bin/env python3
"""Run the Birkholz-Schlegel 2016 benchmark and print a markdown summary.

Usage::

    scripts/benchmark.py --solver pyscf [--molecules NAME ...] [--out PATH]
    scripts/benchmark.py --solver mopac

PySCF mode drives the optimization through ``pyscf.geomopt.berny_solver``
and requires ``pip install pyberny[benchmark]``; MOPAC mode uses
:func:`berny.solvers.MopacSolver` (charge and multiplicity from
``reference.json``) and requires a ``mopac`` binary on ``$PATH``.
Molecules whose ``<solver>_steps`` reference value is ``null`` in
``reference.json`` are documented non-convergers / unmeasured and do not
contribute to the script's exit code.
"""

# Pin numeric-library thread counts to physical cores before importing
# numpy / pyscf below. Cloud VMs (and GitHub Actions runners) typically
# expose SMT-doubled vCPUs; BLAS / MKL / OpenMP default to one thread per
# logical CPU and contend for cache. Honor user overrides.
import glob
import os


def _physical_cores():
    # core_id is only unique within a CPU package; pair with physical_package_id
    # so multi-socket machines don't undercount.
    ids = set()
    for base in glob.glob('/sys/devices/system/cpu/cpu*/topology'):
        try:
            with open(f'{base}/core_id') as f:
                core = f.read().strip()
            with open(f'{base}/physical_package_id') as f:
                pkg = f.read().strip()
        except OSError:
            continue
        ids.add((pkg, core))
    return len(ids) or os.cpu_count() or 1


_n = None
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    # Each var honored independently: a user-set OMP_NUM_THREADS must not
    # leave MKL_NUM_THREADS / OPENBLAS_NUM_THREADS unpinned.
    if _v not in os.environ:
        if _n is None:
            _n = str(_physical_cores())
        os.environ[_v] = _n
del _n, _v

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


def _table_header(kind):
    return (
        f'| Molecule | Atoms | Paper | pyberny ({kind}) | Converged | Wall (s) |\n'
        '|---|---:|---:|---:|---|---:|\n'
    )


def _table_row(row, ref):
    paper = ref.get('paper_steps')
    return (
        f"| {row['name']} | {ref['atoms']} "
        f"| {'-' if paper is None else paper} "
        f"| {'-' if row['steps'] is None else row['steps']} "
        f"| {'yes' if row['converged'] else 'no'} "
        f"| {row['wall']:.1f} |\n"
    )


def format_table(rows, kind, reference):
    out = _table_header(kind)
    for row in rows:
        out += _table_row(row, reference[row['name']])
    return out


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
    ap.add_argument(
        '--out-json', type=Path, default=None, help='write per-row results as JSON'
    )
    ap.add_argument(
        '--append-md',
        type=Path,
        default=None,
        help='append table header + one row per molecule as they complete',
    )
    args = ap.parse_args(argv)

    if args.solver == 'mopac' and not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')

    reference = json.loads((DATA / 'reference.json').read_text())
    names = args.molecules or sorted(reference)
    missing = [n for n in names if n not in reference]
    if missing:
        raise SystemExit(f'unknown molecules: {missing}')

    if args.append_md:
        with open(args.append_md, 'a') as f:
            f.write(_table_header(args.solver))

    rows = []
    for name in names:
        print(f'==> {name}', flush=True)
        row = run_one(name, reference[name], args.solver)
        rows.append(row)
        if args.append_md:
            with open(args.append_md, 'a') as f:
                f.write(_table_row(row, reference[name]))

    table = format_table(rows, args.solver, reference)
    errors = format_errors(rows)
    if args.out:
        args.out.write_text(table + errors)
    if args.out_json:
        args.out_json.write_text(
            json.dumps({'solver': args.solver, 'rows': rows}, indent=2)
        )
    print()
    print(table, end='')
    if errors:
        print(errors, end='')

    # Treat documented-null reference entries (e.g. MOPAC's three
    # known non-convergers) as expected rather than failing the run.
    ref_key = {'mopac': 'mopac_pm7_steps', 'pyscf': 'pyberny_steps'}[args.solver]
    return (
        0
        if all(
            row['converged'] or reference[row['name']][ref_key] is None for row in rows
        )
        else 1
    )


if __name__ == '__main__':
    sys.exit(main())
