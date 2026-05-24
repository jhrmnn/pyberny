#!/usr/bin/env python3
"""Run one of the geometry-optimization benchmarks and print a markdown summary.

Usage::

    scripts/benchmark.py --solver pyscf [--benchmark NAME]
                         [--molecules NAME ...] [--out PATH]
    scripts/benchmark.py --solver mopac

``--benchmark`` selects the molecule set under ``tests/data/`` -- either
``birkholz`` (the Birkholz-Schlegel 2016 19-molecule set, the default) or
``baker`` (the 30-molecule Baker set from Shajan et al., chemrxiv 2023).
PySCF mode drives the optimization through ``pyscf.geomopt.berny_solver``
and requires ``pip install pyberny[benchmark]``; MOPAC mode uses
:func:`berny.solvers.MopacSolver` (charge and multiplicity from
``reference.json``) and requires a ``mopac`` binary on ``$PATH``.
A molecule fails the run when it either does not converge or its step
count drifts from the reference by more than 7% (with an absolute floor
of 2 steps, so the gate stays meaningful for small references);
molecules whose ``<solver>_steps`` reference value is ``null`` in
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

from berny import Berny, geomlib  # noqa: E402

_DATA_ROOT = Path(__file__).resolve().parents[1] / 'tests' / 'data'
BENCHMARKS = {
    'birkholz': _DATA_ROOT / 'birkholz_schlegel',
    'baker': _DATA_ROOT / 'baker_shajan_2023',
}
# Default exposed for backward compatibility: external callers (and
# aggregate_benchmark.py) used to import ``DATA`` directly.
DATA = BENCHMARKS['birkholz']


def run_pyscf(name, ref, data_dir):
    from pyscf import dft, gto, scf
    from pyscf.geomopt import berny_solver

    mol = gto.M(
        atom=str(data_dir / f'{name}.xyz'),
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
    state = {'n': 0, 'energies': []}

    def callback(loc):
        state['n'] = loc['cycle'] + 1
        energy = loc.get('energy')
        if energy is not None:
            state['energies'].append(float(energy))

    converged, _ = berny_solver.kernel(mf, callback=callback)
    return converged, state['n'], state['energies']


def _optimize_recording_energies(berny, solver):
    """Drive ``berny`` with ``solver`` and collect per-step energies."""
    next(solver)
    energies = []
    for geom in berny:
        energy, gradients = solver.send((list(geom), geom.lattice))
        energies.append(energy)
        berny.send((energy, gradients))
    return energies


def run_mopac(name, ref, data_dir):
    from berny.solvers import MopacSolver

    geom = geomlib.readfile(str(data_dir / f'{name}.xyz'))
    # A couple of molecules (raffinose, sphingomyelin) need more than
    # pyberny's default 100-step ceiling under MOPAC PM7 on CI; raise it
    # so they still have a chance to converge and be reported. Both are
    # documented non-convergers in reference.json (mopac_pm7_steps=null)
    # so the regression gate ignores them either way.
    berny = Berny(geom, maxsteps=110)
    solver = MopacSolver(charge=ref['charge'], mult=ref['mult'])
    energies = _optimize_recording_energies(berny, solver)
    return berny.converged, berny._n, energies


def run_one(name, ref, kind, data_dir):
    runner = run_pyscf if kind == 'pyscf' else run_mopac
    t0 = time.perf_counter()
    try:
        converged, n, energies = runner(name, ref, data_dir)
    except Exception as e:  # noqa: BLE001
        return {
            'name': name,
            'converged': False,
            'steps': None,
            'wall': time.perf_counter() - t0,
            'energies': [],
            'error': f'{type(e).__name__}: {e}',
        }
    return {
        'name': name,
        'converged': converged,
        'steps': n,
        'wall': time.perf_counter() - t0,
        'energies': energies,
        'error': None,
    }


REF_STEPS_KEY = {'mopac': 'mopac_pm7_steps', 'pyscf': 'pyberny_steps'}


def format_table(rows, kind, reference):
    """Render the per-molecule markdown table.

    Columns: molecule, atoms, paper (SM reference from Birkholz–Schlegel),
    ref (this solver's regression baseline from ``reference.json``), the
    measured pyberny step count, convergence flag, and wall time.

    Rows whose ``wall`` is ``None`` are treated as "not run" placeholders
    (used by ``aggregate_benchmark.py`` to keep the table shape stable
    across partial CI runs); they render as ``-`` across the measured
    columns but still occupy a line so the molecule list is always
    complete.
    """
    ref_key = REF_STEPS_KEY[kind]
    out = [
        f'| Molecule | Atoms | Paper | Ref ({kind}) '
        f'| pyberny ({kind}) | Converged | Wall (s) |',
        '|---|---:|---:|---:|---:|---|---:|',
    ]
    for row in rows:
        ref = reference[row['name']]
        paper = ref.get('paper_steps')
        ref_steps = ref.get(ref_key)
        not_run = row.get('wall') is None
        steps_s = '-' if not_run or row['steps'] is None else str(row['steps'])
        conv_s = '-' if not_run else ('yes' if row['converged'] else 'no')
        wall_s = '-' if not_run else f"{row['wall']:.1f}"
        out.append(
            f"| {row['name']} | {ref['atoms']} "
            f"| {'-' if paper is None else paper} "
            f"| {'-' if ref_steps is None else ref_steps} "
            f"| {steps_s} "
            f"| {conv_s} "
            f"| {wall_s} |"
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


def regression_reason(row, ref):
    """Return ``None`` if the row passes the regression gate, else a reason.

    ``ref`` is the reference step count for the row's solver (or ``None`` if
    no baseline is recorded, in which case the gate is skipped). A row fails
    if it didn't converge, or if its step count drifted from ``ref`` by more
    than ``max(2, round(0.07 * ref))`` steps — a 7% band, floored at 2 so
    the gate stays meaningful for small references.
    """
    if ref is None:
        return None
    if not row['converged']:
        return 'did not converge'
    drift = row['steps'] - ref
    tolerance = max(2, round(0.07 * ref))
    if abs(drift) > tolerance:
        return f"{row['steps']} steps vs ref {ref} ({drift:+d}, tol {tolerance})"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--solver', choices=['pyscf', 'mopac'], required=True)
    ap.add_argument(
        '--benchmark',
        choices=sorted(BENCHMARKS),
        default='birkholz',
        help='which molecule set to run (default: birkholz)',
    )
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
    args = ap.parse_args(argv)

    if args.solver == 'mopac' and not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')

    data_dir = BENCHMARKS[args.benchmark]
    reference = json.loads((data_dir / 'reference.json').read_text())
    names = args.molecules or sorted(reference)
    missing = [n for n in names if n not in reference]
    if missing:
        raise SystemExit(f'unknown molecules: {missing}')

    rows = []
    for name in names:
        print(f'==> {name}', flush=True)
        rows.append(run_one(name, reference[name], args.solver, data_dir))

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

    # Treat documented-null reference entries (e.g. MOPAC's one
    # known non-converger) as expected rather than failing the run.
    ref_key = REF_STEPS_KEY[args.solver]
    regressions = [
        (row['name'], regression_reason(row, reference[row['name']][ref_key]))
        for row in rows
    ]
    regressions = [(n, r) for n, r in regressions if r]
    if regressions:
        print('\nBenchmark regressions:', file=sys.stderr)
        for n, reason in regressions:
            print(f'  {n}: {reason}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
