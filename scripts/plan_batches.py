#!/usr/bin/env python3
"""Offline tool: bin-pack the Birkholz-Schlegel benchmark into parallel batches.

For each solver, this script times one energy+gradient call per molecule
(``mopac`` with ``1SCF GRADIENTS``; ``pyscf`` with one SCF + analytic
gradient), multiplies by the pyberny step count from ``reference.json``,
and prints a YAML fragment ready to paste into the
``strategy.matrix.include`` block of ``.github/workflows/benchmark.yaml``.

The per-call timing is the only cost source — there is no analytical
N**p fallback. Per-iteration cost varies by an order of magnitude across
this dataset (``easc`` with Al/Cl is ~35x slower than CHNO organics of
similar size), and no closed-form expression in ``atoms`` captures it.

``pyberny_steps`` is ``null`` for every entry in ``reference.json``, so
``paper_steps`` is the only available step proxy for pyscf.
``mopac_pm7_steps`` is ``null`` for one molecule (``bisphenol_a``); the
median over the rest is used.

Bin-packing: Longest-Processing-Time-first (LPT). ``--nbins`` defaults
to ``ceil(total / max_single)`` so the heaviest molecule does not
bottleneck the run.

``--exclude NAME`` drops a molecule from planning entirely (e.g.
``azadirachtin`` for ``pyscf``, where one optimization alone exceeds
GitHub Actions' 360-min job cap). Run the script once per solver if
the exclusions or ``--nbins`` differ between them.

Cache the measurements to skip the (slow) pyscf timings on re-runs::

    python scripts/plan_batches.py --cache /tmp/runtime.json
"""

import argparse
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'birkholz_schlegel'

STEPS_KEY = {'mopac': 'mopac_pm7_steps', 'pyscf': 'pyberny_steps'}
# Used only if STEPS_KEY value is null for a given molecule.
FALLBACK_STEPS_KEY = {'mopac': 'mopac_pm7_steps', 'pyscf': 'paper_steps'}
REPEATS = {'mopac': 3, 'pyscf': 1}


def _pin_threads():
    """Pin BLAS/OMP to physical cores to match CI runner config."""
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
    n = str(len(ids) or os.cpu_count() or 1)
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(v, n)


def _measure_mopac(name, ref):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from berny import geomlib

    geom = geomlib.readfile(str(DATA / f'{name}.xyz'))
    atoms = list(geom)
    mults = {1: '', 2: 'DOUBLET', 3: 'TRIPLET', 4: 'QUARTET', 5: 'QUINTET'}
    kw = ['PM7', '1SCF', 'GRADIENTS']
    if ref['charge']:
        kw.append(f'CHARGE={ref["charge"]}')
    if mults[ref['mult']]:
        kw += [mults[ref['mult']], 'UHF']
    body = (
        ' '.join(kw)
        + '\n\n\n'
        + '\n'.join(f'{el} {x} 1 {y} 1 {z} 1' for el, (x, y, z) in atoms)
    )
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / 'job.mop'
        f.write_text(body)
        t0 = time.perf_counter()
        subprocess.check_call(['mopac', str(f)], stdout=subprocess.DEVNULL)
        return time.perf_counter() - t0


def _measure_pyscf(name, ref):
    from pyscf import dft, gto, scf
    from pyscf.grad import (
        rhf as rhf_grad,
        rks as rks_grad,
        uhf as uhf_grad,
        uks as uks_grad,
    )

    mol = gto.M(
        atom=str(DATA / f'{name}.xyz'),
        basis=ref['paper_steps_basis'],
        charge=ref['charge'],
        spin=ref['mult'] - 1,
        verbose=0,
    )
    method = ref['paper_steps_method']
    closed = ref['mult'] == 1
    if method == 'HF':
        mf = scf.RHF(mol) if closed else scf.UHF(mol)
        grad_cls = rhf_grad.Gradients if closed else uhf_grad.Gradients
    elif method == 'B3LYP':
        mf = dft.RKS(mol) if closed else dft.UKS(mol)
        mf.xc = 'b3lyp'
        grad_cls = rks_grad.Gradients if closed else uks_grad.Gradients
    else:
        raise ValueError(f'unsupported paper method: {method!r}')
    t0 = time.perf_counter()
    mf.kernel()
    grad_cls(mf).kernel()
    return time.perf_counter() - t0


MEASURE = {'mopac': _measure_mopac, 'pyscf': _measure_pyscf}


def _step_count(ref, solver, fallback):
    primary = ref[STEPS_KEY[solver]]
    if primary is not None:
        return primary
    secondary = ref.get(FALLBACK_STEPS_KEY[solver])
    if secondary is not None:
        return secondary
    return fallback


def measure(reference, solver, cache, cache_path=None):
    measure_fn = MEASURE[solver]
    repeats = REPEATS[solver]
    per_call = {}
    for name in sorted(reference):
        key = f'{solver}/{name}'
        if key in cache:
            per_call[name] = cache[key]
            print(f'  {name:25s} (cached) {per_call[name]:.3f}s', file=sys.stderr)
            continue
        ts = sorted(measure_fn(name, reference[name]) for _ in range(repeats))
        per_call[name] = ts[len(ts) // 2]
        cache[key] = per_call[name]
        if cache_path is not None:
            # Flush after every measurement so a crash mid-pyscf doesn't lose
            # 30 minutes of work.
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        print(f'  {name:25s} per-call={per_call[name]:.3f}s', file=sys.stderr)
    return per_call


def costs(reference, solver, per_call):
    fb_key = FALLBACK_STEPS_KEY[solver]
    fb_vals = [
        reference[n][fb_key] for n in reference if reference[n][fb_key] is not None
    ]
    fallback = statistics.median(fb_vals)
    return {
        n: per_call[n] * _step_count(reference[n], solver, fallback) for n in reference
    }


def pack(items, nbins):
    bins = [[] for _ in range(nbins)]
    totals = [0.0] * nbins
    for name, cost in sorted(items.items(), key=lambda kv: (-kv[1], kv[0])):
        i = min(range(nbins), key=lambda j: (totals[j], j))
        bins[i].append(name)
        totals[i] += cost
    return bins, totals


def plan(reference, solver, nbins, per_call):
    c = costs(reference, solver, per_call)
    total = sum(c.values())
    biggest = max(c.values())
    if nbins is None:
        nbins = max(1, min(8, math.ceil(total / biggest)))
    bins, totals = pack(c, nbins)
    for b in bins:
        b.sort(key=lambda n: reference[n]['atoms'])
    return bins, totals, nbins


def emit(reference, solvers, nbins_arg, cache, cache_path=None, exclude=()):
    out = []
    out.append('# Generated by scripts/plan_batches.py — do not hand-edit.')
    out.append('# Cost = measured single energy+gradient call × pyberny step count.')
    exclude = set(exclude)
    for solver in solvers:
        sub_ref = {n: v for n, v in reference.items() if n not in exclude}
        print(f'\nmeasuring {solver} …', file=sys.stderr)
        per_call = measure(sub_ref, solver, cache, cache_path)
        bins, totals, nbins = plan(sub_ref, solver, nbins_arg, per_call)
        mean = sum(totals) / nbins
        spread = (max(totals) - min(totals)) / mean if mean else 0.0
        out.append(
            f'# {solver}: {nbins} batches, cost spread = {spread:.0%} of mean '
            f'(total {sum(totals):.0f}s, heaviest bin {max(totals):.0f}s)'
        )
        for i, names in enumerate(bins):
            mols = ' '.join(names)
            out.append(f'- {{ solver: {solver}, batch_id: b{i}, molecules: {mols} }}')
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--reference', type=Path, default=DATA / 'reference.json')
    ap.add_argument('--solvers', nargs='+', default=['mopac', 'pyscf'])
    ap.add_argument(
        '--nbins',
        type=int,
        default=None,
        help='override per-solver bin count (default: ceil(total/max_single))',
    )
    ap.add_argument(
        '--cache',
        type=Path,
        default=None,
        help='JSON cache of {solver/name: seconds_per_call} '
        '(loaded if exists, written after measurements)',
    )
    ap.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='molecule names to exclude from planning (applies to every '
        '--solvers entry; rerun per solver if exclusions differ)',
    )
    args = ap.parse_args(argv)
    _pin_threads()
    if 'mopac' in args.solvers and not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')
    reference = json.loads(args.reference.read_text())
    unknown = [n for n in args.exclude if n not in reference]
    if unknown:
        raise SystemExit(f'unknown --exclude entries: {unknown}')
    cache = (
        json.loads(args.cache.read_text()) if args.cache and args.cache.exists() else {}
    )
    print(
        emit(reference, args.solvers, args.nbins, cache, args.cache, args.exclude),
        end='',
    )


if __name__ == '__main__':
    main()
