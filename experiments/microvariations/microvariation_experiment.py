#!/usr/bin/env python3
"""Probe pyberny's basin-of-attraction stability under small starting-geometry noise.

For each (molecule, sigma, seed) cell the script adds isotropic Gaussian noise
of standard deviation `sigma` (in angstrom) to every Cartesian coordinate of a
Birkholz-Schlegel benchmark structure, then runs ``Berny`` + ``MopacSolver``
(PM7) and records what happened. A sigma=0 baseline is also run per molecule
so that "did this end at the same minimum" can be checked against a concrete
reference structure rather than against `reference.json`'s step count alone.

Usage::

    experiments/microvariations/microvariation_experiment.py \
        [--out DIR] [--molecules M ...] [--sigmas S ...] \
        [--seeds N] [--maxsteps K]

Writes ``results.json`` and ``summary.md`` to ``--out`` (default
``experiments/microvariations``).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.solvers import MopacSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / 'src' / 'berny' / 'benchmarks' / 'birkholz_schlegel'

DEFAULT_MOLECULES = [
    'estradiol',
    'artemisinin',
    'vitamin_c',
    'codeine',
    'mg_porphin',
    'easc',
]
DEFAULT_SIGMAS = [0.001, 0.005, 0.01, 0.05]
DEFAULT_SEEDS = 10
DEFAULT_MAXSTEPS = 120


def perturb(geom, sigma, seed):
    """Return a deep copy of *geom* with iid Gaussian Cartesian noise."""
    rng = np.random.default_rng(seed)
    new_coords = geom.coords + rng.normal(scale=sigma, size=geom.coords.shape)
    return geomlib.Geometry(list(geom.species), new_coords, geom.lattice)


def kabsch_rmsd(a, b):
    """Return the RMSD between two coordinate arrays after optimal alignment.

    Both inputs are ``(N, 3)`` arrays in the same atom order. The RMSD is
    invariant to translation and rotation: each set is centered, then the
    optimal rotation is found via SVD (Kabsch).
    """
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_rot = a @ rot.T
    return float(np.sqrt(np.mean(np.sum((a_rot - b) ** 2, axis=1))))


def run_one(geom, ref, maxsteps):
    """Run one optimization, return ``(converged, n_steps, final_geom, energy)``."""
    berny = Berny(geom, maxsteps=maxsteps)
    solver = MopacSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    last_energy = None
    last_geom = geom
    for current in berny:
        last_geom = current
        energy, gradients = solver.send((list(current), current.lattice))
        last_energy = energy
        berny.send((energy, gradients))
    return berny.converged, berny._n, last_geom, last_energy


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        '--out', type=Path, default=REPO_ROOT / 'experiments' / 'microvariations'
    )
    ap.add_argument('--molecules', nargs='+', default=DEFAULT_MOLECULES)
    ap.add_argument('--sigmas', nargs='+', type=float, default=DEFAULT_SIGMAS)
    ap.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--maxsteps', type=int, default=DEFAULT_MAXSTEPS)
    ap.add_argument(
        '--resume',
        action='store_true',
        help='reuse rows from existing results.json instead of rerunning them',
    )
    return ap.parse_args(argv)


def summarize(rows, molecules, sigmas):
    """Render a per-molecule, per-sigma Markdown summary."""
    by_cell = {}
    baseline = {}
    for r in rows:
        if r['sigma'] == 0.0:
            baseline[r['molecule']] = r
            continue
        by_cell.setdefault((r['molecule'], r['sigma']), []).append(r)

    n_seeds = len({r['seed'] for r in rows if r['sigma'] != 0})
    out = []
    out.append('# Geometric micro-variation experiment\n')
    out.append(
        'Each row reports the optimizer outcome for one molecule under '
        'Gaussian noise of standard deviation `sigma` (angstrom) applied to '
        'every Cartesian coordinate of the Birkholz-Schlegel starting '
        'geometry. The PES is MOPAC PM7 (the same backend that produced the '
        '`mopac_pm7_steps` column of '
        '`src/berny/benchmarks/birkholz_schlegel/reference.json`). Each non-zero sigma '
        f'cell aggregates {n_seeds} seeds.\n'
    )
    out.append(
        '`conv` is the fraction of seeds that converged within `--maxsteps`. '
        '`steps` is the median step count over converged seeds (parenthetical '
        'min/max). `dE` is the maximum |final_energy - baseline_energy| in '
        'kcal/mol over converged seeds (kept in MOPAC\'s natural unit so the '
        'numbers are easy to read). `RMSD` is the maximum Kabsch-aligned '
        'final-structure RMSD vs the sigma=0 minimum in angstrom.\n'
    )

    for mol in molecules:
        base = baseline.get(mol)
        if base is None:
            continue
        out.append(f'\n## {mol}\n')
        conv_word = 'converged' if base['converged'] else 'did not converge'
        out.append(
            f'Baseline (sigma=0): {conv_word} in {base["steps"]} steps, '
            f'E = {base["energy"]:.3f} hartree.\n'
        )
        out.append(
            '| sigma (A) | conv | steps (min/max) | max dE (kcal/mol) | max RMSD (A) |'
        )
        out.append('|---:|---:|---:|---:|---:|')
        for sigma in sigmas:
            cell = by_cell.get((mol, sigma), [])
            if not cell:
                out.append(f'| {sigma} | - | - | - | - |')
                continue
            converged = [c for c in cell if c['converged']]
            conv_frac = f'{len(converged)}/{len(cell)}'
            if converged:
                steps = [c['steps'] for c in converged]
                steps_s = f'{int(statistics.median(steps))} ({min(steps)}/{max(steps)})'
                if base['energy'] is not None:
                    de = [
                        abs(c['energy'] - base['energy']) * 627.503 for c in converged
                    ]
                    de_s = f'{max(de):.3f}'
                else:
                    de_s = '-'
                rmsds = [
                    c['rmsd_vs_baseline']
                    for c in converged
                    if c['rmsd_vs_baseline'] is not None
                ]
                rmsd_s = f'{max(rmsds):.4f}' if rmsds else '-'
            else:
                steps_s = '-'
                de_s = '-'
                rmsd_s = '-'
            out.append(f'| {sigma} | {conv_frac} | {steps_s} | {de_s} | {rmsd_s} |')

    out.append('')
    out.append('## How to read this\n')
    out.append(
        'A flat row across `sigma` columns means the optimizer is insensitive '
        'to that scale of starting-geometry noise. Step count creeping up with '
        'sigma is the expected behaviour: a larger perturbation is farther '
        'from the minimum and farther outside the initial trust region. '
        'A drop in `conv` indicates seeds that hit the `--maxsteps` ceiling. '
        'A large `max dE` paired with a large `max RMSD` indicates at least '
        'one seed converged to a different minimum (or hit a saddle); the '
        'baseline column tells you what the "intended" minimum was.\n'
    )
    return '\n'.join(out) + '\n'


def _run_cell(mol, sigma, seed, ref, maxsteps, baseline_geoms):
    """Execute one (mol, sigma, seed) run, returning the row dict."""
    base_geom = geomlib.readfile(str(DATA / f'{mol}.xyz'))
    geom = base_geom if sigma == 0.0 else perturb(base_geom, sigma, seed)
    t0 = time.perf_counter()
    try:
        converged, n_steps, final_geom, energy = run_one(geom, ref, maxsteps)
        error = None
    except Exception as e:  # noqa: BLE001
        converged, n_steps, final_geom, energy = False, None, None, None
        error = f'{type(e).__name__}: {e}'
    wall = time.perf_counter() - t0

    rmsd = None
    final_coords = None
    if final_geom is not None:
        final_coords = final_geom.coords.tolist()
        if sigma == 0.0:
            baseline_geoms[mol] = final_geom.coords
        elif mol in baseline_geoms:
            rmsd = kabsch_rmsd(baseline_geoms[mol], final_geom.coords)

    return {
        'molecule': mol,
        'sigma': sigma,
        'seed': seed,
        'converged': converged,
        'steps': n_steps,
        'energy': energy,
        'rmsd_vs_baseline': rmsd,
        'wall': wall,
        'error': error,
        'final_coords': final_coords,
    }


def _persist(args, rows):
    """Write the incremental results.json and final_coords.json."""
    slim = [{k: v for k, v in r.items() if k != 'final_coords'} for r in rows]
    with open(args.out / 'results.json', 'w') as f:
        json.dump(
            {
                'molecules': args.molecules,
                'sigmas': args.sigmas,
                'seeds': args.seeds,
                'maxsteps': args.maxsteps,
                'rows': slim,
            },
            f,
            indent=2,
        )
    with open(args.out / 'final_coords.json', 'w') as f:
        json.dump(
            {
                f'{r["molecule"]}|{r["sigma"]}|{r["seed"]}': r['final_coords']
                for r in rows
            },
            f,
        )


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / 'results.json'

    reference = json.loads((DATA / 'reference.json').read_text())
    for mol in args.molecules:
        if mol not in reference:
            raise SystemExit(f'unknown molecule: {mol}')

    existing = {}
    if args.resume and results_path.exists():
        for row in json.loads(results_path.read_text())['rows']:
            existing[(row['molecule'], row['sigma'], row['seed'])] = row

    # Build the run plan: every (mol, sigma, seed) cell. sigma=0 uses seed=0.
    plan = []
    for mol in args.molecules:
        plan.append((mol, 0.0, 0))
        for sigma in args.sigmas:
            for seed in range(args.seeds):
                plan.append((mol, sigma, seed))

    baseline_geoms = {}  # mol -> final coords array (for RMSD)
    rows = []
    t_start = time.perf_counter()
    for i, (mol, sigma, seed) in enumerate(plan, 1):
        key = (mol, sigma, seed)
        if key in existing:
            row = existing[key]
            print(
                f'[{i}/{len(plan)}] {mol} sigma={sigma} seed={seed}: cached',
                flush=True,
            )
            rows.append(row)
            if sigma == 0.0 and row.get('final_coords') is not None:
                baseline_geoms[mol] = np.array(row['final_coords'])
            continue

        row = _run_cell(mol, sigma, seed, reference[mol], args.maxsteps, baseline_geoms)
        rows.append(row)
        elapsed = time.perf_counter() - t_start
        print(
            f'[{i}/{len(plan)}] {mol} sigma={sigma} seed={seed}: '
            f'converged={row["converged"]} steps={row["steps"]} '
            f'wall={row["wall"]:.1f}s (total {elapsed:.0f}s)',
            flush=True,
        )
        _persist(args, rows)

    summary = summarize(rows, args.molecules, args.sigmas)
    (args.out / 'summary.md').write_text(summary)
    print('\n' + summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
