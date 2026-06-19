#!/usr/bin/env python3
"""Probe how stable a benchmark's optimizations are under noise in the start geometry.

For every molecule in a benchmark this perturbs the starting Cartesian
coordinates with isotropic Gaussian noise of several amplitudes, re-runs
the optimization from many independent random seeds, and records for each
trial whether it converged, how many steps it took, and the final energy.

Two questions are then answered per molecule and in aggregate:

* **Convergence stability** -- does displacing the start cost extra steps,
  and does it ever break convergence within the step ceiling?
* **Minimum stability** -- do the noisy starts all relax back to the same
  minimum (final energy within a tight window of the unperturbed run), or
  does noise occasionally tip a molecule into a different basin?

The unperturbed (sigma = 0) run is included as the reference trial for each
molecule. Energies are compared in kcal/mol relative to that reference.

Usage::

    scripts/noise_stability.py --benchmark baker \
        --seeds 8 --sigmas 0.01 0.03 0.05 0.1 \
        --out noise_report.md --out-json noise_raw.json

The optimizer is driven by GFN2-xTB (via :class:`berny.solvers.XTBSolver`),
which is fast enough to sweep the whole Baker set across many seeds and
amplitudes in well under a minute.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules, load_reference

# Hartree -> kcal/mol, for reporting energy spreads in chemically intuitive units.
HARTREE_KCAL = 627.5094740631

# A noisy start counts as "same minimum" if its converged energy lands within
# this window of the unperturbed reference energy. 0.1 kcal/mol is far tighter
# than the spacing between distinct conformers/minima yet loose enough to
# absorb the residual gradient left at pyberny's convergence thresholds.
SAME_MIN_KCAL = 0.1


def _optimize(geom, ref, maxsteps):
    """Run one GFN2-xTB optimization from ``geom``.

    Returns a dict with ``status`` one of:

    * ``'converged'`` -- met pyberny's convergence criteria within
      ``maxsteps``;
    * ``'unconverged'`` -- ran out the ``maxsteps`` ceiling without
      converging;
    * ``'error'`` -- raised before finishing. The dominant error here is a
      :class:`berny.coords.CoordinateError` from building internal
      coordinates on a noise-distorted geometry (e.g. a near-linear chain
      that the redundant-internal builder can't form dihedrals through);
      the optimization never even starts.

    plus ``steps`` and ``energy`` (the final energy, or ``None`` on error).
    """
    from berny.solvers import XTBSolver

    try:
        berny = Berny(geom, maxsteps=maxsteps)
        solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
        next(solver)
        energy = None
        for g in berny:
            energy, gradients = solver.send((list(g), g.lattice))
            berny.send((energy, gradients))
    except Exception as e:
        return {
            'status': 'error',
            'steps': None,
            'energy': None,
            'error': f'{type(e).__name__}: {e}',
        }
    return {
        'status': 'converged' if berny.converged else 'unconverged',
        'steps': berny._n,
        'energy': energy,
        'error': None,
    }


def _perturb(geom, sigma, rng):
    """Return a copy of ``geom`` with iid Gaussian noise (sigma, in Angstrom)
    added to every Cartesian coordinate."""
    return geomlib.Geometry(
        list(geom.species),
        geom.coords + rng.normal(0.0, sigma, size=geom.coords.shape),
        geom.lattice,
    )


def run_molecule(name, geom, ref, sigmas, seeds, maxsteps):
    """Run the reference plus the full noise sweep for one molecule."""
    # Reference (unperturbed) optimization.
    res0 = _optimize(geom, ref, maxsteps)
    e0 = res0['energy']
    trials = [
        {
            'sigma': 0.0,
            'seed': None,
            'denergy_kcal': 0.0 if e0 is not None else None,
            **res0,
        }
    ]
    for sigma in sigmas:
        for seed in range(seeds):
            # Independent, reproducible stream per (molecule, sigma, seed).
            rng = np.random.default_rng(
                abs(hash((name, round(sigma, 6), seed))) % (2**32)
            )
            noisy = _perturb(geom, sigma, rng)
            res = _optimize(noisy, ref, maxsteps)
            e = res['energy']
            de = (e - e0) * HARTREE_KCAL if (e is not None and e0 is not None) else None
            trials.append({'sigma': sigma, 'seed': seed, 'denergy_kcal': de, **res})
    return trials


def summarize_molecule(name, ref, trials):
    """Reduce a molecule's trials to per-sigma and overall stability stats."""
    ref_trial = trials[0]
    noisy = [t for t in trials if t['seed'] is not None]
    by_sigma = {}
    for t in noisy:
        by_sigma.setdefault(t['sigma'], []).append(t)

    per_sigma = []
    for sigma in sorted(by_sigma):
        ts = by_sigma[sigma]
        n_conv = [t for t in ts if t['status'] == 'converged']
        n_err = [t for t in ts if t['status'] == 'error']
        steps = [t['steps'] for t in n_conv]
        # Energy spread among the runs that converged AND found the ref basin.
        same_basin = [
            t
            for t in n_conv
            if t['denergy_kcal'] is not None and abs(t['denergy_kcal']) <= SAME_MIN_KCAL
        ]
        diff_basin = [
            t
            for t in n_conv
            if t['denergy_kcal'] is not None and abs(t['denergy_kcal']) > SAME_MIN_KCAL
        ]
        max_dabs = max(
            (abs(t['denergy_kcal']) for t in n_conv if t['denergy_kcal'] is not None),
            default=0.0,
        )
        per_sigma.append(
            {
                'sigma': sigma,
                'n': len(ts),
                'n_converged': len(n_conv),
                'n_error': len(n_err),
                'n_unconverged': len(ts) - len(n_conv) - len(n_err),
                'n_same_basin': len(same_basin),
                'n_diff_basin': len(diff_basin),
                'steps_min': min(steps) if steps else None,
                'steps_max': max(steps) if steps else None,
                'steps_mean': statistics.mean(steps) if steps else None,
                'max_denergy_kcal': max_dabs,
            }
        )

    n_conv_all = sum(1 for t in noisy if t['status'] == 'converged')
    n_err_all = sum(1 for t in noisy if t['status'] == 'error')
    diff_all = sum(
        1
        for t in noisy
        if t['status'] == 'converged'
        and t['denergy_kcal'] is not None
        and abs(t['denergy_kcal']) > SAME_MIN_KCAL
    )
    error_examples = sorted(
        {t['error'] for t in noisy if t['status'] == 'error' and t['error']}
    )
    return {
        'name': name,
        'atoms': ref['atoms'],
        'ref_status': ref_trial['status'],
        'ref_steps': ref_trial['steps'],
        'n_trials': len(noisy),
        'n_converged': n_conv_all,
        'n_error': n_err_all,
        'n_unconverged': len(noisy) - n_conv_all - n_err_all,
        'n_diff_basin': diff_all,
        'per_sigma': per_sigma,
        '_error_examples': error_examples,
    }


def format_report(summaries, sigmas, seeds, kind, benchmark, wall):
    lines = []
    lines.append(f'# Noise-stability sweep: {benchmark} / {kind}')
    lines.append('')
    lines.append(
        f'- Noise amplitudes (Angstrom RMS per Cartesian coord): '
        f"{', '.join(str(s) for s in sigmas)}"
    )
    lines.append(f'- Seeds per (molecule, amplitude): {seeds}')
    lines.append(f'- "Same minimum" window: |E - E_ref| <= {SAME_MIN_KCAL} kcal/mol')
    lines.append(f'- Wall time: {wall:.1f} s')
    lines.append('')

    # Aggregate headline numbers.
    total_trials = sum(s['n_trials'] for s in summaries)
    total_conv = sum(s['n_converged'] for s in summaries)
    total_err = sum(s['n_error'] for s in summaries)
    total_unconv = sum(s['n_unconverged'] for s in summaries)
    total_diff = sum(s['n_diff_basin'] for s in summaries)
    mol_diff = [s['name'] for s in summaries if s['n_diff_basin'] > 0]
    mol_err = [s['name'] for s in summaries if s['n_error'] > 0]
    mol_unconv = [s['name'] for s in summaries if s['n_unconverged'] > 0]
    lines.append('## Headline')
    lines.append('')
    lines.append(
        f'- Noisy trials: {total_trials}; converged: {total_conv} '
        f'({100 * total_conv / total_trials:.1f}%), '
        f'hit step ceiling: {total_unconv} '
        f'({100 * total_unconv / total_trials:.1f}%), '
        f'errored before optimizing: {total_err} '
        f'({100 * total_err / total_trials:.1f}%)'
    )
    lines.append(
        f'- Converged trials landing in a different basin than the unperturbed '
        f'run: {total_diff} ({100 * total_diff / total_trials:.2f}%)'
    )
    lines.append(
        '- Molecules with any coordinate-build/other error: '
        + (', '.join(mol_err) if mol_err else 'none')
    )
    lines.append(
        '- Molecules with any step-ceiling non-convergence: '
        + (', '.join(mol_unconv) if mol_unconv else 'none')
    )
    lines.append(
        '- Molecules with any different-basin outcome: '
        + (', '.join(mol_diff) if mol_diff else 'none')
    )
    lines.append('')

    # Per-amplitude aggregate.
    lines.append('## Stability vs noise amplitude (aggregate over all molecules)')
    lines.append('')
    lines.append(
        '| sigma (A) | trials | converged | ceiling | error '
        '| same basin | diff basin | max |dE| (kcal/mol) |'
    )
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for sigma in sigmas:
        trials = conv = unconv = err = same = diff = 0
        maxd = 0.0
        for s in summaries:
            for ps in s['per_sigma']:
                if ps['sigma'] == sigma:
                    trials += ps['n']
                    conv += ps['n_converged']
                    unconv += ps['n_unconverged']
                    err += ps['n_error']
                    same += ps['n_same_basin']
                    diff += ps['n_diff_basin']
                    maxd = max(maxd, ps['max_denergy_kcal'])
        lines.append(
            f'| {sigma} | {trials} | {conv} | {unconv} | {err} '
            f'| {same} | {diff} | {maxd:.3f} |'
        )
    lines.append('')

    # Per-molecule table.
    lines.append('## Per-molecule summary')
    lines.append('')
    lines.append(
        '| Molecule | Atoms | Ref steps | Noisy trials | Converged | Ceiling '
        '| Error | Diff basin | Steps (min/mean/max) | Max |dE| (kcal/mol) |'
    )
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|---:|')
    for s in sorted(
        summaries,
        key=lambda x: (
            -(x['n_error'] + x['n_unconverged']),
            -x['n_diff_basin'],
            x['name'],
        ),
    ):
        steps_min = min(
            (ps['steps_min'] for ps in s['per_sigma'] if ps['steps_min']),
            default=None,
        )
        steps_max = max(
            (ps['steps_max'] for ps in s['per_sigma'] if ps['steps_max']),
            default=None,
        )
        means = [ps['steps_mean'] for ps in s['per_sigma'] if ps['steps_mean']]
        steps_mean = statistics.mean(means) if means else None
        maxd = max((ps['max_denergy_kcal'] for ps in s['per_sigma']), default=0.0)
        steps_s = (
            f'{steps_min}/{steps_mean:.1f}/{steps_max}'
            if steps_mean is not None
            else '-'
        )
        lines.append(
            f"| {s['name']} | {s['atoms']} | {s['ref_steps']} "
            f"| {s['n_trials']} | {s['n_converged']} | {s['n_unconverged']} "
            f"| {s['n_error']} | {s['n_diff_basin']} "
            f"| {steps_s} | {maxd:.3f} |"
        )
    lines.append('')

    # List representative error messages so the failure mode is visible.
    err_examples = {}
    for s in summaries:
        if s['n_error'] == 0:
            continue
        for t in s.get('_error_examples', []):
            err_examples.setdefault(t, s['name'])
    if err_examples:
        lines.append('## Representative errors')
        lines.append('')
        for msg, mol in sorted(err_examples.items(), key=lambda kv: kv[1]):
            lines.append(f'- **{mol}**: `{msg}`')
        lines.append('')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--benchmark', default='baker')
    ap.add_argument('--solver', choices=['xtb'], default='xtb')
    ap.add_argument('--molecules', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument(
        '--sigmas',
        type=float,
        nargs='*',
        default=[0.01, 0.03, 0.05, 0.1],
        help='noise amplitudes in Angstrom (RMS per Cartesian coordinate)',
    )
    ap.add_argument('--maxsteps', type=int, default=100)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--out-json', type=Path, default=None)
    args = ap.parse_args(argv)

    reference = load_reference(args.benchmark)
    names = args.molecules or sorted(reference)

    t0 = time.perf_counter()
    summaries = []
    raw = {}
    for name, geom, ref in iter_molecules(args.benchmark, names):
        print(f'==> {name}', flush=True)
        trials = run_molecule(name, geom, ref, args.sigmas, args.seeds, args.maxsteps)
        raw[name] = trials
        summaries.append(summarize_molecule(name, ref, trials))
    wall = time.perf_counter() - t0

    report = format_report(
        summaries, args.sigmas, args.seeds, args.solver, args.benchmark, wall
    )
    if args.out:
        args.out.write_text(report)
    if args.out_json:
        args.out_json.write_text(
            json.dumps(
                {
                    'benchmark': args.benchmark,
                    'solver': args.solver,
                    'sigmas': args.sigmas,
                    'seeds': args.seeds,
                    'raw': raw,
                    'summaries': summaries,
                },
                indent=2,
            )
        )
    print()
    print(report, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
