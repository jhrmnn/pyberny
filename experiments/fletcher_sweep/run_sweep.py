#!/usr/bin/env python3
"""Sweep Fletcher trust-radius parameters on the fast CI benchmarks.

For each (parameter setting, benchmark, molecule) cell we run
``Berny + MopacSolver`` and record, per step, Fletcher's parameter

    r = dE_actual / dE_predicted

A step counts as a *good quadratic step* when ``r`` lies in the
``[good_lo, good_hi]`` band around 1 (default ``[0.75, 1.25]``).
We also track the share of steps Fletcher classifies as

    - shrink     (r < low_thr)
    - keep       (low_thr <= r <= high_thr)
    - grow       (r > high_thr and step was on the trust sphere)
    - below_noise (|dE_predicted| < 10 * energy_noise; r is suppressed)

Then we vary ``low_thr`` / ``high_thr`` / shrink & grow factors of the
trust-update rule and re-run, so the feedback effect on later steps is
captured (changing the thresholds changes the trust radius trajectory,
which changes future ``r`` values).

Writes ``results.json`` and ``summary.md`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import BENCHMARKS
from berny.solvers import MopacSolver
import berny.berny as berny_mod


# ---------------------------------------------------------------------------
# Configurable Fletcher rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FletcherParams:
    """Tunable knobs of pyberny's Fletcher trust-update rule."""

    name: str
    low_thr: float = 0.25          # r < low_thr  -> shrink
    high_thr: float = 0.75         # r > high_thr -> grow (if on sphere)
    shrink_to: float = 0.25        # new_trust = shrink_to * |dq|  (default 1/4)
    grow_factor: float = 2.0       # new_trust = grow_factor * trust
    energy_noise: float = 2e-8     # |dE_pred| < 10*noise -> suppress


def make_update_trust(params: FletcherParams) -> Callable[..., float]:
    """Build a drop-in replacement for ``berny.berny.update_trust``."""
    from numpy.linalg import norm

    def update_trust(
        trust, dE, dE_predicted, dq,
        log=berny_mod.no_log, *,
        energy_noise: float = 2e-8,   # ignored; we use params.energy_noise
        record=None,
    ):
        if abs(dE_predicted) < 10 * params.energy_noise:
            if abs(norm(dq) - trust) < 1e-10:
                new_trust = params.grow_factor * trust
            else:
                new_trust = trust
            if record is not None:
                record['trust_update'] = {
                    'fletcher': None,
                    'trust': float(new_trust),
                    'below_noise': True,
                }
            return new_trust
        r = dE / dE_predicted if dE != 0 else 1.0
        log(f"Trust update: Fletcher's parameter: {r:.3}")
        if r < params.low_thr:
            new_trust = params.shrink_to * float(norm(dq))
        elif r > params.high_thr and abs(norm(dq) - trust) < 1e-10:
            new_trust = params.grow_factor * trust
        else:
            new_trust = trust
        if record is not None:
            record['trust_update'] = {
                'fletcher': float(r),
                'trust': float(new_trust),
                'below_noise': False,
            }
        return new_trust

    update_trust._energy_noise = params.energy_noise  # for the collector wrapper
    return update_trust


# ---------------------------------------------------------------------------
# Recording driver
# ---------------------------------------------------------------------------


def run_one(name: str, ref: dict, data_dir: Path, maxsteps: int) -> dict:
    """Run one molecule and return per-step Fletcher records + summary."""
    geom = geomlib.readfile(str(data_dir / f'{name}.xyz'))
    records: list[dict] = []

    # Wrap whatever update_trust is currently installed on the module so we
    # also collect a per-step record. We can't piggy-back on Berny's own
    # ``record`` argument because that's only populated when ``trace=`` was
    # passed to Berny -- and trace= would force a JSON dump per step, per
    # molecule, per sweep cell.
    base_update = berny_mod.update_trust

    def update_trust_collecting(trust, dE, dE_predicted, dq, *args, **kw):
        new_trust = base_update(trust, dE, dE_predicted, dq, *args, **kw)
        # Recompute the classification inputs (cheap) so we don't depend on
        # ``record`` being passed in by Berny.
        from numpy.linalg import norm
        # The "below_noise" branch in base_update uses its own configured
        # energy_noise; we read it back from the bound FletcherParams if
        # present, otherwise fall back to the stock default.
        en = getattr(base_update, '_energy_noise', 2e-8)
        if abs(dE_predicted) < 10 * en:
            records.append({
                'fletcher': None, 'trust': float(new_trust),
                'below_noise': True, 'dq_norm': float(norm(dq)),
                'old_trust': float(trust),
            })
        else:
            r = dE / dE_predicted if dE != 0 else 1.0
            records.append({
                'fletcher': float(r), 'trust': float(new_trust),
                'below_noise': False, 'dq_norm': float(norm(dq)),
                'old_trust': float(trust),
            })
        return new_trust

    berny_mod.update_trust = update_trust_collecting
    try:
        berny = Berny(geom, maxsteps=maxsteps)
        solver = MopacSolver(charge=ref['charge'], mult=ref['mult'])
        next(solver)
        for g in berny:
            energy, gradients = solver.send((list(g), g.lattice))
            berny.send((energy, gradients))
    finally:
        berny_mod.update_trust = base_update

    return {
        'converged': bool(berny.converged),
        'steps': berny._n,
        'records': records,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def classify(rec: dict, params: FletcherParams, good_lo: float, good_hi: float) -> str:
    if rec.get('below_noise'):
        return 'below_noise'
    r = rec['fletcher']
    if r is None:
        return 'below_noise'
    if r < params.low_thr:
        bucket = 'shrink'
    elif r > params.high_thr:
        bucket = 'grow_or_keep'  # Fletcher only grows if on-sphere; can't tell
    else:
        bucket = 'keep'
    return bucket


def summarize(per_mol: dict[str, dict], params: FletcherParams,
              good_lo: float, good_hi: float) -> dict:
    all_r: list[float] = []
    n_good = n_total_with_r = n_noise = 0
    bucket_counts = {'shrink': 0, 'keep': 0, 'grow_or_keep': 0, 'below_noise': 0}
    total_steps = 0
    converged = 0
    nmol = len(per_mol)
    for name, res in per_mol.items():
        total_steps += res['steps']
        if res['converged']:
            converged += 1
        for rec in res['records']:
            b = classify(rec, params, good_lo, good_hi)
            bucket_counts[b] += 1
            if b == 'below_noise':
                n_noise += 1
                continue
            r = rec['fletcher']
            all_r.append(r)
            n_total_with_r += 1
            if good_lo <= r <= good_hi:
                n_good += 1
    return {
        'n_molecules': nmol,
        'n_converged': converged,
        'total_steps': total_steps,
        'n_trust_updates': sum(bucket_counts.values()),
        'n_below_noise': n_noise,
        'n_with_fletcher_r': n_total_with_r,
        'n_good_quadratic_steps': n_good,
        'good_step_fraction': (n_good / n_total_with_r) if n_total_with_r else None,
        'bucket_counts': bucket_counts,
        'fletcher_r_stats': {
            'mean': float(np.mean(all_r)) if all_r else None,
            'median': float(np.median(all_r)) if all_r else None,
            'p10': float(np.percentile(all_r, 10)) if all_r else None,
            'p90': float(np.percentile(all_r, 90)) if all_r else None,
            'std': float(np.std(all_r)) if all_r else None,
        },
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def default_param_grid() -> list[FletcherParams]:
    return [
        FletcherParams(name='baseline'),
        # vary thresholds
        FletcherParams(name='strict_thr',     low_thr=0.40, high_thr=0.90),
        FletcherParams(name='loose_thr',      low_thr=0.10, high_thr=0.50),
        FletcherParams(name='symmetric_0.5',  low_thr=0.50, high_thr=0.50),
        # vary aggressiveness of trust changes
        FletcherParams(name='timid_grow',     grow_factor=1.5),
        FletcherParams(name='bold_grow',      grow_factor=3.0),
        FletcherParams(name='soft_shrink',    shrink_to=0.50),
        FletcherParams(name='hard_shrink',    shrink_to=0.10),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--benchmarks', nargs='*', default=['birkholz', 'baker'])
    ap.add_argument('--molecules', nargs='*', default=None,
                    help='restrict to these molecule names (debugging)')
    ap.add_argument('--maxsteps', type=int, default=110)
    ap.add_argument('--good-lo', type=float, default=0.75)
    ap.add_argument('--good-hi', type=float, default=1.25)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).with_name('results.json'))
    ap.add_argument('--summary', type=Path,
                    default=Path(__file__).with_name('summary.md'))
    ap.add_argument('--settings', nargs='*', default=None,
                    help='restrict to these named parameter settings')
    ap.add_argument('--resume', action='store_true',
                    help='if --out exists, load it and skip already-done cells')
    args = ap.parse_args(argv)

    import shutil
    if not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')

    grid = default_param_grid()
    if args.settings:
        grid = [p for p in grid if p.name in args.settings]
    out: dict[str, Any] = {
        'good_band': [args.good_lo, args.good_hi],
        'params': [vars(p) for p in default_param_grid()],
        'cells': {},
    }
    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        out['cells'] = prev.get('cells', {})
        print(f'resumed: already have {sum(len(c) for c in out["cells"].values())} cells')

    t_start = time.perf_counter()
    for params in grid:
        cell_setting: dict[str, Any] = out['cells'].get(params.name, {})
        # install our parametrized update_trust as the module default
        berny_mod.update_trust = make_update_trust(params)
        try:
            for bench in args.benchmarks:
                if bench in cell_setting:
                    print(f'[{params.name}/{bench}] skipping (already done)')
                    continue
                data_dir = BENCHMARKS[bench]
                reference = json.loads((data_dir / 'reference.json').read_text())
                names = args.molecules or sorted(reference)
                names = [n for n in names if n in reference]
                per_mol: dict[str, dict] = {}
                t0 = time.perf_counter()
                for n in names:
                    print(f'[{params.name}/{bench}] {n}', flush=True)
                    try:
                        per_mol[n] = run_one(n, reference[n], data_dir, args.maxsteps)
                    except Exception as e:
                        per_mol[n] = {
                            'converged': False, 'steps': 0, 'records': [],
                            'error': f'{type(e).__name__}: {e}',
                        }
                summary = summarize(per_mol, params, args.good_lo, args.good_hi)
                summary['wall_s'] = time.perf_counter() - t0
                cell_setting[bench] = {'per_mol': per_mol, 'summary': summary}
                print(f'  -> {bench}: '
                      f"good={summary['good_step_fraction']:.3f} "
                      f"({summary['n_good_quadratic_steps']}/"
                      f"{summary['n_with_fletcher_r']}) "
                      f"conv={summary['n_converged']}/{summary['n_molecules']} "
                      f"steps={summary['total_steps']} "
                      f"wall={summary['wall_s']:.1f}s", flush=True)
        finally:
            # restore stock implementation between cells
            import importlib
            importlib.reload(berny_mod)
        out['cells'][params.name] = cell_setting
        args.out.write_text(json.dumps(out, indent=2) + '\n')  # incremental dump

    out['total_wall_s'] = time.perf_counter() - t_start
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    write_summary(out, args.summary)
    print(f'\nWrote {args.out} and {args.summary}')


def write_summary(out: dict, path: Path) -> None:
    lines = []
    lines.append('# Fletcher trust-radius parameter sweep\n')
    lines.append(f"Good-step band: r ∈ [{out['good_band'][0]}, "
                 f"{out['good_band'][1]}]\n")
    benches = sorted({b for cell in out['cells'].values() for b in cell})
    for bench in benches:
        lines.append(f'## {bench}\n')
        lines.append('| setting | low_thr | high_thr | shrink_to | grow_factor | '
                     'good frac | n_good / n_r | shrink | keep | grow_or_keep | '
                     'below_noise | total steps | converged |')
        lines.append('|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|')
        for params_d in out['params']:
            name = params_d['name']
            s = out['cells'][name][bench]['summary']
            bc = s['bucket_counts']
            gf = s['good_step_fraction']
            gf_s = f"{gf:.3f}" if gf is not None else '-'
            lines.append(
                f"| {name} | {params_d['low_thr']} | {params_d['high_thr']} | "
                f"{params_d['shrink_to']} | {params_d['grow_factor']} | "
                f"{gf_s} | {s['n_good_quadratic_steps']} / {s['n_with_fletcher_r']} | "
                f"{bc['shrink']} | {bc['keep']} | {bc['grow_or_keep']} | "
                f"{bc['below_noise']} | {s['total_steps']} | "
                f"{s['n_converged']}/{s['n_molecules']} |"
            )
        lines.append('')
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    sys.exit(main())
