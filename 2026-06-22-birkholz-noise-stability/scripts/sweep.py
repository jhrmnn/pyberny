#!/usr/bin/env python3
"""Resumable, trial-level-parallel driver for the noise-stability sweep.

`par_noise.py` parallelizes at the molecule level, but a single large molecule
(e.g. birkholz `azadirachtin`, 95 atoms) runs all its trials serially and can
take tens of minutes -- longer than a container session or a bounded job
window. This driver instead enumerates every *trial* ``(molecule, sigma, seed)``
(plus the unperturbed clean trial per molecule), runs them across a process pool
(single-threaded xTB per worker), and **checkpoints each trial to its own JSON
file**. Relaunching skips any trial already on disk, so the sweep resumes after
a restart and makes monotonic progress.

Trials store the raw optimizer result + energy; the clean-relative energy
(`denergy_kcal`) and all per-molecule / aggregate summaries are computed at
assembly time, reusing `noise_stability.summarize_molecule` /
`noise_stability.format_report` so the emitted JSON/markdown match the serial
tool exactly.

Usage::

    sweep.py --benchmark birkholz --seeds 6 --sigmas 0.02 0.05 0.1 0.2 0.3 \
        --workers 4 --ckpt-dir ckpt --out out.md --out-json out.json
    # rerun the same command after any restart to resume; when every trial is
    # present it writes the final outputs.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import noise_stability as ns  # noqa: E402
from berny.benchmarks import iter_molecules, load_reference  # noqa: E402

HARTREE_KCAL = ns.HARTREE_KCAL


@lru_cache(maxsize=None)
def _load(benchmark, name):
    for n, g, r in iter_molecules(benchmark, [name]):
        return g, r
    raise KeyError(name)


def _trial_key(name, sigma, seed):
    if seed is None:
        return f'{name}__clean'
    return f'{name}__s{sigma:g}__{seed}'


def _work(args):
    benchmark, name, sigma, seed, maxsteps, ckpt_dir = args
    geom, ref = _load(benchmark, name)
    if seed is None:
        res = ns._optimize(geom, ref, maxsteps)
    else:
        import numpy as np

        rng = np.random.default_rng(abs(hash((name, round(sigma, 6), seed))) % (2**32))
        res = ns._optimize(ns._perturb(geom, sigma, rng), ref, maxsteps)
    trial = {'sigma': sigma, 'seed': seed, **res}
    Path(ckpt_dir, _trial_key(name, sigma, seed) + '.json').write_text(
        json.dumps(trial)
    )
    return name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='baker')
    ap.add_argument('--molecules', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument(
        '--sigmas', type=float, nargs='*', default=[0.02, 0.05, 0.1, 0.2, 0.3]
    )
    ap.add_argument('--maxsteps', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ckpt-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--out-json', type=Path, default=None)
    args = ap.parse_args(argv)

    reference = load_reference(args.benchmark)
    names = args.molecules or sorted(reference)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all trials: clean (seed=None) + noisy grid.
    all_trials = []
    for name in names:
        all_trials.append((name, 0.0, None))
        for sigma in args.sigmas:
            for seed in range(args.seeds):
                all_trials.append((name, sigma, seed))

    todo = [
        (args.benchmark, n, s, sd, args.maxsteps, str(args.ckpt_dir))
        for (n, s, sd) in all_trials
        if not (args.ckpt_dir / (_trial_key(n, s, sd) + '.json')).exists()
    ]
    print(
        f'{len(all_trials) - len(todo)}/{len(all_trials)} trials already done; '
        f'{len(todo)} to run',
        flush=True,
    )

    t0 = time.perf_counter()
    done = 0
    # Recover from a BrokenProcessPool (a worker killed by OOM / container
    # hiccup) by recreating the pool over whatever trials remain unchecked --
    # completed ones are on disk and skipped, so this resumes cleanly.
    remaining = list(todo)
    while remaining:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for _ in ex.map(_work, remaining):
                    done += 1
                    if done % 10 == 0:
                        print(f'  [{done}/{len(todo)}] trials done', flush=True)
            remaining = []
        except Exception as e:  # noqa: BLE001
            print(f'  pool broke ({type(e).__name__}); recreating', flush=True)
            time.sleep(2)
            remaining = [
                t
                for t in todo
                if not (
                    args.ckpt_dir / (_trial_key(t[1], t[2], t[3]) + '.json')
                ).exists()
            ]
    wall = time.perf_counter() - t0

    # Assemble per-molecule trial lists from checkpoints.
    missing = 0
    raw = {}
    summaries = []
    for name in names:
        trials = []
        clean_e = None
        # Clean trial first.
        ck = args.ckpt_dir / (_trial_key(name, 0.0, None) + '.json')
        if not ck.exists():
            missing += 1
            continue
        c = json.loads(ck.read_text())
        clean_e = c['energy']
        trials.append({**c, 'denergy_kcal': 0.0 if clean_e is not None else None})
        for sigma in args.sigmas:
            for seed in range(args.seeds):
                ck = args.ckpt_dir / (_trial_key(name, sigma, seed) + '.json')
                if not ck.exists():
                    missing += 1
                    continue
                t = json.loads(ck.read_text())
                e = t['energy']
                de = (
                    (e - clean_e) * HARTREE_KCAL
                    if (e is not None and clean_e is not None)
                    else None
                )
                trials.append({**t, 'denergy_kcal': de})
        raw[name] = trials
        _, ref = _load(args.benchmark, name)
        summaries.append(ns.summarize_molecule(name, ref, trials))

    if missing:
        print(f'\nINCOMPLETE: {missing} trials still missing; rerun to finish.')
        return

    report = ns.format_report(
        summaries, args.sigmas, args.seeds, 'xtb', args.benchmark, wall
    )
    if args.out:
        args.out.write_text(report)
    if args.out_json:
        args.out_json.write_text(
            json.dumps(
                {
                    'benchmark': args.benchmark,
                    'solver': 'xtb',
                    'sigmas': args.sigmas,
                    'seeds': args.seeds,
                    'raw': raw,
                    'summaries': summaries,
                },
                indent=2,
            )
        )
    print(f'\nCOMPLETE; wall this run {wall:.0f}s')
    print(report, end='')


if __name__ == '__main__':
    main()
