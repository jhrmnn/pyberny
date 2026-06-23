#!/usr/bin/env python3
"""Step-count stability of pyberny trajectories under small start-geometry noise.

Generalized companion to the `2026-06-20-baker-noise-stability` /
`2026-06-22-baker-step-count-stability` reports. Those studies asked: when a
slightly perturbed start converges to the **same** minimum, is the *number of
optimization steps* stable? This version takes any bundled benchmark
(`--benchmark`) and a per-benchmark "frustrated reference" exclusion list
(`--exclude`), so the same question can be asked of `birkholz` and `oligomers`.

Method: for every molecule in the benchmark (optionally restricted to
`--molecules`, minus `--exclude`), optimize the clean start, then optimize
`--nseed` noisy copies at each `--sigmas` amplitude. Only trials that converge
to the same minimum (final energy within `--same-min` kcal/mol of the clean
run) are kept; their step counts are recorded. Writes a JSON consumed by
`plot_step_stability.py`.

GFN2-xTB via tblite. Run from a checkout with pyberny installed
(`pip install -e ".[benchmark]"`).
"""

import argparse
import json
import statistics
import time

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules, load_reference
from berny.solvers import XTBSolver

H2KCAL = 627.5094740631


def optimize(geom, ref, maxsteps):
    """Return (converged, n_steps, final_energy)."""
    berny = Berny(geom, maxsteps=maxsteps)
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    energy = None
    for g in berny:
        energy, gradients = solver.send((list(g), g.lattice))
        berny.send((energy, gradients))
    return berny.converged, berny._n, energy


def perturb(geom, sigma, seed):
    rng = np.random.default_rng((int(sigma * 1000), seed, 11))
    return geomlib.Geometry(
        list(geom.species),
        geom.coords + rng.normal(0.0, sigma, size=geom.coords.shape),
        geom.lattice,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--benchmark', default='baker')
    ap.add_argument('--molecules', nargs='*', default=None)
    ap.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='"frustrated reference" molecules to drop (their clean start is a '
        'saddle / non-minimum, so noisy starts relax to a different basin)',
    )
    ap.add_argument('--sigmas', type=float, nargs='*', default=[0.02, 0.05, 0.1])
    ap.add_argument('--nseed', type=int, default=30)
    ap.add_argument('--maxsteps', type=int, default=150)
    ap.add_argument('--same-min', type=float, default=0.1, help='kcal/mol')
    ap.add_argument('--ckpt-dir', default=None, help='per-molecule resume dir')
    ap.add_argument('--out', default='step_stability.json')
    args = ap.parse_args(argv)

    reference = load_reference(args.benchmark)
    pool = args.molecules or sorted(reference)
    exclude = set(args.exclude)
    names = [n for n in pool if n not in exclude]
    geoms = {n: (g, r) for n, g, r in iter_molecules(args.benchmark, names)}

    ckdir = None
    if args.ckpt_dir:
        from pathlib import Path

        ckdir = Path(args.ckpt_dir)
        ckdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out = {}
    for name in names:
        if ckdir is not None and (ckdir / f'{name}.json').exists():
            out[name] = json.loads((ckdir / f'{name}.json').read_text())
            continue
        geom, ref = geoms[name]
        conv0, n0, e0 = optimize(geom, ref, args.maxsteps)
        rec = {'clean_steps': n0, 'clean_converged': conv0, 'by_sigma': {}}
        for sigma in args.sigmas:
            steps = []
            diff_basin = 0
            errors = 0
            for seed in range(args.nseed):
                try:
                    conv, n, e = optimize(
                        perturb(geom, sigma, seed), ref, args.maxsteps
                    )
                except Exception:
                    errors += 1
                    continue
                if not conv:
                    continue
                if (
                    e is not None
                    and e0 is not None
                    and abs((e - e0) * H2KCAL) > args.same_min
                ):
                    diff_basin += 1
                    continue
                steps.append(n)
            rec['by_sigma'][str(sigma)] = {
                'steps': steps,
                'diff_basin': diff_basin,
                'errors': errors,
            }
        out[name] = rec
        if ckdir is not None:
            (ckdir / f'{name}.json').write_text(json.dumps(rec))
        sig_key = str(args.sigmas[min(1, len(args.sigmas) - 1)])
        s = rec['by_sigma'][sig_key]['steps']
        cv = 100 * statistics.pstdev(s) / statistics.mean(s) if len(s) > 1 else 0
        print(
            f'{name:24s} clean={n0:>3}  sig{sig_key} '
            f'med={int(statistics.median(s)) if s else 0:>3} '
            f'range={min(s) if s else 0}-{max(s) if s else 0} CV={cv:>3.0f}% '
            f'n={len(s)}',
            flush=True,
        )

    with open(args.out, 'w') as f:
        json.dump(
            {'benchmark': args.benchmark, 'sigmas': args.sigmas, 'data': out},
            f,
            indent=2,
        )
    print(
        f'\nDONE {time.time() - t0:.0f}s; {len(names)} molecules x '
        f'{len(args.sigmas)} sigmas x {args.nseed} seeds -> {args.out}'
    )


if __name__ == '__main__':
    main()
