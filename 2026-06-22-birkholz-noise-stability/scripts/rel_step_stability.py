#!/usr/bin/env python3
"""Step-count stability under *per-system* noise scaled to the start->minimum RMSD.

Generalized companion to `step_stability.py`, which used fixed absolute
amplitudes sigma. That design over-perturbs molecules whose pre-relaxed start
already sits almost exactly at the minimum (tiny start->min RMSD), inflating
their step count purely because the noise dwarfs the natural travel distance.
This script removes that confound: for each molecule the noise amplitude is
derived so that the perturbation RMSD is a fixed *fraction* of that molecule's
own start->minimum RMSD (R_sm).

For a fraction f, the per-coordinate Gaussian sigma is `f * R_sm / sqrt(3)`
(raw RMSD of iid N(0,sigma) noise over 3N coordinates is sqrt(3)*sigma), so the
perturbation RMSD is ~= f * R_sm. Run with `--fracs` (default 0.1 0.2 0.4); the
0.2 (20 %) level is the headline. Takes `--benchmark` and the same `--exclude`
frustrated set as `step_stability.py`.

GFN2-xTB via tblite; run from a checkout with pyberny installed.
"""

import argparse
import json
import statistics as st
import time

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules, load_reference
from berny.solvers import XTBSolver

H2KCAL = 627.5094740631


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((a @ rot.T - b) ** 2).sum(1).mean()))


def optimize(geom, ref, maxsteps):
    berny = Berny(geom, maxsteps=maxsteps)
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    energy = None
    g = geom
    for g in berny:
        energy, gradients = solver.send((list(g), g.lattice))
        berny.send((energy, gradients))
    return berny.converged, berny._n, energy, g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--benchmark', default='baker')
    ap.add_argument('--molecules', nargs='*', default=None)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--fracs', type=float, nargs='*', default=[0.1, 0.2, 0.4])
    ap.add_argument('--nseed', type=int, default=30)
    ap.add_argument('--maxsteps', type=int, default=150)
    ap.add_argument('--same-min', type=float, default=0.1, help='kcal/mol')
    ap.add_argument('--ckpt-dir', default=None, help='per-molecule resume dir')
    ap.add_argument('--out', default='rel_step_stability.json')
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
        # Resume at (molecule, frac) granularity: a partial file may already
        # hold some completed fracs for this molecule (large acenes/peptides can
        # exceed a single bounded window for all fracs).
        ckfile = (ckdir / f'{name}.json') if ckdir is not None else None
        if ckfile is not None and ckfile.exists():
            rec = json.loads(ckfile.read_text())
            if len(rec['by_frac']) >= len(args.fracs):
                out[name] = rec
                continue
        else:
            rec = None
        geom, ref = geoms[name]
        if rec is None:
            _conv0, n0, e0, gmin = optimize(geom, ref, args.maxsteps)
            r_sm = kabsch_rmsd(geom.coords, gmin.coords)
            rec = {'clean_steps': n0, 'Rsm': r_sm, 'by_frac': {}}
        else:
            n0, r_sm = rec['clean_steps'], rec['Rsm']
            _c, _n, e0, _g = optimize(geom, ref, args.maxsteps)
        for frac in args.fracs:
            if str(frac) in rec['by_frac']:
                continue
            sigma = frac * r_sm / np.sqrt(3)
            steps = []
            noise_rmsd = []
            diff = 0
            errors = 0
            for seed in range(args.nseed):
                rng = np.random.default_rng((int(frac * 1000), seed, 7))
                disp = rng.normal(0.0, sigma, geom.coords.shape)
                noisy = geomlib.Geometry(
                    list(geom.species), geom.coords + disp, geom.lattice
                )
                noise_rmsd.append(kabsch_rmsd(noisy.coords, geom.coords))
                try:
                    conv, n, e, _ = optimize(noisy, ref, args.maxsteps)
                except Exception:
                    errors += 1
                    continue
                if not conv or (
                    e is not None and abs((e - e0) * H2KCAL) > args.same_min
                ):
                    diff += 1
                    continue
                steps.append(n)
            rec['by_frac'][str(frac)] = {
                'sigma': sigma,
                'steps': steps,
                'noise_rmsd_mean': float(np.mean(noise_rmsd)),
                'diff_basin': diff,
                'errors': errors,
            }
            if ckfile is not None:
                ckfile.write_text(json.dumps(rec))
        out[name] = rec
        if ckdir is not None:
            (ckdir / f'{name}.json').write_text(json.dumps(rec))
        frac_key = str(args.fracs[min(1, len(args.fracs) - 1)])
        s = rec['by_frac'][frac_key]['steps']
        cv = 100 * st.pstdev(s) / st.mean(s) if len(s) > 1 else 0
        infl = st.median(s) / n0 if s else 0
        print(
            f'{name:24s} Rsm={r_sm:.3f} clean={n0:>3}  {frac_key}: '
            f'med={int(st.median(s)) if s else 0:>3} CV={cv:>3.0f}% '
            f'infl={infl:.2f}x n={len(s)}',
            flush=True,
        )

    with open(args.out, 'w') as f:
        json.dump(
            {'benchmark': args.benchmark, 'fracs': args.fracs, 'data': out},
            f,
            indent=2,
        )
    print(f'\nDONE {time.time() - t0:.0f}s -> {args.out}')


if __name__ == '__main__':
    main()
