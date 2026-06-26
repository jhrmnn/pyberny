#!/usr/bin/env python3
"""Step-count scatter for bisphenol_a: clean (repeated) vs noise-perturbed (GFN2-xTB).

Runs the clean optimization several times -- tblite's OpenMP reductions are not
bitwise-deterministic, and near this molecule's flat ridge the ~1e-9 Ha noise
reroutes the trust-radius path, so the *clean* count itself scatters -- then
sweeps small Gaussian start perturbations at several amplitudes, keeping only
trials that converge to the same minimum (|dE| <= 0.2 kcal/mol). Writes
``xtb_scatter.json`` (counts + a representative clean/noisy torsion trajectory).

    python scatter.py --out-dir ../data --nclean 8 --nseed 6
"""
import argparse
import json
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules
from berny.solvers import XTBSolver

HARTREE_KCAL = 627.5094740631
TOR1, TOR2 = (1, 0, 9, 10), (1, 0, 18, 19)


def dih(c, idx):
    p0, p1, p2, p3 = (c[i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1))
    return float(np.degrees(np.arctan2(np.dot(m, n2), np.dot(n1, n2))))


def run(g0, ref):
    berny = Berny(g0, maxsteps=150)
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    energy, t1, t2 = None, [], []
    for g in berny:
        c = np.array(g.coords)
        t1.append(dih(c, TOR1)); t2.append(dih(c, TOR2))
        energy, gradients = solver.send((list(g), g.lattice))
        berny.send((energy, gradients))
    return berny.converged, berny._n, energy, t1, t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=Path('../data'))
    ap.add_argument('--nclean', type=int, default=8)
    ap.add_argument('--nseed', type=int, default=6)
    ap.add_argument('--sigmas', type=float, nargs='*', default=[0.02, 0.05, 0.1])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, geom, ref = next(iter(iter_molecules('birkholz', ['bisphenol_a'])))

    clean_steps, clean_E, rep_clean = [], [], None
    for _ in range(args.nclean):
        conv, n, E, t1, t2 = run(geom, ref)
        clean_steps.append(n); clean_E.append(E)
        if rep_clean is None:
            rep_clean = {'n': n, 't1': t1, 't2': t2}
    e_ref = clean_E[0]
    print(f'clean steps: {clean_steps} | same-min spread: '
          f'{(max(clean_E) - min(clean_E)) * HARTREE_KCAL:.4f} kcal')

    noise, rep_noisy = {}, None
    for sigma in args.sigmas:
        counts = []
        for seed in range(args.nseed):
            rng = np.random.default_rng(seed + int(sigma * 1000))
            g0 = geomlib.Geometry(list(geom.species),
                                  geom.coords + rng.normal(0, sigma, geom.coords.shape),
                                  geom.lattice)
            conv, n, E, t1, t2 = run(g0, ref)
            if conv and abs((E - e_ref) * HARTREE_KCAL) <= 0.2:
                counts.append(n)
                if sigma == args.sigmas[0] and rep_noisy is None:
                    rep_noisy = {'n': n, 't1': t1, 't2': t2}
        noise[str(sigma)] = counts
        print(f'sigma={sigma}: {counts} median={np.median(counts) if counts else None}')

    (args.out_dir / 'xtb_scatter.json').write_text(json.dumps({
        'clean_steps': clean_steps,
        'clean_E_spread_kcal': (max(clean_E) - min(clean_E)) * HARTREE_KCAL,
        'noise': noise, 'rep_clean': rep_clean, 'rep_noisy': rep_noisy}, indent=2))


if __name__ == '__main__':
    main()
