#!/usr/bin/env python3
"""Trace a clean vs noise-perturbed GFN2-xTB optimization of bisphenol_a.

Drives :class:`berny.Berny` with the built-in ``trace=`` JSON facility (one
structured record per step: energy, trust radius, quadratic-step eigenvalues,
predicted vs actual dE, convergence criteria) and additionally records the two
aryl--C(CH3)2 torsions that the descent has to rotate. Writes the per-step
traces and a small summary to ``--out-dir``.

    python trace_opt.py --out-dir ../data --sigma 0.02 --seed 12345

The two torsions are methyl(1)-Cq(0)-Cipso-Cortho for each phenol ring; on the
bundled near-symmetric start they are nearly equal, and the minimum is
asymmetric, so the descent must break the molecule's near-C2 symmetry.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules
from berny.solvers import XTBSolver

HARTREE_KCAL = 627.5094740631
# Central quaternary C = atom index 0; methyl C = 1; aryl ipso C = 9 (ring 1),
# 18 (ring 2); ortho C = 10 (ring 1), 19 (ring 2).
TOR1, TOR2 = (1, 0, 9, 10), (1, 0, 18, 19)


def dihedral(c, idx):
    p0, p1, p2, p3 = (c[i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1))
    return float(np.degrees(np.arctan2(np.dot(m, n2), np.dot(n1, n2))))


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def run(g0, ref, trace_path, maxsteps=150):
    berny = Berny(g0, maxsteps=maxsteps, trace=str(trace_path))
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    energy, t1, t2, coords = None, [], [], []
    for g in berny:
        c = np.array(g.coords)
        coords.append(c)
        t1.append(dihedral(c, TOR1))
        t2.append(dihedral(c, TOR2))
        energy, gradients = solver.send((list(g), g.lattice))
        berny.send((energy, gradients))
    return berny, energy, t1, t2, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=Path('../data'))
    ap.add_argument('--sigma', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, geom, ref = next(iter(iter_molecules('birkholz', ['bisphenol_a'])))
    start = np.array(geom.coords)

    b0, e0, c1, c2, co0 = run(geom, ref, args.out_dir / 'xtb_trace_clean.json')
    print(f'clean: converged={b0.converged} steps={b0._n} E={e0:.8f} '
          f'start->final RMSD={rmsd(start, co0[-1]):.4f}')

    rng = np.random.default_rng(args.seed)
    noisy = geomlib.Geometry(
        list(geom.species),
        geom.coords + rng.normal(0.0, args.sigma, size=geom.coords.shape),
        geom.lattice,
    )
    b1, e1, n1, n2, co1 = run(noisy, ref, args.out_dir / 'xtb_trace_noisy.json')
    print(f'noisy(sigma={args.sigma},seed={args.seed}): converged={b1.converged} '
          f'steps={b1._n} E={e1:.8f} dE={(e1 - e0) * HARTREE_KCAL:+.4f} kcal '
          f'final-final RMSD={rmsd(co0[-1], co1[-1]):.4f}')

    summary = {
        'start_torsions': {'t1': dihedral(start, TOR1), 't2': dihedral(start, TOR2)},
        'clean': {'steps': b0._n, 'E': e0, 'torsions': {'t1': c1, 't2': c2}},
        'noisy': {'sigma': args.sigma, 'seed': args.seed, 'steps': b1._n, 'E': e1,
                  'dE_kcal': (e1 - e0) * HARTREE_KCAL, 'torsions': {'t1': n1, 't2': n2}},
    }
    (args.out_dir / 'xtb_torsions.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
