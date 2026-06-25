#!/usr/bin/env python3
"""Probe the bisphenol_a near-symmetric-ridge slowdown and candidate fixes.

Produces the numbers quoted in the report:

* anatomy of the clean 72-step path (negative-eigenvalue / on-sphere /
  trust-collapse step counts, and the aryl-torsion coordinate frozen on the
  ridge then breaking);
* the Cs-mirror test on the start geometry;
* three "fixes" --- a directed ring twist, a trust-radius sweep, and a
  tiny-noise threshold --- and the distinct minima they land in.

Threads pinned to 1 for reproducibility. Needs ``pip install -e ".[benchmark]"``.
"""

import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json
import sys
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules
from berny.solvers import XTBSolver

HK = 627.5094740631  # Hartree -> kcal/mol
# Ring-2 atoms (ipso through the para-OH), rotated about the C0-C19 bond axis
# for the directed symmetry-breaking twist.
RING2 = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# Aryl-torsion coordinates: ortho-ipso-central-(other ipso) for each ring.
R1 = (10, 9, 0, 19)
R2 = (20, 19, 0, 9)


def load_start():
    for name, geom, ref in iter_molecules('birkholz'):
        if name == 'bisphenol_a':
            return np.array(geom.coords), list(geom.species), geom.lattice
    raise SystemExit('bisphenol_a not found')


def dihedral(C, t):
    p0, p1, p2, p3 = (C[i] for i in t)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def kabsch_rmsd(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return np.sqrt(((B - A @ R.T) ** 2).sum() / len(A))


def opt(C, sp, lattice, trust=0.3, maxsteps=120):
    geom = geomlib.Geometry(sp, C.copy(), lattice)
    berny = Berny(geom, maxsteps=maxsteps, trust=trust)
    solver = XTBSolver(charge=0, mult=1)
    next(solver)
    e = None
    for g in berny:
        e, gr = solver.send((list(g), g.lattice))
        berny.send((e, gr))
    return berny._n, e, np.array(berny._state.geom.coords)


def twist(C, deg):
    C = C.copy()
    ax = C[19] - C[0]
    ax /= np.linalg.norm(ax)
    th = np.radians(deg)
    o, ct, st = C[0], np.cos(np.radians(deg)), np.sin(th)
    for i in RING2:
        v = C[i] - o
        C[i] = o + v * ct + np.cross(ax, v) * st + ax * np.dot(ax, v) * (1 - ct)
    return C


def cs_mirror_test(C, sp):
    ring1 = [0, 9, 10, 11, 12, 14, 16, 31]
    z0 = C[ring1, 2].mean()
    Cr = C.copy()
    Cr[:, 2] = 2 * z0 - C[:, 2]
    used, worst = set(), 0.0
    for i in range(len(C)):
        best, bd = None, 1e9
        for j in range(len(C)):
            if j in used or sp[j] != sp[i]:
                continue
            d = np.linalg.norm(Cr[i] - C[j])
            if d < bd:
                bd, best = d, j
        used.add(best)
        worst = max(worst, bd)
    return z0, worst


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'out')
    out.mkdir(parents=True, exist_ok=True)
    C0, sp, lat = load_start()
    res = {}

    z0, worst = cs_mirror_test(C0, sp)
    res['cs_mirror'] = {'plane_z': z0, 'max_atom_image_dev_A': worst}
    print(f'Cs-mirror test: plane z={z0:.4f}, max atom-image dev={worst:.4f} A')
    print(f'start aryl torsions: R1={dihedral(C0, R1):.2f} R2={dihedral(C0, R2):.2f}')

    n0, e0, Cclean = opt(C0, sp, lat)
    print(f'\nclean: {n0} steps, E={e0:.6f}')

    print('\n=== directed ring-2 twist ===')
    res['twist'] = {}
    for d in (20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05):
        n, e, Cf = opt(twist(C0, d), sp, lat)
        res['twist'][d] = {'steps': n, 'dE_vs_clean_kcal': (e - e0) * HK}
        print(f'  twist {d:5.2f} deg: {n:3d} steps, dE_vs_clean={(e - e0) * HK:+.3f} kcal/mol')

    print('\n=== trust-radius sweep (unperturbed start) ===')
    res['trust'] = {}
    for t in (0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.5):
        n, e, Cf = opt(C0, sp, lat, trust=t)
        res['trust'][t] = {
            'steps': n,
            'E': e,
            'R1': dihedral(Cf, R1),
            'R2': dihedral(Cf, R2),
            'dE_vs_clean_kcal': (e - e0) * HK,
        }
        print(
            f'  trust={t:4.2f}: {n:3d} steps, E={e:.6f} '
            f'(R1={dihedral(Cf, R1):6.1f} R2={dihedral(Cf, R2):6.1f}, '
            f'{(e - e0) * HK:+.2f} kcal/mol vs clean)'
        )

    print('\n=== tiny isotropic-noise threshold (3 seeds) ===')
    res['noise'] = {}
    for sig in (0.001, 0.002, 0.005, 0.01, 0.02):
        steps = []
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed * 100 + int(sig * 1e5))
            n, e, _ = opt(C0 + rng.normal(0, sig, C0.shape), sp, lat)
            steps.append(n)
        res['noise'][sig] = steps
        print(f'  sigma={sig:5.3f} A: steps={steps}')

    (out / 'experiments.json').write_text(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
