#!/usr/bin/env python3
"""Run bisphenol_a under GFN2-xTB, clean and perturbed, capturing full traces.

For each run we record pyberny's structured per-step ``trace`` JSON (energy,
trust radius, Hessian eigenvalues, convergence criteria, ...) *and* the
per-step Cartesian geometry, so the optimization path can be dissected
afterwards (``analyze`` in ``experiments.py`` consumes these).

Threads are pinned to 1 so the clean run is bitwise-reproducible at 72 steps;
with multi-threaded tblite the count scatters (~63-85) because the
non-deterministic OpenMP reductions change *when* the run tips off the
near-symmetric ridge (see the report and birkholz SOURCE.md).

Run from a checkout with ``pip install -e ".[benchmark]"`` (needs tblite).
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

CHARGE, MULT = 0, 1


def load_start():
    for name, geom, ref in iter_molecules('birkholz'):
        if name == 'bisphenol_a':
            return geom, ref
    raise SystemExit('bisphenol_a not found in birkholz benchmark')


def perturb(geom, sigma, seed):
    rng = np.random.default_rng(seed)
    return geomlib.Geometry(
        list(geom.species),
        geom.coords + rng.normal(0.0, sigma, size=geom.coords.shape),
        geom.lattice,
    )


def run(geom, tracepath, maxsteps=120):
    berny = Berny(geom, trace=str(tracepath), maxsteps=maxsteps)
    solver = XTBSolver(charge=CHARGE, mult=MULT)
    next(solver)
    coords_traj, energy = [], None
    for g in berny:
        coords_traj.append(np.array(g.coords))
        energy, grad = solver.send((list(g), g.lattice))
        berny.send((energy, grad))
    return berny.converged, berny._n, energy, np.array(coords_traj)


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else 'out')
    outdir.mkdir(parents=True, exist_ok=True)
    geom0, _ = load_start()
    runs = {'clean': geom0}
    for sigma in (0.02, 0.05, 0.1):
        for seed in (1, 2, 3):
            runs[f's{sigma}_seed{seed}'] = perturb(geom0, sigma, seed)
    summary = {}
    for name, geom in runs.items():
        conv, n, e, traj = run(geom, outdir / f'{name}.trace.json')
        np.save(outdir / f'{name}.traj.npy', traj)
        summary[name] = {'converged': bool(conv), 'steps': int(n), 'energy': float(e)}
        print(f'{name:16s} conv={conv} steps={n:3d} E={e:.8f}')
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
