#!/usr/bin/env python3
"""Re-run the soft-mode-sensitive microvariation cells with thresholds 10x tighter.

Targets the cells where the standard-tolerance microvariation experiment
produced within-basin energy spread far above what gradmax * stepmax should
permit:

- mg_porphin sigma=0.001  (spread 1.3 uHa but max atomic disp 0.084 A)
- mg_porphin sigma=0.005  (spread 205 uHa, same 0.083 A scale)
- estradiol  sigma=0.001  (spread 154 uHa)

Plus a control (vitamin_c sigma=0.001) that was already sub-uHa.

If the hypothesis "loose-on-soft-modes, not basin-hopping" is right, the
tightened spreads should collapse to << 1 uHa for vitamin_c-grade behavior.
"""

import json
import time
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.solvers import MopacSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / 'tests' / 'data' / 'birkholz_schlegel'

CELLS = [
    ('vitamin_c', 0.001),
    ('estradiol', 0.001),
    ('mg_porphin', 0.001),
    ('mg_porphin', 0.005),
]
SEEDS = list(range(10))
MAXSTEPS = 200
TIGHT_PARAMS = {
    'gradientmax': 0.45e-4,
    'gradientrms': 0.15e-4,
    'stepmax': 1.8e-4,
    'steprms': 1.2e-4,
}


def perturb(geom, sigma, seed):
    rng = np.random.default_rng(seed)
    new_coords = geom.coords + rng.normal(scale=sigma, size=geom.coords.shape)
    return geomlib.Geometry(list(geom.species), new_coords, geom.lattice)


def kabsch_rmsd(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_rot = a @ rot.T
    return float(np.sqrt(np.mean(np.sum((a_rot - b) ** 2, axis=1))))


def run_one(geom, ref, maxsteps, params):
    berny = Berny(geom, maxsteps=maxsteps, **params)
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


def main():
    out_dir = REPO_ROOT / 'experiments' / 'microvariations'
    reference = json.loads((DATA / 'reference.json').read_text())

    plan = []
    for mol, sigma in CELLS:
        plan.append((mol, 0.0, 0))
        for seed in SEEDS:
            plan.append((mol, sigma, seed))

    rows = []
    baseline_geoms = {}
    t_start = time.perf_counter()
    for i, (mol, sigma, seed) in enumerate(plan, 1):
        ref = reference[mol]
        base_geom = geomlib.readfile(str(DATA / f'{mol}.xyz'))
        geom = base_geom if sigma == 0.0 else perturb(base_geom, sigma, seed)
        t0 = time.perf_counter()
        try:
            converged, n_steps, final_geom, energy = run_one(
                geom, ref, MAXSTEPS, TIGHT_PARAMS
            )
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

        rows.append(
            {
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
        )
        elapsed = time.perf_counter() - t_start
        print(
            f'[{i}/{len(plan)}] {mol} sigma={sigma} seed={seed}: '
            f'converged={converged} steps={n_steps} wall={wall:.1f}s '
            f'(total {elapsed:.0f}s)',
            flush=True,
        )

        slim = [{k: v for k, v in r.items() if k != 'final_coords'} for r in rows]
        with open(out_dir / 'results_tight.json', 'w') as f:
            json.dump(
                {
                    'cells': CELLS,
                    'seeds': SEEDS,
                    'maxsteps': MAXSTEPS,
                    'params': TIGHT_PARAMS,
                    'rows': slim,
                },
                f,
                indent=2,
            )
        with open(out_dir / 'final_coords_tight.json', 'w') as f:
            json.dump(
                {
                    f'{r["molecule"]}|{r["sigma"]}|{r["seed"]}': r['final_coords']
                    for r in rows
                },
                f,
            )


if __name__ == '__main__':
    main()
