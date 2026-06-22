#!/usr/bin/env python3
"""Step-count stability of pyberny trajectories under small start-geometry noise.

Companion experiment to the `2026-06-20-baker-noise-stability` report. That
study showed the *located minimum* is stable to small Cartesian noise; this one
asks the orthogonal question: when slightly perturbed starts converge to the
**same** minimum, is the *number of optimization steps* stable?

Method: for every Baker molecule except the seven "frustrated reference" ones
(whose unperturbed start sits on/near a symmetric saddle and which therefore
relax to a *different* minimum under any noise -- see the
`baker-symmetry-saddle` and `baker-ethanol-histidine-conformer` reports and
pyberny#148), optimize the clean start, then optimize `NSEED` noisy copies at
each small amplitude. Only trials that converge to the same minimum (final
energy within 0.1 kcal/mol of the clean run) are kept; their step counts are
recorded. Writes `step_stability.json` consumed by `plot_step_stability.py`.

GFN2-xTB via tblite. Run from a checkout with pyberny installed
(`pip install -e ".[benchmark]"`).
"""

import json
import statistics
import time

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules, load_reference
from berny.solvers import XTBSolver

H2KCAL = 627.5094740631

# Excluded: their no-noise reference is a symmetric saddle / non-minimum, so
# noisy starts relax to a *different* basin (pyberny#148). The step-count
# question only makes sense among trials that reach the same minimum.
FRUSTRATED = {
    'methylamine',
    'mesityl_oxide',
    'benzidine',
    'acanil01',
    'caffeine',
    'ethanol',
    'histidine',
}

SIGMAS = [0.02, 0.05, 0.1]  # Angstrom RMS per Cartesian coordinate
NSEED = 30
SAME_MIN_KCAL = 0.1


def optimize(geom, ref, maxsteps=150):
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


def main():
    reference = load_reference('baker')
    names = sorted(n for n in reference if n not in FRUSTRATED)
    geoms = {n: (g, r) for n, g, r in iter_molecules('baker', names)}

    t0 = time.time()
    out = {}
    for name in names:
        geom, ref = geoms[name]
        conv0, n0, e0 = optimize(geom, ref)
        rec = {'clean_steps': n0, 'clean_converged': conv0, 'by_sigma': {}}
        for sigma in SIGMAS:
            steps = []
            diff_basin = 0
            for seed in range(NSEED):
                try:
                    conv, n, e = optimize(perturb(geom, sigma, seed), ref)
                except Exception:
                    continue
                if not conv:
                    continue
                if e is not None and e0 is not None and abs((e - e0) * H2KCAL) > SAME_MIN_KCAL:
                    diff_basin += 1
                    continue
                steps.append(n)
            rec['by_sigma'][str(sigma)] = {'steps': steps, 'diff_basin': diff_basin}
        out[name] = rec
        s = rec['by_sigma']['0.05']['steps']
        cv = 100 * statistics.pstdev(s) / statistics.mean(s) if len(s) > 1 else 0
        print(
            f'{name:24s} clean={n0:>2}  sig0.05 med={int(statistics.median(s)) if s else 0:>2} '
            f'range={min(s) if s else 0}-{max(s) if s else 0} CV={cv:>3.0f}%',
            flush=True,
        )

    with open('step_stability.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(
        f'\nDONE {time.time() - t0:.0f}s; {len(names)} molecules x '
        f'{len(SIGMAS)} sigmas x {NSEED} seeds -> step_stability.json'
    )


if __name__ == '__main__':
    main()
