#!/usr/bin/env python3
"""Probe how stable pyberny's convergence is when the gradients it optimizes
against are corrupted by random noise.

Geometry optimizers consume an energy and a gradient at every step. Real
electronic-structure codes never return these exactly: SCF is only converged
to a threshold, grids are finite, and finite-difference gradients carry their
own error. This script asks a focused question on the Baker test set
(GFN2-xTB): *as the gradient noise grows, how gracefully does pyberny
degrade?*

For each molecule it first runs a clean, deterministic optimization to obtain
a reference minimum, then re-runs the optimization at a grid of noise levels
and, at each level, across several random seeds. At every step the clean
gradient from tblite is perturbed by additive Gaussian noise

    g_noisy = g_clean + N(0, sigma^2)            [sigma in atomic units]

before being handed to the optimizer; the *energy is left clean* so the
experiment isolates the effect of gradient error on the search direction and
on the gradient-based convergence test. Per run we record whether pyberny
reported convergence, how many steps it took, and -- crucially -- the *true*
(clean) gradient and energy at the geometry pyberny stopped at, plus that
geometry's RMSD from the clean reference minimum. The clean gradient at the
stopping point is what distinguishes a genuine minimum from a *false
convergence*: noise can make a noisy gradient dip under the threshold while
the true gradient is still large.

Outputs (under ``--out-dir``, default ``scripts/noise_stability_out``):

* ``raw.json``     -- every run's record, plus run metadata.
* ``summary.md``   -- per-noise-level aggregate table + per-molecule table.
* ``*.png``        -- plots of convergence rate / steps / true-gradient vs noise.

Run with no arguments for the full sweep, or use ``--molecules`` /
``--seeds`` / ``--levels`` / ``--quick`` to shrink it.
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from berny import Berny
from berny.benchmarks import iter_molecules, load_reference

# Convergence thresholds pyberny tests against (atomic units); pulled in for
# the report so "false convergence" is judged against the same bar pyberny
# uses, not a hand-picked one.
from berny.berny import BernyParams
from berny.solvers import XTBSolver

GRADIENTMAX = BernyParams.gradientmax  # 0.45e-3
GRADIENTRMS = BernyParams.gradientrms  # 0.15e-3

# Default gradient-noise standard deviations (a.u.). Chosen to bracket the
# convergence thresholds above: 1e-5 sits well below gradientrms, 3e-4 sits
# above gradientmax, so the grid spans "imperceptible" to "overwhelming".
DEFAULT_LEVELS = [1e-5, 3e-5, 1e-4, 3e-4]
DEFAULT_SEEDS = 6


def _grms(gradients):
    return float(np.sqrt(np.mean(gradients**2)))


def _gmax(gradients):
    return float(np.abs(gradients).max())


def run_once(geom, ref, sigma, rng, maxsteps):
    """Optimize ``geom`` with GFN2-xTB, perturbing each gradient by N(0,sigma^2).

    Returns a dict with the optimizer's verdict plus the *clean* energy and
    gradient at the geometry it stopped at (the last gradient computed in the
    loop is, by pyberny's protocol, evaluated at that final geometry).
    """
    berny = Berny(geom, maxsteps=maxsteps)
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    last_clean_energy = None
    last_clean_gmax = None
    last_clean_grms = None
    final_coords = None
    for g in berny:
        energy, gradients = solver.send((list(g), g.lattice))
        # Record the clean (true) values at this geometry before corrupting.
        last_clean_energy = float(energy)
        last_clean_gmax = _gmax(gradients)
        last_clean_grms = _grms(gradients)
        final_coords = np.array(g.coords)
        if sigma > 0:
            gradients = gradients + rng.normal(0.0, sigma, size=gradients.shape)
        berny.send((energy, gradients))
    return {
        'converged': bool(berny.converged),
        'steps': int(berny._n),
        'final_energy': last_clean_energy,
        'final_true_gmax': last_clean_gmax,
        'final_true_grms': last_clean_grms,
        'final_coords': final_coords,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--molecules', nargs='*', default=None)
    ap.add_argument(
        '--levels',
        nargs='*',
        type=float,
        default=DEFAULT_LEVELS,
        help='gradient-noise std devs in a.u. (default: %(default)s)',
    )
    ap.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--maxsteps', type=int, default=100)
    ap.add_argument(
        '--quick',
        action='store_true',
        help='small fast sweep: 6 molecules, 3 seeds, 2 noise levels',
    )
    ap.add_argument('--out-dir', type=Path, default=Path('scripts/noise_stability_out'))
    args = ap.parse_args(argv)

    levels = args.levels
    seeds = args.seeds
    molecules = args.molecules
    if args.quick:
        molecules = molecules or [
            'water',
            'ammonia',
            'acetone',
            'benzene',
            'histidine',
            'caffeine',
        ]
        seeds = 3
        levels = [3e-5, 1e-4]

    reference = load_reference('baker')
    names = molecules or sorted(reference)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    runs = []
    for name, geom, ref in iter_molecules('baker', names):
        print(f'==> {name} ({ref["atoms"]} atoms)', flush=True)
        # Clean reference optimization (deterministic, sigma=0).
        clean = run_once(geom, ref, 0.0, None, args.maxsteps)
        ref_coords = clean['final_coords']
        runs.append(
            {
                'molecule': name,
                'atoms': ref['atoms'],
                'sigma': 0.0,
                'seed': None,
                'converged': clean['converged'],
                'steps': clean['steps'],
                'final_energy': clean['final_energy'],
                'final_true_gmax': clean['final_true_gmax'],
                'final_true_grms': clean['final_true_grms'],
                'rmsd_from_clean': 0.0,
                'denergy_from_clean': 0.0,
            }
        )
        print(
            f'    clean: conv={clean["converged"]} steps={clean["steps"]} '
            f'E={clean["final_energy"]:.6f}',
            flush=True,
        )
        for sigma in levels:
            for seed in range(seeds):
                # Seed deterministically per (molecule, level, seed) so the
                # whole sweep is reproducible across processes (builtin hash()
                # is salted, so it would not be) and runs are independent.
                key = f'{name}|{sigma:.3e}|{seed}'.encode()
                rng = np.random.default_rng(zlib.crc32(key))
                r = run_once(geom, ref, sigma, rng, args.maxsteps)
                coords = r['final_coords']
                rmsd = (
                    float(np.sqrt(np.mean((coords - ref_coords) ** 2)))
                    if ref_coords is not None and coords.shape == ref_coords.shape
                    else None
                )
                runs.append(
                    {
                        'molecule': name,
                        'atoms': ref['atoms'],
                        'sigma': float(sigma),
                        'seed': seed,
                        'converged': r['converged'],
                        'steps': r['steps'],
                        'final_energy': r['final_energy'],
                        'final_true_gmax': r['final_true_gmax'],
                        'final_true_grms': r['final_true_grms'],
                        'rmsd_from_clean': rmsd,
                        'denergy_from_clean': (
                            r['final_energy'] - clean['final_energy']
                        ),
                    }
                )
            # Brief per-level progress line.
            sel = [
                x for x in runs if x['molecule'] == name and x['sigma'] == float(sigma)
            ]
            cr = np.mean([x['converged'] for x in sel])
            print(
                f'    sigma={sigma:.0e}: conv_rate={cr:.2f} '
                f'steps={np.mean([x["steps"] for x in sel]):.1f}',
                flush=True,
            )

    wall = time.perf_counter() - t_start
    meta = {
        'benchmark': 'baker',
        'solver': 'GFN2-xTB (tblite)',
        'noise_model': 'additive Gaussian on gradients (a.u.), energy clean',
        'levels': levels,
        'seeds': seeds,
        'maxsteps': args.maxsteps,
        'gradientmax': GRADIENTMAX,
        'gradientrms': GRADIENTRMS,
        'n_molecules': len(names),
        'wall_seconds': wall,
    }
    (args.out_dir / 'raw.json').write_text(
        json.dumps({'meta': meta, 'runs': runs}, indent=2)
    )
    print(f'\nWrote {args.out_dir / "raw.json"} ({wall:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
