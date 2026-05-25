#!/usr/bin/env python3
"""Re-run estradiol seeds that landed in 'basin 3' and 'basin 4' with full
pyberny verbosity, then dump the optimizer state at termination so we can see
exactly which convergence criteria fired and what the BFGS Hessian looks like.

The microvariation experiment found that linear-interpolation paths from these
two basins toward the deepest PM7 minimum go monotonically downhill - which
should mean these aren't true minima. This diagnostic captures:

- pyberny's per-step verbose log (energy, predicted step, convergence criteria
  values vs thresholds);
- the final internal-coordinate gradient norm/max;
- the BFGS Hessian's eigenvalue spectrum at termination;
- the Cartesian gradient at the final geometry, projected onto the Cartesian
  direction toward the deeper basin (basin 0).
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.coords import angstrom

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / 'src' / 'berny' / 'benchmarks' / 'birkholz_schlegel'
OUT_DIR = REPO_ROOT / 'experiments' / 'microvariations'

# Two pivotal seeds plus a sanity-check seed (basin 0, the deepest):
CASES = [
    {
        'label': 'basin 4 (high)',
        'sigma': 0.01,
        'seed': 0,
        'E_expected': -0.151174,
    },
    {'label': 'basin 3', 'sigma': 0.01, 'seed': 8, 'E_expected': -0.153738},
    {
        'label': 'basin 0 (deepest, sanity)',
        'sigma': 0.05,
        'seed': 6,
        'E_expected': -0.158928,
    },
]


def perturb(geom, sigma, seed):
    rng = np.random.default_rng(seed)
    new_coords = geom.coords + rng.normal(scale=sigma, size=geom.coords.shape)
    return geomlib.Geometry(list(geom.species), new_coords, geom.lattice)


def mopac_solver_capture():
    """A MopacSolver clone that also returns the gradients we sent the optimizer.
    (Identical SCF parameters as the original; needed only because we want to
    grab the last-call gradients alongside the BFGS state.)"""
    tmpdir = tempfile.mkdtemp()
    kcal = 1 / 627.503
    state = {'last_grad_cart': None}
    try:
        atoms, lattice = yield state
        while True:
            mopac_input = 'PM7 1SCF GRADIENTS\n\n\n' + '\n'.join(
                f'{el} {x} 1 {y} 1 {z} 1' for el, (x, y, z) in atoms
            )
            with open(os.path.join(tmpdir, 'job.mop'), 'w') as f:
                f.write(mopac_input)
            subprocess.check_call(
                ['mopac', os.path.join(tmpdir, 'job.mop')],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(os.path.join(tmpdir, 'job.out')) as f:
                energy = float(
                    next(
                        line for line in f if 'FINAL HEAT OF FORMATION' in line
                    ).split()[5]
                )
                next(line for line in f if 'FINAL  POINT  AND  DERIVATIVES' in line)
                next(f)
                next(f)
                gradients = np.array(
                    [
                        [float(next(f).split()[6]) for _ in range(3)]
                        for _ in range(len(atoms))
                    ]
                )
            state['last_grad_cart'] = gradients * kcal / angstrom
            atoms, lattice = yield energy * kcal, gradients * kcal / angstrom
    finally:
        shutil.rmtree(tmpdir)


def run_with_logging(geom, ref, log_path):
    """Run one pyberny optimization in debug mode, with per-step log captured to file.

    Returns (converged, n_steps, last_geom, last_energy, debug_state, last_grad_cart).
    """
    handler = logging.FileHandler(log_path, mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    berny_logger = logging.getLogger('berny')
    berny_logger.setLevel(logging.INFO)
    berny_logger.addHandler(handler)
    try:
        berny = Berny(geom, maxsteps=120, debug=True)
        solver = mopac_solver_capture()
        solver_state = next(solver)
        debug_state = None
        last_geom = geom
        last_energy = None
        for current in berny:
            last_geom = current
            energy, gradients = solver.send((list(current), current.lattice))
            last_energy = energy
            debug_state = berny.send((energy, gradients))
        return (
            berny.converged,
            berny._n,
            last_geom,
            last_energy,
            debug_state,
            solver_state['last_grad_cart'],
        )
    finally:
        handler.close()
        berny_logger.removeHandler(handler)


def kabsch_align(target, mobile):
    t_c = target.mean(axis=0)
    m_c = mobile.mean(axis=0)
    h = (target - t_c).T @ (mobile - m_c)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (mobile - m_c) @ rot.T + t_c


def main():
    out_dir = OUT_DIR / 'verbose_diag'
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads((DATA / 'reference.json').read_text())['estradiol']
    base_geom = geomlib.readfile(str(DATA / 'estradiol.xyz'))

    # Load basin 0 final coords (deepest minimum, as our "reference direction")
    coords_db = json.load(open(OUT_DIR / 'final_coords.json'))
    basin0_coords = np.array(coords_db['estradiol|0.05|6'])

    report = {}
    for case in CASES:
        label = case['label']
        print(f'\n=== {label}: sigma={case["sigma"]}, seed={case["seed"]} ===')
        log_path = out_dir / f'log_{case["sigma"]}_{case["seed"]}.txt'
        geom = perturb(base_geom, case['sigma'], case['seed'])
        converged, n, last_geom, last_E, dbg, grad_cart = run_with_logging(
            geom, reference, log_path
        )
        print(f'  converged={converged}, steps={n}, E={last_E:.6f}')
        print(f'  log -> {log_path}')

        H = np.array(dbg['H'])
        prev = dbg['previous']  # OptPoint with q (internals) and g (internal gradient)
        future = dbg['future']  # OptPoint with q only

        n_internals = H.shape[0]
        grad_internal_max = float(np.max(np.abs(prev.g)))
        grad_internal_rms = float(np.sqrt(np.mean(prev.g**2)))
        step_internal = np.array(future.q) - np.array(prev.q)
        step_max = float(np.max(np.abs(step_internal)))
        step_rms = float(np.sqrt(np.mean(step_internal**2)))

        eigs = np.linalg.eigvalsh(H)
        eigs_sorted = np.sort(eigs)

        # Cartesian gradient at final geometry, in atomic units (Ha/bohr).
        # Project it onto the Cartesian direction toward basin 0 (deepest).
        aligned_basin0 = kabsch_align(last_geom.coords, basin0_coords)
        direction = (aligned_basin0 - last_geom.coords) * angstrom  # in bohr
        direction_flat = direction.reshape(-1)
        unit = direction_flat / np.linalg.norm(direction_flat)
        grad_flat = grad_cart.reshape(-1)
        # grad_cart is Ha/bohr; projection has the same units
        projected_grad = float(grad_flat @ unit)
        # Convert to energy/distance change estimate along the path
        # E change per unit_path = projected_grad * |direction in bohr|
        dist_to_basin0_bohr = float(np.linalg.norm(direction_flat))

        print(
            f'  internal-coord gradient: '
            f'max={grad_internal_max:.3e}  rms={grad_internal_rms:.3e}'
        )
        print('      thresholds:         max=4.50e-4 rms=1.50e-4')
        print(f'  predicted step:          max={step_max:.3e}  rms={step_rms:.3e}')
        print('      thresholds:         max=1.80e-3 rms=1.20e-3')
        print(f'  BFGS Hessian eigvals: smallest 5 = {eigs_sorted[:5]}')
        print(f'                        largest 3 = {eigs_sorted[-3:]}')
        gnorm = np.linalg.norm(grad_flat)
        print('  Cartesian gradient projected onto direction-to-basin0:')
        print(f'      g . unit = {projected_grad:.3e} Ha/bohr')
        print(f'      |g_full| = {gnorm:.3e} Ha/bohr')
        print(f'      cosine of angle = {projected_grad / gnorm:.4f}')
        print(
            f'      distance to basin 0 along this line = '
            f'{dist_to_basin0_bohr:.2f} bohr'
        )

        report[label] = {
            'sigma': case['sigma'],
            'seed': case['seed'],
            'converged': converged,
            'steps': n,
            'energy': last_E,
            'grad_internal_max': grad_internal_max,
            'grad_internal_rms': grad_internal_rms,
            'step_max': step_max,
            'step_rms': step_rms,
            'hessian_eigvals_sorted': eigs_sorted.tolist(),
            'cart_grad_norm_Ha_per_bohr': float(np.linalg.norm(grad_flat)),
            'projected_grad_toward_basin0_Ha_per_bohr': projected_grad,
            'cosine_with_basin0_direction': projected_grad / np.linalg.norm(grad_flat),
            'distance_to_basin0_bohr': dist_to_basin0_bohr,
            'n_internals': n_internals,
            'log_path': str(log_path.relative_to(REPO_ROOT)),
        }

    (out_dir / 'diagnostics.json').write_text(json.dumps(report, indent=2))
    print(f'\nWrote {out_dir / "diagnostics.json"}')


if __name__ == '__main__':
    main()
