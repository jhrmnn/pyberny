#!/usr/bin/env python3
"""Per-step internal-coordinate health check for the problematic estradiol seeds.

For each step of the optimization, capture:

- The Wilson B-matrix (n_internal x 3N).
- SVD of B B^T: the singular value spectrum, plus the location/magnitude of
  the first "pseudoinverse gap" (consecutive-ratio > 1000) that
  ``berny.Math.pinv`` would truncate at.
- Which specific internal coordinates contribute most to the left singular
  vector at that gap (i.e. which internals are going degenerate).
- Any angle > 175 deg (near-linear; would make adjacent dihedrals ill-defined),
  and any dihedral whose containing angles are > 175 deg.

Outputs ``per_step.json`` (compact per-step report) and a console summary
per case. Verbose log of pyberny itself goes to
``experiments/microvariations/internals_diag/log_*.txt``.
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
from berny.coords import Angle, Bond, Dihedral, InternalCoords, angstrom

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / 'tests' / 'data' / 'birkholz_schlegel'
OUT_DIR = REPO_ROOT / 'experiments' / 'microvariations' / 'internals_diag'

CASES = [
    {'label': 'basin 4 (high)', 'sigma': 0.01, 'seed': 0},
    {'label': 'basin 3', 'sigma': 0.01, 'seed': 8},
    {'label': 'basin 0 (deepest, sanity)', 'sigma': 0.05, 'seed': 6},
]
MAXSTEPS = 30
NEAR_LINEAR_DEG = 175.0


def perturb(geom, sigma, seed):
    rng = np.random.default_rng(seed)
    new_coords = geom.coords + rng.normal(scale=sigma, size=geom.coords.shape)
    return geomlib.Geometry(list(geom.species), new_coords, geom.lattice)


def mopac_solver():
    tmpdir = tempfile.mkdtemp()
    kcal = 1 / 627.503
    try:
        atoms, lattice = yield
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
            atoms, lattice = yield energy * kcal, gradients * kcal / angstrom
    finally:
        shutil.rmtree(tmpdir)


def classify_coord(coord):
    if isinstance(coord, Bond):
        return f'Bond({coord.i}-{coord.j})'
    if isinstance(coord, Angle):
        return f'Angle({coord.i}-{coord.j}-{coord.k})'
    if isinstance(coord, Dihedral):
        return f'Dihedral({coord.i}-{coord.j}-{coord.k}-{coord.l})'
    return repr(coord)


def first_big_gap(D, thre=1e3):
    """Mirror berny.Math.pinv: return (index_of_first_gap > thre, value)."""
    gaps = D[:-1] / D[1:]
    above = np.flatnonzero(gaps > thre)
    if len(above) == 0:
        return None, None
    n = int(above[0])
    return n, float(gaps[n])


def analyse_step(coords_obj: InternalCoords, geom):
    """Return a dict of diagnostics for the current geometry's B-matrix."""
    B = coords_obj.B_matrix(geom)  # shape (n_internal, 3N)
    BBT = B @ B.T
    # eigvalsh + sort descending == singular values of a PSD matrix
    eigs = np.linalg.eigvalsh(BBT)
    eigs = np.sort(eigs)[::-1]
    # Avoid zero division; clip from below
    pos = np.maximum(eigs, 1e-30)
    n_gap, gap = first_big_gap(pos, thre=1e3)

    # Identify which internals contribute most to the left singular vector
    # at the gap (the direction that gets truncated first past the gap).
    # B B^T is symmetric, so eigenvectors are the left singular vectors of B.
    # np.linalg.eigh returns eigenvalues in ascending order; we want the one
    # whose eigenvalue corresponds to index n_gap+1 in our descending ordering.
    if n_gap is not None and n_gap + 1 < len(pos):
        ev, vecs = np.linalg.eigh(BBT)
        # ascending order: smallest first; we want the one matching pos[n_gap+1]
        target_eig = eigs[n_gap + 1]
        idx = int(np.argmin(np.abs(ev - target_eig)))
        truncated_vec = vecs[:, idx]
        contributions = truncated_vec**2
        order = np.argsort(-contributions)
        top_contrib = [
            (int(i), float(contributions[i]), classify_coord(coords_obj._coords[i]))
            for i in order[:5]
        ]
    else:
        top_contrib = []

    # Geometric red flags: angles near 180, dihedrals around near-linear angles
    near_linear_angles = []
    for i, c in enumerate(coords_obj._coords):
        if isinstance(c, Angle):
            phi = float(np.degrees(c.eval(geom.coords)))
            if phi > NEAR_LINEAR_DEG:
                near_linear_angles.append((i, phi, classify_coord(c)))

    near_linear_dihedrals = []
    for i, c in enumerate(coords_obj._coords):
        if isinstance(c, Dihedral):
            phi_ijk = float(np.degrees(Angle(c.i, c.j, c.k).eval(geom.coords)))
            phi_jkl = float(np.degrees(Angle(c.j, c.k, c.l).eval(geom.coords)))
            if phi_ijk > NEAR_LINEAR_DEG or phi_jkl > NEAR_LINEAR_DEG:
                near_linear_dihedrals.append((i, phi_ijk, phi_jkl, classify_coord(c)))

    return {
        'sv_max': float(pos[0]),
        'sv_min_nonzero': (
            float(pos[: n_gap + 1].min()) if n_gap is not None else float(pos.min())
        ),
        'condition_number': float(pos[0] / pos[-1]) if pos[-1] > 0 else float('inf'),
        'pinv_gap_index': n_gap,
        'pinv_gap_value': gap,
        'sv_at_gap': float(pos[n_gap]) if n_gap is not None else None,
        'sv_just_after_gap': float(pos[n_gap + 1]) if n_gap is not None else None,
        'n_truncated': len(pos) - (n_gap + 1) if n_gap is not None else 0,
        'top_5_truncated_contribs': top_contrib,
        'near_linear_angles': near_linear_angles,
        'near_linear_dihedrals_count': len(near_linear_dihedrals),
        'near_linear_dihedrals_sample': near_linear_dihedrals[:3],
    }


def run_one(label, sigma, seed, base_geom):
    geom = perturb(base_geom, sigma, seed)
    log_path = OUT_DIR / f'log_{sigma}_{seed}.txt'
    handler = logging.FileHandler(log_path, mode='w')
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.setLevel(logging.INFO)
    berny_logger = logging.getLogger('berny')
    berny_logger.setLevel(logging.INFO)
    berny_logger.addHandler(handler)
    per_step = []
    try:
        berny = Berny(geom, maxsteps=MAXSTEPS)
        coords_obj = berny._state.coords
        solver = mopac_solver()
        next(solver)
        for current in berny:
            diag = analyse_step(coords_obj, current)
            per_step.append(diag)
            print(
                f'  step {len(per_step):3d}: pinv_gap_idx={diag["pinv_gap_index"]} '
                f'gap={diag["pinv_gap_value"]} '
                f'sv_min={diag["sv_min_nonzero"]:.3e} '
                f'#near-linear-angles={len(diag["near_linear_angles"])} '
                f'#near-linear-dihedrals={diag["near_linear_dihedrals_count"]}'
            )
            energy, gradients = solver.send((list(current), current.lattice))
            berny.send((energy, gradients))
        return berny.converged, berny._n, per_step
    finally:
        handler.close()
        berny_logger.removeHandler(handler)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_geom = geomlib.readfile(str(DATA / 'estradiol.xyz'))

    report = {}
    for case in CASES:
        print(f'\n=== {case["label"]}: sigma={case["sigma"]} seed={case["seed"]} ===')
        converged, n_steps, per_step = run_one(
            case['label'], case['sigma'], case['seed'], base_geom
        )
        report[case['label']] = {
            'sigma': case['sigma'],
            'seed': case['seed'],
            'converged': converged,
            'steps': n_steps,
            'per_step': per_step,
        }
        if per_step:
            last = per_step[-1]
            print(
                f'  FINAL: pinv_gap at index {last["pinv_gap_index"]} '
                f'(value {last["pinv_gap_value"]}), '
                f'sv truncated: {last["sv_at_gap"]} -> {last["sv_just_after_gap"]}, '
                f'#truncated dirs: {last["n_truncated"]}'
            )
            print('  Top contributors to the FIRST truncated direction:')
            for _idx, w, desc in last['top_5_truncated_contribs']:
                print(f'    {desc:35} weight={w:.3f}')

    (OUT_DIR / 'per_step.json').write_text(json.dumps(report, indent=2))
    print(f'\nWrote {OUT_DIR / "per_step.json"}')


if __name__ == '__main__':
    main()
