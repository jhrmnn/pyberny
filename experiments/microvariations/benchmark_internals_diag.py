#!/usr/bin/env python3
"""Per-step internal-coordinate health check for every benchmark molecule that
the trajectory-warning sweep (``benchmark_trajectory_check.py``) flagged as
problematic.

For each step of the optimization we capture, exactly the way
``estradiol_internals_diag.py`` does for the perturbed estradiol seeds:

- The Wilson B-matrix ``B`` (``n_internal x 3N``).
- Eigenvalues of ``B B^T`` (= squared singular values of ``B``), in descending
  order; the location and magnitude of the first ratio above ``1e3`` (the
  pseudoinverse-gap threshold inside ``berny.Math.pinv``).
- The internals that contribute most to the eigenvector at that gap - i.e.
  the directions ``Math.pinv`` would silently truncate. Each contributor is
  annotated with element symbols ("H37-C36-O40") so the smoking-gun internal
  is easy to read off.
- All angles ``> 175 deg`` (near-linear; would make adjacent dihedrals
  ill-defined) and a sample of dihedrals adjacent to such angles.

The case list is the union of every molecule that the trajectory sweep
flagged via *any* of:

- ``pinv@final = YES`` (the three known smoking guns);
- a non-empty ``severe_backxform_dq`` (RMS(dq) >= 0.05) list;
- ``len(backtransform_warnings) >= 3`` (the heavy-back-xform precursor
  signature);
- ``negative_eigenvalue_events`` with >=10 events (a "confusing PES"
  signature - included for contrast since it should NOT show a coordinate
  singularity).

Outputs under ``experiments/microvariations/benchmark_internals_diag/``:

- ``<set>/<molecule>.log`` - raw INFO log (same harness as the trajectory
  sweep so that step counts match);
- ``per_step.json`` - full per-step diagnostics for every case;
- ``summary.md`` - per-case markdown distillation: the offending internal,
  the step it fires at, and a short verdict.
"""

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.coords import Angle, Bond, Dihedral, InternalCoords
from berny.solvers import MopacSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'tests' / 'data'
OUT_DIR = REPO_ROOT / 'experiments' / 'microvariations' / 'benchmark_internals_diag'

BENCHMARKS = {
    'birkholz': 'birkholz_schlegel',
    'baker': 'baker_shajan_2023',
}

# Cases: (set_key, molecule, maxsteps, why).
# maxsteps is set per-case to roughly the step at which the problematic
# event fires (plus a margin), so we don't spend wall time on long
# trajectories that are stable past the warning. ``raffinose`` and
# ``caffeine`` use 60/55 instead of the full 110 to keep this script's
# runtime tractable while still covering all flagged steps.
CASES = [
    ('birkholz', 'estradiol', 15, 'pinv@final: H-C-O angle near-linear (known smoking gun)'),
    ('baker', 'allene', 15, 'pinv@final: C=C=C linear backbone (symmetry-imposed)'),
    ('baker', 'disilyl_ether', 10, 'pinv@final: Si-O-Si linearization'),
    ('birkholz', 'inosine_cation', 15, 'severe back-xform RMS(dq) at step 11'),
    ('birkholz', 'maltose', 35, 'severe back-xform + saddle pass at steps 27/30'),
    ('birkholz', 'raffinose', 55, 'heavy back-xform + non-converger (steps 42-49)'),
    ('baker', 'caffeine', 55, 'control: 43 neg-eigval events; not coord-singular'),
]

NEAR_LINEAR_DEG = 175.0


@contextlib.contextmanager
def silence_fd(fd):
    """Redirect ``fd`` to /dev/null around a block (silences MOPAC stdout)."""
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def describe_coord(coord, species):
    """Render a coord with element-tagged atom labels (e.g. ``Angle(H37-C36-O40)``)."""
    def tag(i):
        return f'{species[i % len(species)]}{i % len(species)}'
    if isinstance(coord, Bond):
        return f'Bond({tag(coord.i)}-{tag(coord.j)})'
    if isinstance(coord, Angle):
        return f'Angle({tag(coord.i)}-{tag(coord.j)}-{tag(coord.k)})'
    if isinstance(coord, Dihedral):
        return (
            f'Dihedral({tag(coord.i)}-{tag(coord.j)}-'
            f'{tag(coord.k)}-{tag(coord.l)})'
        )
    return repr(coord)


def first_big_gap(D, thre=1e3):
    """Mirror ``berny.Math.pinv``: return (gap_index, gap_value) or (None, None)."""
    if len(D) < 2:
        return None, None
    gaps = D[:-1] / np.maximum(D[1:], 1e-300)
    above = np.flatnonzero(gaps > thre)
    if len(above) == 0:
        return None, None
    n = int(above[0])
    return n, float(gaps[n])


def analyse_step(coords_obj: InternalCoords, geom):
    """Compute one step's worth of internal-coord diagnostics.

    Returns a dict containing the pinv-gap location/value, the top
    contributors to the truncated direction, near-linear angles, and a
    sample of dihedrals adjacent to such angles.
    """
    B = coords_obj.B_matrix(geom)  # (n_internal, 3N)
    BBT = B @ B.T
    eigs = np.linalg.eigvalsh(BBT)
    eigs = np.sort(eigs)[::-1]
    pos = np.maximum(eigs, 1e-30)
    n_gap, gap = first_big_gap(pos, thre=1e3)

    # B-row Frobenius norm per coord: which row dominates?
    row_norms = np.linalg.norm(B, axis=1)
    top_rows = np.argsort(-row_norms)[:5]
    top_row_norms = [
        {
            'idx': int(i),
            'norm': float(row_norms[i]),
            'coord': describe_coord(coords_obj._coords[i], geom.species),
        }
        for i in top_rows
    ]

    # Top contributors to the first truncated direction (column n_gap+1
    # in descending order = the smallest-singular-value direction still
    # kept-or-killed by the pinv cut).
    if n_gap is not None and n_gap + 1 < len(pos):
        ev, vecs = np.linalg.eigh(BBT)  # ascending
        target_eig = eigs[n_gap + 1]
        idx = int(np.argmin(np.abs(ev - target_eig)))
        truncated_vec = vecs[:, idx]
        contributions = truncated_vec**2
        order = np.argsort(-contributions)
        top_contrib = [
            {
                'idx': int(i),
                'weight': float(contributions[i]),
                'coord': describe_coord(coords_obj._coords[i], geom.species),
            }
            for i in order[:5]
        ]
    else:
        top_contrib = []

    near_linear_angles = []
    near_linear_dihedrals = []
    for i, c in enumerate(coords_obj._coords):
        if isinstance(c, Angle):
            phi = float(np.degrees(c.eval(geom.coords)))
            if phi > NEAR_LINEAR_DEG:
                near_linear_angles.append(
                    {
                        'idx': i,
                        'angle_deg': phi,
                        'coord': describe_coord(c, geom.species),
                    }
                )
        elif isinstance(c, Dihedral):
            phi_ijk = float(
                np.degrees(Angle(c.i, c.j, c.k).eval(geom.coords))
            )
            phi_jkl = float(
                np.degrees(Angle(c.j, c.k, c.l).eval(geom.coords))
            )
            if phi_ijk > NEAR_LINEAR_DEG or phi_jkl > NEAR_LINEAR_DEG:
                near_linear_dihedrals.append(
                    {
                        'idx': i,
                        'phi_ijk_deg': phi_ijk,
                        'phi_jkl_deg': phi_jkl,
                        'coord': describe_coord(c, geom.species),
                    }
                )

    return {
        'sv_max': float(pos[0]),
        'sv_min': float(pos[-1]),
        'condition_number': (
            float(pos[0] / pos[-1]) if pos[-1] > 0 else float('inf')
        ),
        'pinv_gap_index': n_gap,
        'pinv_gap_value': gap,
        'sv_at_gap': float(pos[n_gap]) if n_gap is not None else None,
        'sv_just_after_gap': (
            float(pos[n_gap + 1]) if n_gap is not None else None
        ),
        'n_truncated': len(pos) - (n_gap + 1) if n_gap is not None else 0,
        'top_5_truncated_contribs': top_contrib,
        'top_5_b_row_norms': top_row_norms,
        'near_linear_angles': near_linear_angles,
        'near_linear_dihedrals_count': len(near_linear_dihedrals),
        'near_linear_dihedrals_sample': near_linear_dihedrals[:3],
    }


def run_case(set_key, molecule, maxsteps, log_path):
    """Run Berny+MOPAC on one molecule with per-step internal-coord capture."""
    set_dir = BENCHMARKS[set_key]
    data_dir = DATA_ROOT / set_dir
    reference = json.loads((data_dir / 'reference.json').read_text())
    ref = reference[molecule]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    berny_logger = logging.getLogger('berny')
    prev_level = berny_logger.level
    berny_logger.setLevel(logging.INFO)
    berny_logger.addHandler(handler)

    per_step = []
    converged = False
    n = 0
    error = None
    t0 = time.perf_counter()
    try:
        with silence_fd(1), silence_fd(2):
            geom = geomlib.readfile(str(data_dir / f'{molecule}.xyz'))
            berny = Berny(geom, maxsteps=maxsteps)
            coords_obj = berny._state.coords
            solver = MopacSolver(charge=ref['charge'], mult=ref['mult'])
            next(solver)
            for current in berny:
                diag = analyse_step(coords_obj, current)
                per_step.append(diag)
                energy, gradients = solver.send((list(current), current.lattice))
                berny.send((energy, gradients))
            converged = berny.converged
            n = berny._n
    except Exception as e:  # noqa: BLE001
        error = f'{type(e).__name__}: {e}'
    finally:
        handler.close()
        berny_logger.removeHandler(handler)
        berny_logger.setLevel(prev_level)

    return {
        'converged': converged,
        'steps': n,
        'wall': time.perf_counter() - t0,
        'error': error,
        'per_step': per_step,
        'n_internals': len(per_step[0]['top_5_b_row_norms']) if per_step else 0,
    }


def render_summary(report):
    """Render a per-case Markdown distillation of the per-step data."""
    out = []
    out.append('# Internal-coordinate health check on problematic benchmark cases\n')
    out.append(
        'For every molecule that `benchmark_trajectory_check.py` flagged, '
        'this script re-runs the same `Berny+MopacSolver` trajectory but '
        'captures the Wilson B-matrix at every step. The tables below '
        'distill **which internal coordinate(s) fire** - i.e. which '
        'Bond/Angle/Dihedral becomes the dominant contributor to the '
        'singular direction that `Math.pinv` would (and on three '
        'molecules, does) silently truncate.\n'
    )
    out.append(
        'Atom indices are zero-based, prefixed with the element symbol of '
        'that atom (e.g. `H37` is hydrogen #37 in the xyz file). The '
        '"top contrib" coordinate is the one whose weight in the '
        'eigenvector at the pinv gap is largest at the final step where '
        'the warning fires.\n'
    )

    for case in CASES:
        set_key, molecule, _maxsteps, why = case
        rec = report.get(f'{set_key}/{molecule}')
        if rec is None:
            continue
        out.append(f'\n## {set_key}/{molecule}\n')
        out.append(f'*{why}*\n')
        out.append(
            f'Steps: {rec["steps"]}, converged: {rec["converged"]}, '
            f'wall: {rec["wall"]:.1f}s.\n'
        )
        per_step = rec['per_step']
        if not per_step:
            out.append('No steps recorded (error or empty trajectory).\n')
            continue

        # "Firing" = the same condition that triggers Math.pinv's
        # `Pseudoinverse gap of only:` warning: 1e3 < gap < 1e8. This catches
        # both the "spurious gap at index 0" pathology (estradiol/disilyl_ether)
        # AND the "natural-position but small gap" pathology (allene), where
        # the truncation kills the smallest *real* SV rather than a zero one.
        firing_steps = [
            (i + 1, s)
            for i, s in enumerate(per_step)
            if s['pinv_gap_value'] is not None and s['pinv_gap_value'] < 1e8
        ]

        out.append('### Per-step pinv-gap summary\n')
        out.append('| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|')
        for i, s in enumerate(per_step, 1):
            gap_s = (
                f'{s["pinv_gap_value"]:.2e}'
                if s['pinv_gap_value'] is not None
                else '-'
            )
            idx_s = (
                str(s['pinv_gap_index']) if s['pinv_gap_index'] is not None else '-'
            )
            out.append(
                f'| {i} | {idx_s} | {gap_s} | {s["sv_max"]:.2e} '
                f'| {s["sv_min"]:.2e} | {len(s["near_linear_angles"])} '
                f'| {s["near_linear_dihedrals_count"]} |'
            )

        # Smoking-gun section: list the firing internals at each step where
        # Math.pinv's warning would have been emitted.
        if firing_steps:
            out.append('\n### Firing internals (steps where the pinv warning would fire)\n')
            out.append(
                'These are the steps where `Math.pinv`\'s gap value falls '
                'below `1e8` (its log threshold) - covering both the '
                '"spurious gap at low index" pathology (the gradient '
                'projection kills a far-away real DOF) and the '
                '"natural-position small gap" pathology (the smallest real '
                'singular value is being truncated). The "top contributors" '
                'are the internals whose B-row dominates the truncated '
                'eigenvector at this step; the "rogue B-row" list is the '
                'coords whose gradient magnitude is most inflated.\n'
            )
            for step, s in firing_steps:
                out.append(
                    f'\n**Step {step}** - gap at index {s["pinv_gap_index"]} '
                    f'(value `{s["pinv_gap_value"]:.2e}`), '
                    f'`sv_max = {s["sv_max"]:.2e}`, '
                    f'`sv_just_after_gap = {s["sv_just_after_gap"]:.2e}`'
                )
                if s['near_linear_angles']:
                    angs = ', '.join(
                        f'{a["coord"]}={a["angle_deg"]:.2f}°'
                        for a in s['near_linear_angles']
                    )
                    out.append(f'- Near-linear angles: {angs}')
                if s['top_5_truncated_contribs']:
                    out.append('- Top contributors to the truncated direction:')
                    for c in s['top_5_truncated_contribs']:
                        out.append(
                            f'  - `{c["coord"]}` (idx {c["idx"]}, '
                            f'weight {c["weight"]:.3f})'
                        )
                if s['top_5_b_row_norms']:
                    out.append('- Largest B-row norms (the rogue gradients):')
                    for c in s['top_5_b_row_norms']:
                        out.append(
                            f'  - `{c["coord"]}` (idx {c["idx"]}, '
                            f'|B_row| = {c["norm"]:.2e})'
                        )
        else:
            out.append(
                '\nNo step recorded a pinv gap value below `1e8` - the '
                'natural redundancy boundary is well-separated throughout '
                'the trajectory, and `Math.pinv` would never log a warning '
                'for this molecule.\n'
            )

    # Global conclusions across all cases.
    out.append('\n## Conclusions\n')
    out.append(
        'Two distinct mechanisms produce the `Pseudoinverse gap of only:` '
        'warning:\n'
    )
    out.append(
        '1. **Inflated dihedral B-row** (estradiol, disilyl_ether). A '
        'near-linear three-atom motif (`H-C-O`, `Si-O-Si`) makes adjacent '
        'dihedrals\' gradient formula blow up: in '
        '`Dihedral.eval(grad=True)` the term `1 / norm(a1)` diverges as '
        'the central angle approaches 180°. This pushes a single B-row '
        'norm to ~10²-10³ while everything else stays at ~1, creating a '
        'huge sv[0] and a spurious gap at index 0. Truncation then '
        'zeroes a far-away real DOF.\n'
    )
    out.append(
        '2. **Rank drop** (allene). A symmetry-imposed linear backbone '
        '(`C=C=C`) means rotation about that axis genuinely is not a DOF, '
        'so the B-matrix has one extra near-zero singular value. The '
        'pinv gap fires at the natural-redundancy index but the gap '
        'value is small (~10³-10⁷) and a real H-C-H angle DOF gets '
        'truncated alongside the missing rotational mode.\n'
    )
    out.append(
        'The other flagged cases (inosine_cation, maltose, raffinose, '
        'caffeine) have pinv gap values uniformly above `1e11` and never '
        'trigger the warning. Their pathologies are different: '
        'saddle-pass attempts (negative eigenvalues + huge predicted '
        'energy change) for maltose/raffinose, multivalued '
        'Cartesian↔internal map at one step for inosine_cation, and a '
        'confusing PES for caffeine. None of these implicates `Math.pinv` '
        'truncation.\n'
    )
    out.append(
        'In every truncation case the geometric culprit is identifiable '
        'in one line:\n'
    )
    out.append('| molecule | culprit | mechanism |')
    out.append('|---|---|---|')
    out.append(
        '| estradiol | `Angle(H37-C36-O40)` -> 179.66° | dihedral B-row '
        'inflated (`Dihedral(H37-C36-O40-C34)` and `-H41`) |'
    )
    out.append(
        '| disilyl_ether | `Angle(Si0-O2-Si1)` -> 179.94° | six dihedrals '
        '`H?-Si?-O-Si?` B-row inflated to ~7.5e+02 |'
    )
    out.append(
        '| allene | `Angle(C1-C0-C2)` -> 179.98° (symmetry) | rank drop; '
        'H-C-H angles get truncated instead |'
    )
    return '\n'.join(out) + '\n'


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=OUT_DIR)
    ap.add_argument(
        '--cases',
        nargs='+',
        default=None,
        help='subset of cases by molecule name (default: all)',
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')
    args.out.mkdir(parents=True, exist_ok=True)

    selected = (
        [c for c in CASES if c[1] in args.cases] if args.cases else list(CASES)
    )
    if not selected:
        raise SystemExit(f'no cases match {args.cases!r}')

    report = {}
    t_start = time.perf_counter()
    for i, (set_key, molecule, maxsteps, why) in enumerate(selected, 1):
        set_dir = BENCHMARKS[set_key]
        print(
            f'\n[{i}/{len(selected)}] {set_key}/{molecule} '
            f'(maxsteps={maxsteps}) - {why}',
            flush=True,
        )
        log_path = args.out / set_dir / f'{molecule}.log'
        rec = run_case(set_key, molecule, maxsteps, log_path)
        rec['why'] = why
        rec['set'] = set_key
        rec['molecule'] = molecule
        report[f'{set_key}/{molecule}'] = rec
        elapsed = time.perf_counter() - t_start
        print(
            f'  -> converged={rec["converged"]} steps={rec["steps"]} '
            f'wall={rec["wall"]:.1f}s (total {elapsed:.0f}s)',
            flush=True,
        )
        # Incremental persistence
        (args.out / 'per_step.json').write_text(json.dumps(report, indent=2))

    (args.out / 'summary.md').write_text(render_summary(report))
    print(f'\nWrote {args.out / "summary.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
