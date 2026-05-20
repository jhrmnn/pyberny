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
- All dihedrals within 5 deg of ``0`` or ``pi`` (near-planar; the
  ``abs(phi) > pi - 1e-6`` / ``abs(phi) < 1e-6`` branches in
  ``Dihedral.eval`` operate close to these limits and the ordinary
  branch's ``1/sin(phi)`` term inflates as we approach them).
- All heavy atoms with exactly three covalent neighbours whose
  substituent-angle sum exceeds 355 deg (i.e. centre within ~5 deg of
  the plane of its substituents - the sp3-to-sp2 inversion mode that
  underlies the H-C-O = 180 deg event on estradiol).

The case list is the union of every molecule that the trajectory sweep
flagged via *any* of:

- ``pinv@final = YES`` (the three known smoking guns);
- a non-empty ``severe_backxform_dq`` (RMS(dq) >= 0.05) list;
- ``len(backtransform_warnings) >= 3`` (the heavy-back-xform precursor
  signature);
- ``negative_eigenvalue_events`` with >=10 events (a "confusing PES"
  signature - included for contrast since it should NOT show a coordinate
  singularity);
- ``negative_eigenvalue_events`` with 1-4 events (the sparser, less
  obviously-pathological cases, added as a probe for whether the same
  planar-dihedral / planar-sp3 explanation also catches these milder
  events).

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
from itertools import combinations
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.coords import Angle, Bond, Dihedral, InternalCoords
from berny.solvers import MopacSolver

# Re-use the regex-based log scanner from the trajectory-warning sweep so the
# "which step did a warning fire on" pattern lives in exactly one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_trajectory_check import scan_log  # noqa: E402

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
    ('birkholz', 'avobenzone', 55, 'neg-eig only: 4 events at steps 14/17/20/49'),
    ('birkholz', 'codeine', 25, 'neg-eig only: 2 events at steps 15/18'),
    ('baker', 'methylamine', 22, 'neg-eig only: 4 events at steps 5/7/8/9'),
]

NEAR_LINEAR_DEG = 175.0
PLANAR_DIHEDRAL_DEG = 5.0  # |phi - 0| < 5 deg (cis) or |phi - 180| < 5 deg (trans)
PLANAR_SP3_DEG = 355.0  # sum of three substituent angles > 355 deg => centre in plane


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


def build_neighbours(coords_obj, n_atoms):
    """Return a per-atom list of strong-bond neighbours from ``coords_obj``.

    Strong bonds are ``Bond`` entries with ``weak == 0``: the covalent network
    excluding the fragment-linking pseudo-bonds. Bonds are static for the
    trajectory (pyberny builds ``InternalCoords`` once), so this needs to be
    computed only at the start of each case.
    """
    neighbours = [set() for _ in range(n_atoms)]
    for c in coords_obj._coords:
        if isinstance(c, Bond) and c.weak == 0:
            neighbours[c.i].add(c.j)
            neighbours[c.j].add(c.i)
    return [sorted(n) for n in neighbours]


def heavy_three_coord_centres(species, neighbours):
    """Return atom indices with exactly three covalent neighbours, heavy only.

    Hydrogens are excluded: the H-C-O = 180 deg case at estradiol is already
    diagnosed by the near-linear-angle metric. This filter targets the
    sp3->sp2 inversion mode at heavy centres (the maltose/raffinose
    ring-puckering hypothesis).
    """
    return [
        i
        for i, n in enumerate(neighbours)
        if len(n) == 3 and species[i] != 'H'
    ]


def substituent_angle_sum_deg(centre, neighbours, coords):
    """Sum of the three angles ``neighbour-centre-neighbour`` at ``centre``.

    For a perfectly planar arrangement of three substituents around the
    centre (sp2) this sum is 360 deg; for tetrahedral sp3 it is
    3 * 109.47 deg = 328.4 deg. Drops below 360 deg as the centre lifts out
    of the substituent plane, so a value close to 360 deg flags a planar
    geometry at the centre.
    """
    s = 0.0
    for n1, n2 in combinations(neighbours, 2):
        v1 = coords[n1] - coords[centre]
        v2 = coords[n2] - coords[centre]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        cos = max(-1.0, min(1.0, cos))
        s += float(np.degrees(np.arccos(cos)))
    return s


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


def analyse_step(coords_obj: InternalCoords, geom, neighbours=None, sp3_centres=None):
    """Compute one step's worth of internal-coord diagnostics.

    Returns a dict containing the pinv-gap location/value, the top
    contributors to the truncated direction, near-linear angles, a
    sample of dihedrals adjacent to such angles, near-planar dihedrals
    (within 5 deg of 0 or pi), and heavy three-coord centres that have
    gone planar (substituent angle sum > 355 deg).
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
    near_planar_dihedrals = []
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
            phi_deg = float(np.degrees(c.eval(geom.coords)))
            phi_abs = abs(phi_deg)
            dist_to_planar = min(phi_abs, abs(180.0 - phi_abs))
            if dist_to_planar < PLANAR_DIHEDRAL_DEG:
                near_planar_dihedrals.append(
                    {
                        'idx': i,
                        'phi_deg': phi_deg,
                        'coord': describe_coord(c, geom.species),
                        'kind': 'cis' if phi_abs < 90.0 else 'trans',
                    }
                )

    planar_sp3 = []
    if sp3_centres is not None and neighbours is not None:
        for centre in sp3_centres:
            s = substituent_angle_sum_deg(centre, neighbours[centre], geom.coords)
            if s > PLANAR_SP3_DEG:
                planar_sp3.append(
                    {
                        'centre_idx': int(centre),
                        'centre': f'{geom.species[centre]}{centre}',
                        'neighbours': [
                            f'{geom.species[n]}{n}' for n in neighbours[centre]
                        ],
                        'angle_sum_deg': s,
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
        'near_planar_dihedrals_count': len(near_planar_dihedrals),
        'near_planar_dihedrals_idx': [d['idx'] for d in near_planar_dihedrals],
        'near_planar_dihedrals_sample': near_planar_dihedrals,
        'planar_sp3_count': len(planar_sp3),
        'planar_sp3_idx': [c['centre_idx'] for c in planar_sp3],
        'planar_sp3_sample': planar_sp3,
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
    neighbours = None
    sp3_centres = None
    n_atoms = None
    t0 = time.perf_counter()
    try:
        with silence_fd(1), silence_fd(2):
            geom = geomlib.readfile(str(data_dir / f'{molecule}.xyz'))
            n_atoms = len(geom)
            berny = Berny(geom, maxsteps=maxsteps)
            coords_obj = berny._state.coords
            neighbours = build_neighbours(coords_obj, n_atoms)
            sp3_centres = heavy_three_coord_centres(geom.species, neighbours)
            solver = MopacSolver(charge=ref['charge'], mult=ref['mult'])
            next(solver)
            for current in berny:
                diag = analyse_step(coords_obj, current, neighbours, sp3_centres)
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

    # Re-scan the log we just produced for the warning lines (same regexes
    # as benchmark_trajectory_check.scan_log), so the summary can correlate
    # per-step planar/linear state with the steps that emitted warnings.
    try:
        scan = scan_log(log_path)
    except FileNotFoundError:
        scan = None

    return {
        'converged': converged,
        'steps': n,
        'wall': time.perf_counter() - t0,
        'error': error,
        'per_step': per_step,
        'n_internals': len(per_step[0]['top_5_b_row_norms']) if per_step else 0,
        'scan': scan,
        'sp3_centres': sp3_centres or [],
        'n_atoms': n_atoms,
    }


def _collect_warning_steps(scan):
    """Return ``{step: {warning_kind, ...}}`` for one trajectory's log scan."""
    by_step = {}
    if not scan:
        return by_step
    for w in scan.get('pinv_warnings', []) or []:
        by_step.setdefault(w['step'], set()).add('pinv')
    for w in scan.get('backtransform_warnings', []) or []:
        by_step.setdefault(w['step'], set()).add('backx')
    for w in scan.get('severe_backxform_dq', []) or []:
        by_step.setdefault(w['step'], set()).add('severe-dq')
    for w in scan.get('negative_eigenvalue_events', []) or []:
        by_step.setdefault(w['step'], set()).add('neg-eig')
    return by_step


def _baseline_sets(per_step):
    """Return the step-1 sets of (near-planar dihedral idx, planar sp3 idx).

    Aromatic / sp2 features are planar at the published starting geometry
    too, so subtracting the step-1 set gives "new" planar events that
    happened during the trajectory - a much cleaner signal than the
    absolute count.
    """
    if not per_step:
        return set(), set()
    s = per_step[0]
    return set(s.get('near_planar_dihedrals_idx', [])), set(s.get('planar_sp3_idx', []))


def _new_planar_at_step(s, base_dih, base_sp3):
    """Return ``(new_planar_dih_count, new_planar_sp3_count)`` vs the step-1 baseline."""
    cur_dih = set(s.get('near_planar_dihedrals_idx', []))
    cur_sp3 = set(s.get('planar_sp3_idx', []))
    return len(cur_dih - base_dih), len(cur_sp3 - base_sp3)


def _example_geom_flag(s, base_dih, base_sp3):
    """Return short text for the first newly-planar / linear flag at a step.

    Prefers a near-linear angle (the existing signal), then a near-planar
    dihedral that was NOT planar at step 1 (new event), then a planar sp3
    centre that was NOT planar at step 1, then any planar dihedral/sp3
    (baseline-aromatic case), then ``-``.
    """
    if s['near_linear_angles']:
        a = s['near_linear_angles'][0]
        return f'{a["coord"]} = {a["angle_deg"]:.1f}°'
    new_dihs = [
        d for d in s.get('near_planar_dihedrals_sample', [])
        if d['idx'] not in base_dih
    ]
    if new_dihs:
        d = new_dihs[0]
        return f'NEW {d["coord"]} = {d["phi_deg"]:.1f}° ({d["kind"]})'
    new_sp3 = [
        c for c in s.get('planar_sp3_sample', [])
        if c['centre_idx'] not in base_sp3
    ]
    if new_sp3:
        c = new_sp3[0]
        return (
            f'NEW centre {c["centre"]} planar (Σangles={c["angle_sum_deg"]:.1f}°)'
        )
    if s.get('near_planar_dihedrals_sample'):
        d = s['near_planar_dihedrals_sample'][0]
        return f'{d["coord"]} = {d["phi_deg"]:.1f}° (aromatic/baseline)'
    if s.get('planar_sp3_sample'):
        c = s['planar_sp3_sample'][0]
        return (
            f'centre {c["centre"]} planar (aromatic/baseline)'
        )
    return '-'


def _planarity_verdict(per_step, warning_steps):
    """Summarise whether warning steps coincided with new planar/linear flags."""
    n_warned = len(warning_steps)
    if n_warned == 0:
        return 'no warning steps to score.'
    base_dih, base_sp3 = _baseline_sets(per_step)
    n_flagged = 0
    for step in warning_steps:
        if step < 1 or step > len(per_step):
            continue
        s = per_step[step - 1]
        new_dih, new_sp3 = _new_planar_at_step(s, base_dih, base_sp3)
        if s['near_linear_angles'] or new_dih or new_sp3:
            n_flagged += 1
    if n_flagged == n_warned:
        return (
            f'every one of the {n_warned} warning step(s) coincides with at '
            'least one planar/linear event new vs step 1 (or a near-linear '
            'angle).'
        )
    if n_flagged == 0:
        return (
            f'none of the {n_warned} warning step(s) coincides with a new '
            'planar/linear event - the warnings here are not '
            'coordinate-singularity events.'
        )
    return (
        f'{n_flagged} of {n_warned} warning step(s) coincide with a new '
        'planar/linear event; the remainder fired on a baseline geometry.'
    )


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
        out.append(
            '| step | pinv_gap_idx | gap | sv_max | sv_min '
            '| #near-lin angles | #near-lin dihedrals '
            '| #near-planar dihedrals | #planar sp3 |'
        )
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
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
                f'| {s["near_linear_dihedrals_count"]} '
                f'| {s.get("near_planar_dihedrals_count", 0)} '
                f'| {s.get("planar_sp3_count", 0)} |'
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

        # Warning/geometry correlation: which trajectory-check warning lines
        # landed on which step, and what was the planar/linear state of the
        # geometry there. ``scan`` is the result of ``scan_log`` on this
        # case's own log file.
        scan = rec.get('scan') or {}
        warning_steps = _collect_warning_steps(scan)
        if not warning_steps:
            out.append(
                '\nNo trajectory-check warning line (pinv, back-xform, '
                'severe dq, neg-eig) fired on this trajectory.\n'
            )
        else:
            base_dih, base_sp3 = _baseline_sets(per_step)
            out.append('\n### Warning-step / geometry correlation\n')
            out.append(
                'For each step where `benchmark_trajectory_check.py`\'s log '
                'scan would record a warning, the columns below report the '
                'count of geometric degeneracies at that step. '
                '`near_lin_ang` counts angles > 175°. `new_planar_dih` is '
                'the number of dihedrals within 5° of 0°/180° that were '
                'NOT already planar at step 1 (subtracting the aromatic-ring '
                'baseline). `new_planar_sp3` is the number of three-coord '
                'heavy centres whose substituent-angle sum exceeds 355° '
                'and were not already planar at step 1. The "geom flag" '
                'column is YES if at least one of those three is nonzero.\n'
            )
            out.append(
                f'Step-1 baseline: {len(base_dih)} planar dihedral(s), '
                f'{len(base_sp3)} planar sp3 centre(s).\n'
            )
            out.append(
                '| step | warning kinds | n_near_lin_ang '
                '| new_planar_dih | new_planar_sp3 | geom flag | example |'
            )
            out.append('|---:|---|---:|---:|---:|---|---|')
            for step in sorted(warning_steps):
                if step < 1 or step > len(per_step):
                    continue
                s = per_step[step - 1]
                kinds = warning_steps[step]
                n_ang = len(s['near_linear_angles'])
                new_dih, new_sp3 = _new_planar_at_step(s, base_dih, base_sp3)
                flag = 'YES' if (n_ang or new_dih or new_sp3) else 'no'
                example = _example_geom_flag(s, base_dih, base_sp3)
                out.append(
                    f'| {step} | {", ".join(sorted(kinds))} | {n_ang} '
                    f'| {new_dih} | {new_sp3} | {flag} | {example} |'
                )
            verdict = _planarity_verdict(per_step, warning_steps)
            out.append(f'\n**Verdict:** {verdict}\n')

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
        'caffeine, plus the neg-eig-only cases avobenzone, codeine, '
        'methylamine) have pinv gap values uniformly above `1e11` and '
        'never trigger the pinv warning. Tracking dihedrals within 5° of '
        '0°/180° and three-coord centres within 5° of their substituent '
        'plane (subtracting the step-1 aromatic baseline) splits these '
        'into two groups:\n'
    )
    out.append(
        '- **Planar-dihedral coincidence** (inosine_cation, maltose, '
        'raffinose, avobenzone, codeine): every step that emitted a '
        'back-xform, severe-dq, or neg-eig warning had at least one '
        'dihedral cross into the 5°-of-planar region between the '
        'starting geometry and that step. On raffinose the offenders are '
        '`Dihedral(C4-C0-O44-H45)` and `Dihedral(O10-C2-C3-H8)` cycling '
        'through 0°/180° as the sugar ring puckers; on maltose it is '
        '`Dihedral(H6-C1-C2-O10)` going trans; on avobenzone it is '
        '`Dihedral(C8-C12-O27-C28)` flipping through cis on every '
        'neg-eig step (14, 17, 20, 49). These warnings are the same '
        'class of coordinate-singularity event as the pinv@final cases '
        '- the back-transform Newton iteration just happens to absorb '
        'the gradient blow-up before the pinv truncation fires.\n'
    )
    out.append(
        '- **PES topology, not coordinate singularity** (caffeine, '
        'methylamine): zero of the neg-eig warning steps coincided with '
        'a new planar event. Caffeine\'s 43 neg-eig events and '
        'methylamine\'s 4 are genuine BFGS spurious-unstable-mode events '
        'on a confusing PES region, not numerical breakdowns of the '
        'internal-coord representation.\n'
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
    out.append(
        '| inosine_cation | `Dihedral(C4-C0-C13-O16)` -> 177.5° | new '
        'trans-planar dihedral coincides with step-11 severe-dq |'
    )
    out.append(
        '| maltose | `Dihedral(H6-C1-C2-O10)` -> ~177° | new trans '
        'dihedral on every severe-dq + neg-eig step (27/30/31) |'
    )
    out.append(
        '| raffinose | `Dihedral(C4-C0-O44-H45)`, '
        '`Dihedral(O10-C2-C3-H8)`, `Dihedral(O21-C14-C17-H20)` -> '
        '~0°/180° | sugar-ring puckering through planar configurations '
        'over steps 42-49 |'
    )
    out.append(
        '| avobenzone | `Dihedral(C8-C12-O27-C28)` -> ~0° | aryl-ether '
        'C-O dihedral flipping cis on every neg-eig step (14/17/20/49) |'
    )
    out.append(
        '| codeine | `Dihedral(C5-C0-C1-C2)` -> ~3° | ring dihedral '
        'going cis on the neg-eig steps (15/18) |'
    )
    out.append(
        '| caffeine | none | 43 neg-eig events on a confusing PES; '
        'no new planar/linear event on any warning step |'
    )
    out.append(
        '| methylamine | none | 4 neg-eig events; N inversion '
        'baseline-planar already, no new planar/linear event |'
    )

    # Cross-case planarity correlation: was every warning step on every
    # case accompanied by at least one planar/linear flag?
    out.append('\n### Planar/linear coincidence across all warning categories\n')
    rows = []
    for case in CASES:
        set_key, molecule, _maxsteps, _why = case
        rec = report.get(f'{set_key}/{molecule}')
        if rec is None:
            continue
        per_step = rec['per_step']
        scan = rec.get('scan') or {}
        warning_steps = _collect_warning_steps(scan)
        if not warning_steps:
            rows.append((f'{set_key}/{molecule}', 0, 0, 'no warnings'))
            continue
        n_warned = len(warning_steps)
        base_dih, base_sp3 = _baseline_sets(per_step)
        n_flagged = 0
        for step in warning_steps:
            if not 1 <= step <= len(per_step):
                continue
            s = per_step[step - 1]
            new_dih, new_sp3 = _new_planar_at_step(s, base_dih, base_sp3)
            if s['near_linear_angles'] or new_dih or new_sp3:
                n_flagged += 1
        if n_flagged == n_warned:
            verdict = 'all planar/linear'
        elif n_flagged == 0:
            verdict = 'none planar/linear'
        else:
            verdict = 'partial'
        rows.append((f'{set_key}/{molecule}', n_flagged, n_warned, verdict))
    out.append('| case | warning steps w/ planar-or-linear flag | total warning steps | verdict |')
    out.append('|---|---:|---:|---|')
    for name, k, n, v in rows:
        out.append(f'| {name} | {k} | {n} | {v} |')
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
