#!/usr/bin/env python3
"""Per-step internal-coordinate health check across the benchmark "bad" cases.

This is a generalisation of ``estradiol_internals_diag.py``. For every
molecule that lit up a warning in ``benchmark_trajectory_check.py`` (plus a
few controls), it re-runs ``Berny + MopacSolver`` and at every optimizer
step records:

Geometry health (the "planar / linear" hypothesis):

- For every ``Angle`` in ``InternalCoords``, the value in degrees. The list
  of angles ``>= 170 deg`` is reported.
- For every 3-coordinate atom (3 ``Bond`` neighbours in the pyberny internal
  set), the sum of its three neighbour-pair angles. Sum ``-> 360 deg`` means
  the centre has flattened to sp2-like geometry.
- For every 4-coordinate atom, the minimum out-of-plane angle (the
  pyramidalisation), the smallest angle between bond k->m and the plane
  through bonds k->i, k->j, k->l (over the 4 choices of m). Going to 0 means
  inversion through planar.
- Bond lengths > 1.5x the sum of covalent radii (impending dissociation).

B-matrix / pinv health:

- Singular value spectrum of ``B B^T``.
- Index of the first consecutive-ratio gap > 10^3 (what ``Math.pinv`` would
  truncate at) and the gap magnitude. Compared against the "natural" gap at
  3N - 6 / 3N - 5 chemical DOFs.
- Top contributors to the *first truncated* left-singular vector.

Step / back-transform health (correlated with warning lines):

- Warning lines emitted that step: ``Pseudoinverse gap of only:``,
  ``Transformation did not converge in N iterations``, RMS(dq) of the
  back-transform, ``Number of negative eigenvalues``.

Outputs (under ``--out``, default ``benchmark_internals_diag/``):

- ``<molecule>.json`` per molecule (per-step diagnostics).
- ``log_<molecule>.txt`` per molecule (raw INFO log from the berny logger).
- ``triggers.md`` (co-occurrence + mechanism classification roll-up table).

Idempotent: per-molecule JSON, log and plot are skipped if they already
exist. Use ``--force`` to rerun, or ``--molecules ...`` to scope.
"""

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.coords import Angle, Bond, Dihedral
from berny.species_data import get_property
from berny.solvers import MopacSolver


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'tests' / 'data'
DEFAULT_OUT = REPO_ROOT / 'experiments' / 'microvariations' / 'benchmark_internals_diag'

# (set_dir, molecule, class label for the roll-up)
CASES = [
    ('baker_shajan_2023', 'allene', 'pinv@final'),
    ('baker_shajan_2023', 'disilyl_ether', 'pinv@final'),
    ('birkholz_schlegel', 'estradiol', 'pinv@final (cross-validate)'),
    ('birkholz_schlegel', 'raffinose', 'heavy back-xform + saddles'),
    ('baker_shajan_2023', 'caffeine', 'sustained neg-eig'),
    ('birkholz_schlegel', 'maltose', 'severe dq (converges right)'),
    ('birkholz_schlegel', 'inosine_cation', 'severe dq (converges right)'),
    ('birkholz_schlegel', 'ochratoxin_a', 'hit maxsteps'),
    ('birkholz_schlegel', 'bisphenol_a', 'hit maxsteps'),
    ('birkholz_schlegel', 'artemisinin', 'control (clean)'),
    ('birkholz_schlegel', 'vitamin_c', 'control (clean)'),
    ('baker_shajan_2023', 'benzene', 'control (clean)'),
]

NEAR_LINEAR_DEG = 170.0  # report-list threshold for angles
ANGLE_CRIT_DEG = 175.0  # trigger threshold for "linear-angle-dihedral"
PLANAR_SUM_DEG = 355.0  # trigger threshold for 3-coord planarisation
PYRAMID_CRIT_DEG = 5.0  # trigger threshold for sp3 inversion
BOND_STRETCH_FACTOR = 1.5  # bond > 1.5x sum of covalent radii => "bond-stretch"

# Warning-line regexes - mirror benchmark_trajectory_check.py
PINV_RE = re.compile(r'^(\d+)\s+Pseudoinverse gap of only:\s+([0-9.eE+-]+)\s*$')
BACKXFORM_RE = re.compile(
    r'^(\d+)\s+Transformation did not converge in (\d+) iterations\s*$'
)
NEGEIG_RE = re.compile(
    r'^(\d+)\s+\* Number of negative eigenvalues:\s+([0-9]+)\s*$'
)
BACKXFORM_DQ_RE = re.compile(
    r'^(\d+)\s+\* RMS\(dcart\):\s+([0-9.eE+-]+),\s+RMS\(dq\):\s+([0-9.eE+-]+)\s*$'
)
ALL_CRITERIA_RE = re.compile(r'^(\d+)\s+\* All criteria matched\s*$')
SEVERE_DQ_THRESHOLD = 0.05


@contextlib.contextmanager
def silence_fd(fd):
    """Redirect OS fd ``fd`` to /dev/null - needed because MOPAC writes a banner per call."""
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


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


def neighbour_table(coords_obj, n_atoms):
    """Map atom index -> sorted list of bonded neighbour indices (from pyberny's internal bonds)."""
    nb = defaultdict(set)
    for c in coords_obj._coords:
        if isinstance(c, Bond):
            nb[c.i].add(c.j)
            nb[c.j].add(c.i)
    return {a: sorted(nb[a]) for a in range(n_atoms)}


def angle_deg(coords, i, j, k):
    """Bond angle i-j-k in degrees."""
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pyramidalisation_deg(coords, centre, neighbours):
    """Out-of-plane angle for a 4-coord centre.

    For each choice of "apex" neighbour ``m``, fit a plane through the
    other three neighbour positions (relative to the centre), and return
    the angle (in degrees) between bond centre->m and that plane. The
    minimum over the 4 choices is the pyramidalisation: 0 -> planar
    (sp3 has inverted), ~35 -> tetrahedral.
    """
    assert len(neighbours) == 4
    c = coords[centre]
    vs = np.array([coords[n] - c for n in neighbours])  # 4x3
    vs = vs / np.linalg.norm(vs, axis=1, keepdims=True)
    angles = []
    for m_idx in range(4):
        others = [i for i in range(4) if i != m_idx]
        plane = vs[others]  # 3x3
        # plane normal = right-singular vector with smallest singular value
        _, _, vh = np.linalg.svd(plane - plane.mean(axis=0), full_matrices=False)
        normal = vh[-1]
        cos_to_plane = abs(np.dot(vs[m_idx], normal))
        # angle from plane = 90 - angle from normal
        ang_from_normal = np.degrees(np.arccos(np.clip(cos_to_plane, 0.0, 1.0)))
        angles.append(90.0 - ang_from_normal)
    return float(min(angles))


def geometric_flags(coords_obj, geom):
    """Compute the "going planar / linear" diagnostics."""
    cart = np.asarray(geom.coords)
    species = list(geom.species)
    n_atoms = len(species)
    nb = neighbour_table(coords_obj, n_atoms)

    # All angles in InternalCoords - list those >= NEAR_LINEAR_DEG
    angles_report = []
    max_angle = 0.0
    for c in coords_obj._coords:
        if isinstance(c, Angle):
            phi = angle_deg(cart, c.i, c.j, c.k)
            if phi > max_angle:
                max_angle = phi
            if phi >= NEAR_LINEAR_DEG:
                angles_report.append([c.j, phi, f'{species[c.i]}{c.i}-'
                                                f'{species[c.j]}{c.j}-'
                                                f'{species[c.k]}{c.k}'])

    # 3-coord atoms: angle sum
    planar_centres = []
    max_3sum = 0.0
    for atom, ns in nb.items():
        if len(ns) == 3:
            a = angle_deg(cart, ns[0], atom, ns[1])
            b = angle_deg(cart, ns[0], atom, ns[2])
            c = angle_deg(cart, ns[1], atom, ns[2])
            s = a + b + c
            if s > max_3sum:
                max_3sum = s
            if s >= PLANAR_SUM_DEG:
                planar_centres.append([atom, s, species[atom]])

    # 4-coord atoms: minimum pyramidalisation
    sp3_inverted = []
    min_pyr = None
    for atom, ns in nb.items():
        if len(ns) == 4:
            pyr = pyramidalisation_deg(cart, atom, ns)
            if min_pyr is None or pyr < min_pyr:
                min_pyr = pyr
            if pyr < PYRAMID_CRIT_DEG:
                sp3_inverted.append([atom, pyr, species[atom]])

    # Over-stretched bonds (vs sum of covalent radii)
    stretched = []
    max_stretch = 0.0
    for c in coords_obj._coords:
        if isinstance(c, Bond):
            r = float(np.linalg.norm(cart[c.i] - cart[c.j]))
            r_ref = (
                get_property(species[c.i], 'covalent_radius')
                + get_property(species[c.j], 'covalent_radius')
            )
            ratio = r / r_ref
            if ratio > max_stretch:
                max_stretch = ratio
            if ratio >= BOND_STRETCH_FACTOR:
                stretched.append([c.i, c.j, r, ratio])

    return {
        'max_angle_deg': max_angle,
        'near_linear_angles': angles_report,
        'max_3coord_anglesum_deg': max_3sum,
        'planar_3coord_centres': planar_centres,
        'min_pyramidalisation_deg': min_pyr,
        'sp3_inverted_centres': sp3_inverted,
        'max_bond_stretch_ratio': max_stretch,
        'stretched_bonds': stretched,
    }


def bmatrix_health(coords_obj, geom):
    B = coords_obj.B_matrix(geom)
    BBT = B @ B.T
    eigs = np.linalg.eigvalsh(BBT)
    eigs = np.sort(eigs)[::-1]
    pos = np.maximum(eigs, 1e-30)
    n_gap, gap = first_big_gap(pos, thre=1e3)

    top_contrib = []
    if n_gap is not None and n_gap + 1 < len(pos):
        ev, vecs = np.linalg.eigh(BBT)
        target_eig = eigs[n_gap + 1]
        idx = int(np.argmin(np.abs(ev - target_eig)))
        truncated_vec = vecs[:, idx]
        contributions = truncated_vec**2
        order = np.argsort(-contributions)
        top_contrib = [
            [int(i), float(contributions[i]), classify_coord(coords_obj._coords[i])]
            for i in order[:5]
        ]

    return {
        'n_internals': int(B.shape[0]),
        'n_cart': int(B.shape[1]),
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
    }


def scan_log(log_path):
    """Per-step warning record from a berny log file."""
    pinv = defaultdict(list)
    backx = defaultdict(list)
    negeig = defaultdict(list)
    dq = {}
    final = None
    with open(log_path) as fh:
        for line in fh:
            m = PINV_RE.match(line)
            if m:
                pinv[int(m.group(1))].append(float(m.group(2)))
                continue
            m = BACKXFORM_RE.match(line)
            if m:
                backx[int(m.group(1))].append(int(m.group(2)))
                continue
            m = NEGEIG_RE.match(line)
            if m:
                if int(m.group(2)) > 0:
                    negeig[int(m.group(1))].append(int(m.group(2)))
                continue
            m = BACKXFORM_DQ_RE.match(line)
            if m:
                dq[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                continue
            m = ALL_CRITERIA_RE.match(line)
            if m:
                final = int(m.group(1))
    return {
        'pinv': dict(pinv),
        'backxform': dict(backx),
        'negeig': dict(negeig),
        'dq': dq,
        'convergence_step': final,
    }


def run_molecule(data_dir, name, charge, mult, maxsteps, log_path):
    """Run berny+mopac and capture per-step diagnostics."""
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
    energy_path = []
    error = None
    t0 = time.perf_counter()
    try:
        with silence_fd(1), silence_fd(2):
            geom = geomlib.readfile(str(data_dir / f'{name}.xyz'))
            berny = Berny(geom, maxsteps=maxsteps)
            coords_obj = berny._state.coords
            solver = MopacSolver(charge=charge, mult=mult)
            next(solver)
            for current in berny:
                geom_d = geometric_flags(coords_obj, current)
                bmat_d = bmatrix_health(coords_obj, current)
                energy, gradients = solver.send((list(current), current.lattice))
                energy_path.append(float(energy))
                per_step.append(
                    {
                        'step': len(per_step) + 1,
                        'energy_ha': float(energy),
                        **geom_d,
                        **bmat_d,
                    }
                )
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
        'wall_s': time.perf_counter() - t0,
        'error': error,
        'per_step': per_step,
    }


def classify_mechanism(step_diag):
    """Assign a mechanism label for a step, given its geometric flags."""
    if step_diag['near_linear_angles']:
        # Any near-180 angle.
        max_a = max(a[1] for a in step_diag['near_linear_angles'])
        if max_a >= ANGLE_CRIT_DEG:
            return 'linear-angle-dihedral'
    if step_diag['planar_3coord_centres']:
        return 'planar-center'
    if step_diag['sp3_inverted_centres']:
        return 'sp3-inversion'
    if step_diag['stretched_bonds']:
        return 'bond-stretch'
    return 'unattributed'


def step_warning_label(scan, step):
    flags = []
    if step in scan['pinv']:
        flags.append('pinv')
    if step in scan['backxform']:
        flags.append('back-xform')
    if step in scan['negeig']:
        flags.append('neg-eig')
    dq = scan['dq'].get(step)
    if dq is not None and dq[1] >= SEVERE_DQ_THRESHOLD:
        flags.append('severe-dq')
    return flags


def cooccurrence(per_step, scan):
    """Return list of {step, warnings, mechanism, geom_flag_bits}."""
    rows = []
    for d in per_step:
        s = d['step']
        warns = step_warning_label(scan, s)
        bits = {
            'angle_ge_175': bool(d['near_linear_angles'])
            and max(a[1] for a in d['near_linear_angles']) >= ANGLE_CRIT_DEG,
            'planar_3coord': bool(d['planar_3coord_centres']),
            'sp3_inverted': bool(d['sp3_inverted_centres']),
            'pinv_truncation_pre_chemical': (
                # the *real* gap should be at 3N-6 chemical DOFs;
                # any gap below that is the "spurious" one
                d['pinv_gap_index'] is not None
                and d['pinv_gap_index'] < (d['n_cart'] - 6)
            ),
        }
        rows.append(
            {
                'step': s,
                'warnings': warns,
                'mechanism': classify_mechanism(d),
                **bits,
            }
        )
    return rows


def cross_validate_estradiol(out_dir):
    """Compare a representative estradiol step against the bespoke ``internals_diag/per_step.json``.

    The bespoke run perturbs starting coords; the published-geometry run
    here does not. We can't compare per-step diagnostics directly, but we
    can confirm the new analyse functions give the same ``pinv_gap_index``
    / ``top_5_truncated_contribs`` pattern when given the exact estradiol
    geometries from the bespoke run's final step.

    Specifically: re-run ``estradiol`` from its published .xyz; if at any
    step it produces the "spurious gap at index 0" signature with H-C-O
    near-linear that the bespoke run identified at sigma>=0.01, log it.
    The published .xyz is not expected to reproduce the catastrophe (that
    requires perturbation), but the per-step pinv-gap-index trajectory
    should start at the natural gap (n_cart - 6) and only drop on
    near-linear-angle events.
    """
    new = (out_dir / 'estradiol.json')
    if not new.exists():
        return None
    data = json.loads(new.read_text())
    natural_gap = data['per_step'][0]['n_cart'] - 6
    drops = [
        d['step']
        for d in data['per_step']
        if d['pinv_gap_index'] is not None and d['pinv_gap_index'] < natural_gap
    ]
    return {
        'natural_gap_expected': natural_gap,
        'per_step_gap_idx_at_first_step': data['per_step'][0]['pinv_gap_index'],
        'per_step_gap_idx_at_last_step': data['per_step'][-1]['pinv_gap_index'],
        'spurious_gap_steps': drops,
    }


def render_triggers_md(records, out_path):
    out = [
        '# Internal-coordinate triggers along benchmark trajectories\n',
        'Per-step diagnostics for every "bad case" molecule from '
        '`benchmark_diag/warnings.json`. Each row in the per-molecule '
        'mechanism table reports the warning class fired that step and '
        'the strongest geometric flag, classified per the rules at the '
        'bottom.\n',
    ]

    # Roll-up: extends benchmark_diag/warnings.md with mechanism column.
    out.append('\n## Roll-up: warning steps and their geometric trigger\n')
    out.append(
        '| set | molecule | class | converged | steps | warning step(s) '
        '| first geometric flag at/before warning | mechanism |'
    )
    out.append('|---|---|---|---|---:|---|---|---|')
    for rec in records:
        cls = rec['class_label']
        run = rec['run']
        scan = rec['scan']
        per_step = run['per_step']
        if run.get('error'):
            n_done = len(per_step)
            # Still summarise warnings observed before the crash.
            warning_steps = sorted(
                set(scan['pinv'])
                | set(scan['backxform'])
                | set(scan['negeig'])
                | {s for s, v in scan['dq'].items() if v[1] >= SEVERE_DQ_THRESHOLD}
            )
            if warning_steps:
                ws = (
                    ','.join(map(str, warning_steps[:5]))
                    + ('...' if len(warning_steps) > 5 else '')
                )
            else:
                ws = '-'
            out.append(
                f'| {rec["set_dir"]} | {rec["molecule"]} | {cls} | '
                f'ERR ({run["error"]}) | {n_done} | {ws} | - | crashed |'
            )
            continue
        warning_steps = sorted(
            set(scan['pinv'])
            | set(scan['backxform'])
            | set(scan['negeig'])
            | {s for s, v in scan['dq'].items() if v[1] >= SEVERE_DQ_THRESHOLD}
        )
        if not warning_steps:
            out.append(
                f'| {rec["set_dir"]} | {rec["molecule"]} | {cls} | '
                f'{"yes" if run["converged"] else "no"} | {run["steps"]} | '
                f'- | - | clean |'
            )
            continue

        first_w = warning_steps[0]
        # Mechanism rollup: most-specific mechanism observed across ANY
        # warning step (priority order below). Using "first step only"
        # would mis-classify trajectories where the linear-angle event
        # arrives one step after an earlier aromatic-planar false positive.
        priority = [
            'linear-angle-dihedral',
            'sp3-inversion',
            'bond-stretch',
            'planar-center',
            'unattributed',
        ]
        mechs_observed = set()
        for w in warning_steps:
            if w - 1 < len(per_step):
                mechs_observed.add(classify_mechanism(per_step[w - 1]))
        mech = next((m for m in priority if m in mechs_observed), 'unattributed')

        # Geometric flag string from the FIRST warning step (for the
        # "first geometric flag at/before warning" column).
        if first_w - 1 < len(per_step):
            d = per_step[first_w - 1]
            flag_str = []
            if d['near_linear_angles']:
                a = max(d['near_linear_angles'], key=lambda x: x[1])
                flag_str.append(f'angle {a[2]}={a[1]:.1f} deg')
            if d['planar_3coord_centres']:
                a = max(d['planar_3coord_centres'], key=lambda x: x[1])
                flag_str.append(f'3-coord {a[2]}{a[0]} sum={a[1]:.1f} deg')
            if d['sp3_inverted_centres']:
                a = min(d['sp3_inverted_centres'], key=lambda x: x[1])
                flag_str.append(f'sp3 {a[2]}{a[0]} pyr={a[1]:.2f} deg')
            if d['stretched_bonds']:
                a = max(d['stretched_bonds'], key=lambda x: x[3])
                flag_str.append(f'bond {a[0]}-{a[1]} ratio={a[3]:.2f}')
            flag_summary = '; '.join(flag_str) if flag_str else '-'
        else:
            flag_summary = '-'

        warn_steps_short = (
            ','.join(map(str, warning_steps[:5]))
            + ('...' if len(warning_steps) > 5 else '')
        )
        out.append(
            f'| {rec["set_dir"]} | {rec["molecule"]} | {cls} | '
            f'{"yes" if run["converged"] else "no"} | {run["steps"]} | '
            f'{warn_steps_short} | {flag_summary} | {mech} |'
        )

    # Per-molecule co-occurrence summary
    out.append('\n## Per-molecule co-occurrence of warnings and geometric flags\n')
    out.append(
        'Each row is one optimizer step. `mechanism` is the classification '
        'applied at that step. Only steps that fired *any* warning are '
        'tabulated. Steps with no warnings and no geometric flags are not '
        'shown.\n'
    )
    for rec in records:
        run = rec['run']
        scan = rec['scan']
        per_step = run['per_step']
        if not per_step:
            continue
        co = cooccurrence(per_step, scan)
        flagged = [r for r in co if r['warnings']]
        if not flagged:
            continue
        title = f'{rec["set_dir"]}/{rec["molecule"]}'
        if run.get('error'):
            title += f' (crashed at step {len(per_step)})'
        out.append(f'\n### {title}\n')
        out.append('| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |')
        out.append('|---:|---|---|---|---|---|---|')
        for r in flagged:
            out.append(
                f'| {r["step"]} | {",".join(r["warnings"])} | '
                f'{"Y" if r["angle_ge_175"] else "."} | '
                f'{"Y" if r["planar_3coord"] else "."} | '
                f'{"Y" if r["sp3_inverted"] else "."} | '
                f'{"Y" if r["pinv_truncation_pre_chemical"] else "."} | '
                f'{r["mechanism"]} |'
            )

    # Mechanism counts and Jaccard-style co-occurrence
    out.append('\n## Mechanism summary across molecules\n')
    mech_counts = defaultdict(int)
    warning_total = 0
    cooc = defaultdict(int)  # mechanism -> warning-step count
    for rec in records:
        run = rec['run']
        scan = rec['scan']
        per_step = run['per_step']
        if not per_step:
            continue
        co = cooccurrence(per_step, scan)
        for r in co:
            if r['warnings']:
                warning_total += 1
                mech = r['mechanism']
                cooc[mech] += 1
                mech_counts[mech] += 1
    if warning_total:
        out.append('| mechanism | warning steps with this mechanism | share |')
        out.append('|---|---:|---:|')
        for mech in [
            'linear-angle-dihedral',
            'planar-center',
            'sp3-inversion',
            'bond-stretch',
            'unattributed',
        ]:
            n = cooc.get(mech, 0)
            out.append(
                f'| {mech} | {n} / {warning_total} | {n / warning_total:.0%} |'
            )

    out.append('\n## Classification rules\n')
    out.append(
        f'- `linear-angle-dihedral`: any `Angle` in `InternalCoords` '
        f'>= {ANGLE_CRIT_DEG} deg. Reproduces the estradiol H-C-O '
        f'mechanism: 1/||a1|| in `Dihedral.eval` diverges as the '
        f'containing angle approaches 180 deg.\n'
        f'- `planar-center`: any 3-bond-neighbour atom whose three '
        f'neighbour-pair angles sum to >= {PLANAR_SUM_DEG} deg (sp2-like '
        f'flattening). **Caveat**: this fires permanently for every '
        f'aromatic-ring carbon (benzene, all six C atoms always sum to '
        f'~360 deg) and is therefore a *false positive* in the mechanism '
        f'column for any molecule whose backbone contains an aromatic '
        f'ring (caffeine, ochratoxin_a, bisphenol_a, inosine_cation, '
        f'estradiol). Use the *transition* in `max_3coord_anglesum_deg` '
        f'across the warning step (not its absolute value) when '
        f'interpreting those rows; a step where a previously '
        f'non-planar centre suddenly flattens is the genuine signal.\n'
        f'- `sp3-inversion`: any 4-bond-neighbour atom whose minimum '
        f'out-of-plane angle (defined as the smallest of the 4 '
        f'centre->m to plane-of-the-other-three angles) drops below '
        f'{PYRAMID_CRIT_DEG} deg (sp3 inverted through planar).\n'
        f'- `bond-stretch`: any `Bond` whose length exceeds '
        f'{BOND_STRETCH_FACTOR} x the sum of its atoms\' covalent radii.\n'
        f'- `unattributed`: no geometric flag triggered at this step. '
        f'These are the cases that *do not* match the estradiol-style '
        f'singular-coordinate story and need a separate explanation '
        f'(candidate: BFGS Hessian flips / RFO saddle-mode descent on a '
        f'flat torsional manifold; not investigated here).\n'
    )

    out.append('\n## Estradiol cross-validation\n')
    cv = render_estradiol_cv()
    out.append(cv)
    out_path.write_text('\n'.join(out) + '\n')


def render_estradiol_cv():
    bespoke = REPO_ROOT / 'experiments' / 'microvariations' / 'internals_diag' / 'per_step.json'
    if not bespoke.exists():
        return 'Bespoke `internals_diag/per_step.json` not found - cannot cross-validate.\n'
    data = json.loads(bespoke.read_text())
    lines = [
        'The bespoke estradiol driver perturbs the starting geometry to '
        'reach the basin where the pinv-at-final catastrophe fires. The '
        'new driver runs the *published* estradiol.xyz unperturbed, so '
        'the new run is expected to behave like a non-pathological '
        'trajectory: the pinv gap should stay at the "natural" '
        '3N - 6 chemical-DOF position and never drop to index 0. '
        'Confirmation of the bespoke run\'s findings on the perturbed '
        'seeds (basin 4 and basin 3) is reproduced below:',
        '',
        '| case | step | pinv gap idx | gap | top contributor |',
        '|---|---:|---:|---:|---|',
    ]
    for label, rec in data.items():
        for step_idx in (-2, -1):
            if abs(step_idx) > len(rec['per_step']):
                continue
            d = rec['per_step'][step_idx]
            top = d['top_5_truncated_contribs'][0] if d['top_5_truncated_contribs'] else None
            lines.append(
                f'| {label} | {len(rec["per_step"]) + step_idx + 1} | '
                f'{d["pinv_gap_index"]} | '
                f'{d["pinv_gap_value"]:.2e} | '
                f'{top[2] if top else "-"} |'
            )
    lines.append('')
    lines.append(
        'Re-running the new driver on `estradiol` (published .xyz) is '
        'a sanity check: it should converge in 11 steps with the '
        'pinv-at-final signature flagged at the final step, but with '
        'the *same* H37-C36-O40 H-C-O angle as the trigger '
        '(see the per-molecule co-occurrence table above).'
    )
    return '\n'.join(lines) + '\n'


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        '--molecules', nargs='+', default=None,
        help='restrict to these molecule names (any set)',
    )
    ap.add_argument('--maxsteps', type=int, default=110)
    ap.add_argument(
        '--force', action='store_true',
        help='rerun even if per-molecule JSON already exists',
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')
    args.out.mkdir(parents=True, exist_ok=True)

    refs = {}
    for set_dir in {c[0] for c in CASES}:
        refs[set_dir] = json.loads(
            (DATA_ROOT / set_dir / 'reference.json').read_text()
        )

    records = []
    t_start = time.perf_counter()
    for i, (set_dir, mol, cls) in enumerate(CASES, 1):
        if args.molecules and mol not in args.molecules:
            continue
        data_dir = DATA_ROOT / set_dir
        ref = refs[set_dir][mol]
        log_path = args.out / f'log_{mol}.txt'
        json_path = args.out / f'{mol}.json'

        if json_path.exists() and not args.force:
            run = json.loads(json_path.read_text())
            cached = True
        else:
            run = run_molecule(
                data_dir, mol,
                ref.get('charge', 0), ref.get('mult', 1),
                args.maxsteps, log_path,
            )
            # Persist scan inside the JSON so the log file is only
            # needed for the initial run; the rollup can be regenerated
            # from the JSONs alone (logs are gitignored).
            run['scan'] = scan_log(log_path)
            # JSON keys must be strings - convert int step keys.
            run['scan'] = {
                k: ({str(s): v for s, v in d.items()} if isinstance(d, dict) else d)
                for k, d in run['scan'].items()
            }
            json_path.write_text(json.dumps(run, indent=2))
            cached = False

        if 'scan' in run:
            raw = run['scan']
            scan = {
                'pinv': {int(k): v for k, v in raw.get('pinv', {}).items()},
                'backxform': {int(k): v for k, v in raw.get('backxform', {}).items()},
                'negeig': {int(k): v for k, v in raw.get('negeig', {}).items()},
                'dq': {int(k): tuple(v) for k, v in raw.get('dq', {}).items()},
                'convergence_step': raw.get('convergence_step'),
            }
        elif log_path.exists():
            scan = scan_log(log_path)
        else:
            scan = {
                'pinv': {}, 'backxform': {}, 'negeig': {}, 'dq': {},
                'convergence_step': None,
            }

        records.append(
            {
                'set_dir': set_dir,
                'molecule': mol,
                'class_label': cls,
                'run': run,
                'scan': scan,
            }
        )
        elapsed = time.perf_counter() - t_start
        n_warn = (
            len(scan['pinv']) + len(scan['backxform'])
            + len(scan['negeig']) + sum(
                1 for v in scan['dq'].values() if v[1] >= SEVERE_DQ_THRESHOLD
            )
        )
        tag = 'CACHED' if cached else f'{run["wall_s"]:.0f}s'
        print(
            f'[{i}/{len(CASES)}] {set_dir}/{mol}: {tag} '
            f'steps={run["steps"]} conv={run["converged"]} '
            f'warnings={n_warn} (total {elapsed:.0f}s)',
            flush=True,
        )

    aggregate = {
        'records': [
            {
                'set_dir': r['set_dir'],
                'molecule': r['molecule'],
                'class_label': r['class_label'],
                'converged': r['run']['converged'],
                'steps': r['run']['steps'],
                'wall_s': r['run']['wall_s'],
                'error': r['run']['error'],
                'scan_summary': {
                    'pinv_steps': sorted(r['scan']['pinv']),
                    'backxform_steps': sorted(r['scan']['backxform']),
                    'negeig_steps': sorted(r['scan']['negeig']),
                    'severe_dq_steps': sorted(
                        s for s, v in r['scan']['dq'].items()
                        if v[1] >= SEVERE_DQ_THRESHOLD
                    ),
                    'convergence_step': r['scan']['convergence_step'],
                },
                'cooccurrence': cooccurrence(r['run']['per_step'], r['scan']),
            }
            for r in records
        ],
    }
    (args.out / 'per_step.json').write_text(json.dumps(aggregate, indent=2))
    render_triggers_md(records, args.out / 'triggers.md')
    print(f'\nWrote {args.out / "per_step.json"}')
    print(f'Wrote {args.out / "triggers.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
