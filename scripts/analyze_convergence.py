#!/usr/bin/env python3
"""Analyze MOPAC energy-convergence behavior across benchmark shards."""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import BENCHMARKS  # noqa: E402

from berny import geomlib  # noqa: E402

HARTREE_TO_KCAL = 627.503
ENERGY_TOL_KCAL = 1e-3

TRUST_RE = re.compile(r'^(\d+)\s+\* Trust radius:\s+([0-9.eE+-]+)\s*$')
REBUILD_RE = re.compile(
    r'^(\d+)\s+Linear-bend topology changed; rebuilding internal coordinates\s*$'
)
NOFIT_RE = re.compile(r'^(\d+)\s+\* No fit succeeded, .*point\s*$')
SPHERE_RE = re.compile(r'^(\d+)\s+Minimization on sphere was performed:\s*$')


def load_rows(results_dir, benchmark, solver):
    rows = {}
    for path in sorted(results_dir.glob(f'{solver}-{benchmark}-*.json')):
        data = json.loads(path.read_text())
        for row in data.get('rows', []):
            energies = row.get('energies') or []
            if not energies:
                continue
            rows[row['name']] = row
    return rows


def settled_step(energies, tol_kcal=ENERGY_TOL_KCAL):
    arr = np.array(energies, dtype=float)
    delta = (arr - arr.min()) * HARTREE_TO_KCAL
    above = np.flatnonzero(delta > tol_kcal)
    return int(above[-1] + 1) if len(above) else 1


def increase_steps(energies):
    arr = np.array(energies, dtype=float)
    diff = np.diff(arr)
    return [int(i + 2) for i, d in enumerate(diff) if d > 0]


def parse_log(log_path):
    trust = []
    rebuild_steps = set()
    nofit_steps = set()
    sphere_steps = set()
    with open(log_path) as fh:
        for line in fh:
            m = TRUST_RE.match(line)
            if m:
                trust.append((int(m.group(1)), float(m.group(2))))
                continue
            m = REBUILD_RE.match(line)
            if m:
                rebuild_steps.add(int(m.group(1)))
                continue
            m = NOFIT_RE.match(line)
            if m:
                nofit_steps.add(int(m.group(1)))
                continue
            m = SPHERE_RE.match(line)
            if m:
                sphere_steps.add(int(m.group(1)))
    trust.sort()
    trust_reductions = set()
    for (s0, t0), (s1, t1) in zip(trust, trust[1:]):
        if s1 > s0 and t1 < t0:
            trust_reductions.add(s1)
    return {
        'trust_reduction_steps': trust_reductions,
        'rebuild_steps': rebuild_steps,
        'nofit_steps': nofit_steps,
        'sphere_steps': sphere_steps,
    }


def load_log_events(logs_dir, benchmark, name):
    if logs_dir is None:
        return None
    cands = [
        logs_dir / BENCHMARKS[benchmark].name / f'{name}.log',
        logs_dir / benchmark / f'{name}.log',
        logs_dir / f'{name}.log',
    ]
    for path in cands:
        if path.exists():
            return parse_log(path)
    return None


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    if np.std(xs) == 0 or np.std(ys) == 0:
        return None
    try:
        return float(np.corrcoef(xs, ys)[0, 1])
    except Exception:  # noqa: BLE001
        return None


def structural_metrics(data_dir, name):
    geom = geomlib.readfile(str(data_dir / f'{name}.xyz'))
    n_atoms = len(geom)
    dof = max(0, 3 * n_atoms - 6)
    bond = geom.bondmatrix()
    degree = bond.sum(axis=1).astype(int)
    heavy = [i for i, s in enumerate(geom.species) if s != 'H']
    rot = 0
    for i in heavy:
        for j in heavy:
            if j <= i or not bond[i, j]:
                continue
            if degree[i] > 1 and degree[j] > 1:
                rot += 1
    electroneg = {'N', 'O', 'S'}
    acceptors = [i for i, s in enumerate(geom.species) if s in electroneg]
    donors = []
    for i, s in enumerate(geom.species):
        if s not in electroneg:
            continue
        if any(geom.species[j] == 'H' for j in np.flatnonzero(bond[i])):
            donors.append(i)
    dist = geom.dist()
    hbond_network = sum(
        1 for d in donors for a in acceptors if d != a and dist[d, a] <= 3.5
    )
    return {
        'atoms': n_atoms,
        'dof': dof,
        'rotatable_bonds': rot,
        'hbond_network': hbond_network,
    }


def near_linear_angles_count(data_dir, name):
    geom = geomlib.readfile(str(data_dir / f'{name}.xyz'))
    bond = geom.bondmatrix()
    count = 0
    for j in range(len(geom)):
        nbr = list(np.flatnonzero(bond[j]))
        for ia, i in enumerate(nbr):
            for k in nbr[ia + 1 :]:
                v1 = geom.coords[i] - geom.coords[j]
                v2 = geom.coords[k] - geom.coords[j]
                c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                deg = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
                if deg >= 175.0:
                    count += 1
    return count


def analyze(rows, benchmark, reference, data_dir, logs_dir):
    per_molecule = []
    for name, row in sorted(rows.items()):
        energies = row['energies']
        arr = np.array(energies, dtype=float)
        inc_steps = increase_steps(energies)
        event = load_log_events(logs_dir, benchmark, name)
        causes = {
            'trust_reduction': 0,
            'coord_rebuild': 0,
            'line_search': 0,
            'rfo_sphere': 0,
        }
        if event is not None:
            for step in inc_steps:
                causes['trust_reduction'] += int(step in event['trust_reduction_steps'])
                causes['coord_rebuild'] += int(step in event['rebuild_steps'])
                causes['line_search'] += int(step in event['nofit_steps'])
                causes['rfo_sphere'] += int(step in event['sphere_steps'])
        sm = structural_metrics(data_dir, name)
        paper = reference[name].get('paper_steps')
        per_molecule.append(
            {
                'name': name,
                'steps': int(len(energies)),
                'settled_step': settled_step(energies),
                'energy_increase_steps': int(len(inc_steps)),
                'max_jump_kcal': (
                    float(np.max(np.diff(arr)) * HARTREE_TO_KCAL)
                    if len(arr) > 1
                    else 0.0
                ),
                'paper_steps': paper,
                'paper_delta': None if paper is None else int(len(energies) - paper),
                **sm,
                'increase_causes': causes if event is not None else None,
            }
        )
    steps = [r['settled_step'] for r in per_molecule]
    corr = {
        'atoms': _pearson([r['atoms'] for r in per_molecule], steps),
        'dof': _pearson([r['dof'] for r in per_molecule], steps),
        'rotatable_bonds': _pearson(
            [r['rotatable_bonds'] for r in per_molecule], steps
        ),
        'hbond_network': _pearson([r['hbond_network'] for r in per_molecule], steps),
    }
    slow_small = [
        {
            **r,
            'near_linear_angles': near_linear_angles_count(data_dir, r['name']),
        }
        for r in per_molecule
        if r['atoms'] <= 10 and r['settled_step'] >= 20
    ]
    return {
        'per_molecule': per_molecule,
        'slowest': sorted(per_molecule, key=lambda x: x['settled_step'], reverse=True)[
            :10
        ],
        'non_monotonic': sorted(
            [r for r in per_molecule if r['energy_increase_steps'] > 0],
            key=lambda x: x['energy_increase_steps'],
            reverse=True,
        ),
        'correlation': corr,
        'slow_small': sorted(slow_small, key=lambda x: x['settled_step'], reverse=True),
        'paper_comparison': sorted(
            [r for r in per_molecule if r['paper_delta'] is not None],
            key=lambda x: abs(x['paper_delta']),
            reverse=True,
        ),
        'median_settled': statistics.median(steps) if steps else None,
        'max_settled': max(steps) if steps else None,
    }


def render_markdown(benchmark, solver, result):
    lines = [f'# Convergence analysis: {benchmark} ({solver})', '']
    lines.append(
        f"- median settled step @ {ENERGY_TOL_KCAL:.0e} kcal/mol: "
        f"{result['median_settled']}"
    )
    lines.append(f"- max settled step: {result['max_settled']}")
    lines.append('')
    lines.append('## Slowest systems (steps-to-settle)')
    lines.append('')
    lines.append(
        '| molecule | settled | total steps | atoms | dof | rot bonds | hbond net |'
    )
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in result['slowest']:
        lines.append(
            f"| {r['name']} | {r['settled_step']} | {r['steps']} | {r['atoms']} "
            f"| {r['dof']} | {r['rotatable_bonds']} | {r['hbond_network']} |"
        )
    lines.append('')
    lines.append('## Non-monotonic energy episodes')
    lines.append('')
    lines.append(
        '| molecule | #increases | max upward jump (kcal/mol) | attributed causes |'
    )
    lines.append('|---|---:|---:|---|')
    for r in result['non_monotonic'][:15]:
        causes = r['increase_causes']
        if causes is None:
            cause_s = 'n/a (no log file)'
        else:
            cause_s = ', '.join(f'{k}:{v}' for k, v in causes.items())
        lines.append(
            f"| {r['name']} | {r['energy_increase_steps']} | {r['max_jump_kcal']:.3g} "
            f"| {cause_s} |"
        )
    lines.append('')
    lines.append('## Step-count correlation with structure')
    lines.append('')
    lines.append('| metric | Pearson r vs settled step |')
    lines.append('|---|---:|')
    for k, v in result['correlation'].items():
        val = '-' if v is None or math.isnan(v) else f'{v:.3f}'
        lines.append(f'| {k} | {val} |')
    lines.append('')
    lines.append('## Slow small-molecule tail cases')
    lines.append('')
    lines.append('| molecule | atoms | settled | near-linear angles (>=175°) |')
    lines.append('|---|---:|---:|---:|')
    for r in result['slow_small']:
        lines.append(
            f"| {r['name']} | {r['atoms']} | {r['settled_step']} "
            f"| {r['near_linear_angles']} |"
        )
    lines.append('')
    lines.append('## Comparison vs Standard Method (paper) step counts')
    lines.append('')
    lines.append('| molecule | paper | measured | delta |')
    lines.append('|---|---:|---:|---:|')
    for r in result['paper_comparison'][:15]:
        lines.append(
            f"| {r['name']} | {r['paper_steps']} | {r['steps']} "
            f"| {r['paper_delta']:+d} |"
        )
    lines.append('')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--benchmark', choices=sorted(BENCHMARKS), default='birkholz')
    ap.add_argument('--solver', choices=['mopac', 'pyscf'], default='mopac')
    ap.add_argument(
        '--logs-dir',
        type=Path,
        default=None,
        help='optional directory containing <set>/<molecule>.log files',
    )
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--out-json', type=Path, default=None)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = Path(f'results/convergence-analysis-{args.benchmark}.md')
    if args.out_json is None:
        args.out_json = Path(f'results/convergence-analysis-{args.benchmark}.json')

    reference = json.loads((BENCHMARKS[args.benchmark] / 'reference.json').read_text())
    rows = load_rows(args.results_dir, args.benchmark, args.solver)
    if not rows:
        return 0
    result = analyze(
        rows=rows,
        benchmark=args.benchmark,
        reference=reference,
        data_dir=BENCHMARKS[args.benchmark],
        logs_dir=args.logs_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(args.benchmark, args.solver, result))
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f'wrote {args.out}')
    print(f'wrote {args.out_json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
