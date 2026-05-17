#!/usr/bin/env python3
"""Merge per-shard benchmark JSONs into a single markdown summary.

Reads ``<solver>-<batch_id>.json`` files produced by ``benchmark.py --out-json``,
re-sorts rows to ``sorted(reference)`` order, renders one markdown table per
solver via ``benchmark.format_table`` / ``format_errors``, appends the result
to ``$GITHUB_STEP_SUMMARY`` (when set) and writes ``results/summary.md``.

Exits 1 if any molecule with a non-``null`` reference entry for the active
solver either failed to converge or drifts from its reference step count
by more than 7% (with an absolute floor of 2 steps) — same rule as
``benchmark.py``'s exit-code logic. ``pyberny_steps`` is currently
``null`` for every entry, so pyscf is in baseline-establishment mode: no
pyscf run can trip either half of this gate yet. Fill in ``pyberny_steps``
as pyscf results stabilize to enable the regression check.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import (  # noqa: E402
    DATA,
    format_errors,
    format_table,
    regression_reason,
)

REF_KEY = {'mopac': 'mopac_pm7_steps', 'pyscf': 'pyberny_steps'}


def load_rows(results_dir, solver):
    rows = {}
    sources = {}
    for path in sorted(results_dir.glob(f'{solver}-*.json')):
        data = json.loads(path.read_text())
        for row in data['rows']:
            name = row['name']
            if name in rows:
                raise SystemExit(
                    f'duplicate row for {solver}/{name}: '
                    f'in {sources[name].name} and {path.name}'
                )
            rows[name] = row
            sources[name] = path
    return rows


def totals_row(rows, reference):
    """Render a markdown totals row summing the per-molecule columns.

    ``paper_steps`` and ``steps`` may be ``None`` (no paper reference /
    non-converged run); those are skipped from their respective sums so a
    missing entry doesn't poison the total. ``Converged`` is reported as
    ``converged/total`` rather than summed.
    """
    atoms = sum(reference[r['name']]['atoms'] for r in rows)
    paper = sum(
        reference[r['name']].get('paper_steps') or 0
        for r in rows
        if reference[r['name']].get('paper_steps') is not None
    )
    steps = sum(r['steps'] for r in rows if r['steps'] is not None)
    converged = sum(1 for r in rows if r['converged'])
    wall = sum(r['wall'] for r in rows)
    return (
        f'| **Total** | **{atoms}** | **{paper}** | **{steps}** '
        f'| **{converged}/{len(rows)}** | **{wall:.1f}** |\n'
    )


def render(reference, solver, rows_by_name):
    rows = [rows_by_name[n] for n in sorted(reference) if n in rows_by_name]
    return (
        format_table(rows, solver, reference)
        + totals_row(rows, reference)
        + format_errors(rows)
    )


def violations(reference, solver, rows_by_name):
    key = REF_KEY[solver]
    out = []
    for n, r in rows_by_name.items():
        reason = regression_reason(r, reference[n][key])
        if reason:
            out.append((n, reason))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--solvers', nargs='+', required=True)
    ap.add_argument('--reference', type=Path, default=DATA / 'reference.json')
    ap.add_argument('--out', type=Path, default=Path('results/summary.md'))
    args = ap.parse_args(argv)

    reference = json.loads(args.reference.read_text())
    parts = []
    failed = []
    for solver in args.solvers:
        rows_by_name = load_rows(args.results_dir, solver)
        if not rows_by_name:
            continue
        parts.append(f'## {solver}\n\n' + render(reference, solver, rows_by_name))
        failed.extend(
            f'{solver}/{n}: {reason}'
            for n, reason in violations(reference, solver, rows_by_name)
        )

    summary = '\n'.join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary)
    step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary:
        with open(step_summary, 'a') as f:
            f.write(summary)
    print(summary, end='')

    if failed:
        print('\nBenchmark regressions:', file=sys.stderr)
        for entry in failed:
            print(f'  {entry}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
