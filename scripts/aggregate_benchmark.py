#!/usr/bin/env python3
"""Merge per-shard benchmark JSONs into a single markdown summary.

Reads ``<solver>-<batch_id>.json`` files produced by ``benchmark.py --out-json``,
re-sorts rows to ``sorted(reference)`` order, renders one markdown table per
solver via ``benchmark.format_table`` / ``format_errors``, appends the result
to ``$GITHUB_STEP_SUMMARY`` (when set) and writes ``results/summary.md``.

Exits 1 if any molecule failed to converge and its reference entry for that
solver is not ``null`` — same rule as ``benchmark.py``'s exit-code logic.
``pyberny_steps`` is currently ``null`` for every entry, so pyscf is in
baseline-establishment mode: no pyscf run can trip this gate yet. Fill in
``pyberny_steps`` as pyscf results stabilize to enable the regression check.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import DATA, format_errors, format_table  # noqa: E402

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


def render(reference, solver, rows_by_name):
    rows = [rows_by_name[n] for n in sorted(reference) if n in rows_by_name]
    return format_table(rows, solver, reference) + format_errors(rows)


def violations(reference, solver, rows_by_name):
    key = REF_KEY[solver]
    return [
        n
        for n, r in rows_by_name.items()
        if not r['converged'] and reference[n][key] is not None
    ]


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
            f'{solver}/{n}' for n in violations(reference, solver, rows_by_name)
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
        print(f'\nUnexpected non-convergence: {failed}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
