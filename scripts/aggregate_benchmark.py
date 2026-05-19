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
    BENCHMARKS,
    REF_STEPS_KEY,
    format_errors,
    format_table,
    regression_reason,
)


def load_rows(results_dir, solver, benchmark):
    rows = {}
    sources = {}
    # Filenames are ``<solver>-<benchmark>-<batch_id>.json`` (set by the CI
    # workflow's Run-batch step). Globbing on both axes lets a single
    # results directory hold shards from multiple benchmarks side-by-side,
    # which is what the push/pull_request auto-trigger produces.
    for path in sorted(results_dir.glob(f'{solver}-{benchmark}-*.json')):
        data = json.loads(path.read_text())
        for row in data['rows']:
            name = row['name']
            if name in rows:
                raise SystemExit(
                    f'duplicate row for {solver}/{benchmark}/{name}: '
                    f'in {sources[name].name} and {path.name}'
                )
            rows[name] = row
            sources[name] = path
    return rows


def totals_row(rows, reference, solver):
    """Render a markdown totals row summing the per-molecule columns.

    ``paper_steps`` and the per-solver reference may be ``None`` (no paper
    reference / documented non-converger); those are skipped from their
    respective sums so a missing entry doesn't poison the total.
    ``Converged`` is reported as ``converged/total`` rather than summed.

    Rows whose ``wall`` is ``None`` are "not run" placeholders inserted by
    :func:`render` so the table always covers every molecule in
    ``reference.json``; they're skipped from the wall and step sums and
    don't count as converged, but still contribute to ``atoms`` and to the
    ``total`` denominator (so a partial CI run is visibly partial).
    """
    solver_key = REF_STEPS_KEY[solver]
    run_rows = [r for r in rows if r.get('wall') is not None]
    atoms = sum(reference[r['name']]['atoms'] for r in rows)
    paper = sum(
        reference[r['name']].get('paper_steps') or 0
        for r in rows
        if reference[r['name']].get('paper_steps') is not None
    )
    ref_total = sum(
        reference[r['name']].get(solver_key) or 0
        for r in rows
        if reference[r['name']].get(solver_key) is not None
    )
    steps = sum(r['steps'] for r in run_rows if r['steps'] is not None)
    converged = sum(1 for r in run_rows if r['converged'])
    wall = sum(r['wall'] for r in run_rows)
    return (
        f'| **Total** | **{atoms}** | **{paper}** | **{ref_total}** '
        f'| **{steps}** | **{converged}/{len(rows)}** | **{wall:.1f}** |\n'
    )


def _stub_row(name):
    # ``wall is None`` is the sentinel both format_table and totals_row use
    # to recognize a "not run" placeholder.
    return {
        'name': name,
        'converged': False,
        'steps': None,
        'wall': None,
        'error': None,
    }


def render(reference, solver, rows_by_name):
    """Render the table for ``solver``, padding missing molecules with stubs.

    Iterating over ``sorted(reference)`` (rather than only ``rows_by_name``)
    keeps the table shape stable across partial CI runs: a shard failure or
    a ``--molecules`` subset leaves a placeholder line per missing molecule
    rather than silently shrinking the table.
    """
    rows = [rows_by_name.get(n, _stub_row(n)) for n in sorted(reference)]
    return (
        format_table(rows, solver, reference)
        + totals_row(rows, reference, solver)
        + format_errors(rows)
    )


def violations(reference, solver, rows_by_name):
    key = REF_STEPS_KEY[solver]
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
    ap.add_argument(
        '--benchmark',
        choices=sorted(BENCHMARKS),
        default='birkholz',
        help='which benchmark these shards belong to (default: birkholz)',
    )
    ap.add_argument(
        '--reference',
        type=Path,
        default=None,
        help='reference.json path (default: derived from --benchmark)',
    )
    ap.add_argument(
        '--out',
        type=Path,
        default=None,
        help='summary path (default: results/summary-<benchmark>.md)',
    )
    args = ap.parse_args(argv)
    if args.reference is None:
        args.reference = BENCHMARKS[args.benchmark] / 'reference.json'
    if args.out is None:
        args.out = Path(f'results/summary-{args.benchmark}.md')

    reference = json.loads(args.reference.read_text())
    parts = []
    failed = []
    for solver in args.solvers:
        rows_by_name = load_rows(args.results_dir, solver, args.benchmark)
        if not rows_by_name:
            continue
        parts.append(f'## {solver}\n\n' + render(reference, solver, rows_by_name))
        failed.extend(
            f'{solver}/{n}: {reason}'
            for n, reason in violations(reference, solver, rows_by_name)
        )

    # Skip writing the summary entirely if no shards for this benchmark were
    # produced -- otherwise a multi-benchmark CI run that only exercised one
    # set would leave an empty summary-<other>.md artifact.
    if not parts:
        return 0
    summary = f'# {args.benchmark}\n\n' + '\n'.join(parts)
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
