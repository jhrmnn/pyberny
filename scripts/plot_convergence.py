#!/usr/bin/env python3
"""Plot energy-convergence curves for one benchmark's molecules.

Reads the per-shard ``<solver>-<benchmark>-<batch_id>.json`` files written by
``benchmark.py --out-json`` (the same artifacts the CI ``aggregate`` job
downloads) and draws a single combined figure: one curve per molecule of the
energy above the run's minimum, ``E_n - min(E)``, in kcal/mol on a log axis
against the optimization step. The converged point itself (``E_n == min(E)``)
is dropped since it has no place on a log axis. Saved to
``results/convergence-<benchmark>-<solver>.png``.

Usage::

    scripts/plot_convergence.py --results-dir DIR [--benchmark NAME]
                                [--solver SOLVER] [--out PATH]

``--benchmark`` mirrors ``benchmark.py``; the CI auto-trigger runs every solver
on both reference sets, so the workflow invokes this once per (benchmark,
solver) axis to get a plot for each. ``--solver`` accepts any string -- the plot
is drawn straight from the recorded energy trajectories, which have the same
shape for every runner, so there is no per-solver logic to gate on.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import BENCHMARKS

HARTREE_TO_KCAL = 627.503


def load_trajectories(results_dir, benchmark, solver):
    """Return ``{molecule: [energy, ...]}`` for one (benchmark, solver) axis.

    Skips rows without a recorded energy trajectory (e.g. errored runs), so a
    partial CI run simply contributes fewer curves rather than failing.
    """
    trajectories = {}
    for path in sorted(results_dir.glob(f'{solver}-{benchmark}-*.json')):
        # Per-molecule trace files (``<solver>-<benchmark>-<name>.trace.json``,
        # written by benchmark.py --out-trace-dir) live alongside the batch
        # shards and match the same glob; they're JSON lists rather than the
        # ``{'rows': [...]}`` shards we want here, so skip them.
        if path.name.endswith('.trace.json'):
            continue
        data = json.loads(path.read_text())
        for row in data['rows']:
            energies = row.get('energies')
            if energies:
                trajectories[row['name']] = energies
    return trajectories


def plot(benchmark, solver, trajectories, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, energies in sorted(trajectories.items()):
        energies = np.array(energies, dtype=float)
        delta = (energies - energies.min()) * HARTREE_TO_KCAL
        steps = np.arange(len(energies))
        mask = delta > 0
        ax.semilogy(steps[mask], delta[mask], marker='.', ms=3, lw=1, label=name)
    ax.set_xlabel('optimization step')
    ax.set_ylabel(r'$E - E_\mathrm{min}$  (kcal/mol)')
    ax.set_title(f'{benchmark} energy convergence ({solver})')
    ax.grid(True, which='both', ls=':', lw=0.5, alpha=0.5)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize='small',
        ncol=1 if len(trajectories) <= 20 else 2,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument(
        '--benchmark',
        choices=sorted(BENCHMARKS),
        default='birkholz',
        help='which benchmark these shards belong to (default: birkholz)',
    )
    ap.add_argument(
        '--solver',
        default='xtb',
        help=(
            'which solver the shards were run with (default: xtb). Any '
            'value is accepted: the plot is built straight from the recorded '
            'energy trajectories, which have the same shape for every runner, '
            'so no per-solver logic gates this script.'
        ),
    )
    ap.add_argument(
        '--out',
        type=Path,
        default=None,
        help='output PNG path (default: results/convergence-<benchmark>-<solver>.png)',
    )
    args = ap.parse_args(argv)
    if args.out is None:
        # Include the solver so plotting multiple solvers for one benchmark
        # writes a figure per axis rather than overwriting a shared file.
        args.out = Path(f'results/convergence-{args.benchmark}-{args.solver}.png')

    trajectories = load_trajectories(args.results_dir, args.benchmark, args.solver)
    # Mirror aggregate_benchmark.py: skip a benchmark/solver axis with no shard
    # files so a multi-benchmark run that only exercised one set doesn't error or
    # leave an empty figure behind. Now that --solver accepts any string, a
    # typo'd name lands here too; note it on stderr so it isn't indistinguishable
    # from a legitimately empty axis, but still exit 0 so a workflow loop over
    # solvers stays unaffected by axes it didn't run.
    if not trajectories:
        print(
            f'no {args.solver}-{args.benchmark}-*.json shards in '
            f'{args.results_dir}; nothing to plot',
            file=sys.stderr,
        )
        return 0
    plot(args.benchmark, args.solver, trajectories, args.out)
    print(f'wrote {args.out} ({len(trajectories)} molecules)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
