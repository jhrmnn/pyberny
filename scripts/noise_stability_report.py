#!/usr/bin/env python3
"""Turn ``noise_stability.py``'s ``raw.json`` into a markdown report + plots.

Reads ``<out-dir>/raw.json`` and writes ``summary.md`` plus PNG figures into
the same directory. Separated from the (slow) sweep so the analysis can be
re-rendered cheaply without re-running optimizations.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(out_dir):
    data = json.loads((out_dir / 'raw.json').read_text())
    return data['meta'], data['runs']


def by_level(runs, include_clean=False):
    """Group noisy runs by sigma (optionally folding in the clean baseline)."""
    groups = defaultdict(list)
    for r in runs:
        if r['seed'] is None and not include_clean:
            continue
        groups[r['sigma']].append(r)
    return dict(sorted(groups.items()))


def fmt(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def level_table(meta, runs):
    """Aggregate stability metrics per noise level."""
    gmax = meta['gradientmax']
    clean = [r for r in runs if r['seed'] is None]
    clean_conv = np.mean([r['converged'] for r in clean])
    clean_steps = np.mean([r['steps'] for r in clean])

    lines = [
        '| Gradient noise σ (a.u.) | Runs | Converged | Mean steps | '
        'Median steps | False conv.* | Mean true gmax | Mean RMSD vs clean (Å) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        f'| 0 (clean) | {len(clean)} | {clean_conv * 100:.0f}% | '
        f'{clean_steps:.1f} | {np.median([r["steps"] for r in clean]):.0f} | '
        f'0% | {np.mean([r["final_true_gmax"] for r in clean]):.2e} | 0.000 |',
    ]
    rows = []
    for sigma, grp in by_level(runs).items():
        conv = [r for r in grp if r['converged']]
        conv_rate = len(conv) / len(grp)
        # False convergence: optimizer declared success but the *true* gradient
        # max at the stopping point still exceeds pyberny's own threshold.
        false_conv = [r for r in conv if r['final_true_gmax'] > gmax]
        steps = [r['steps'] for r in grp]
        rmsds = [r['rmsd_from_clean'] for r in conv if r['rmsd_from_clean'] is not None]
        lines.append(
            f'| {sigma:.0e} | {len(grp)} | {conv_rate * 100:.0f}% | '
            f'{np.mean(steps):.1f} | {np.median(steps):.0f} | '
            f'{len(false_conv) / len(grp) * 100:.0f}% | '
            f'{np.mean([r["final_true_gmax"] for r in grp]):.2e} | '
            f'{(np.mean(rmsds) if rmsds else float("nan")):.3f} |'
        )
        rows.append((sigma, conv_rate, np.mean(steps), len(false_conv) / len(grp)))
    return '\n'.join(lines), rows


def molecule_table(meta, runs):
    """Per-molecule convergence rate at each noise level (robustness ranking)."""
    levels = meta['levels']
    mols = {}
    for r in runs:
        if r['seed'] is None:
            mols.setdefault(r['molecule'], {})['atoms'] = r['atoms']
            mols.setdefault(r['molecule'], {})['clean_steps'] = r['steps']
    per = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r['seed'] is None:
            continue
        per[r['molecule']][r['sigma']].append(r['converged'])
    header = (
        '| Molecule | Atoms | Clean steps | '
        + ' | '.join(f'σ={s:.0e}' for s in levels)
        + ' |'
    )
    sep = '|---|---:|---:|' + '|'.join(['---:'] * len(levels)) + '|'
    lines = [header, sep]

    # Sort by mean convergence rate across levels (most fragile last).
    def frag(m):
        return np.mean([np.mean(per[m][s]) if per[m][s] else 1.0 for s in levels])

    for m in sorted(mols, key=frag, reverse=True):
        cells = []
        for s in levels:
            vals = per[m][s]
            cells.append(f'{np.mean(vals) * 100:.0f}%' if vals else '-')
        lines.append(
            f'| {m} | {mols[m]["atoms"]} | {mols[m]["clean_steps"]} | '
            + ' | '.join(cells)
            + ' |'
        )
    return '\n'.join(lines)


def plot_summary(meta, runs, out_dir):
    levels = meta['levels']
    gmax = meta['gradientmax']
    # Convergence rate and false-convergence rate vs noise.
    conv_rate, false_rate, steps_mean, steps_p90 = [], [], [], []
    true_gmax = []
    for s in levels:
        grp = [r for r in runs if r['sigma'] == s and r['seed'] is not None]
        conv = [r for r in grp if r['converged']]
        conv_rate.append(len(conv) / len(grp))
        false_rate.append(
            len([r for r in conv if r['final_true_gmax'] > gmax]) / len(grp)
        )
        steps_mean.append(np.mean([r['steps'] for r in grp]))
        steps_p90.append(np.percentile([r['steps'] for r in grp], 90))
        true_gmax.append(np.mean([r['final_true_gmax'] for r in grp]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(levels, [c * 100 for c in conv_rate], 'o-', label='reported converged')
    ax.plot(
        levels,
        [(c - f) * 100 for c, f in zip(conv_rate, false_rate)],
        's--',
        label='true minimum',
    )
    ax.set_xscale('log')
    ax.set_xlabel('gradient noise σ (a.u.)')
    ax.set_ylabel('rate (%)')
    ax.set_title('Convergence vs noise')
    ax.axvline(gmax, color='grey', ls=':', lw=1)
    ax.axvline(meta['gradientrms'], color='grey', ls=':', lw=1)
    ax.text(gmax, 5, ' gradientmax', rotation=90, va='bottom', fontsize=8, color='grey')
    ax.set_ylim(-3, 103)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(levels, steps_mean, 'o-', label='mean')
    ax.plot(levels, steps_p90, 's--', label='90th pct')
    ax.axhline(
        np.mean([r['steps'] for r in runs if r['seed'] is None]),
        color='green',
        ls=':',
        label='clean mean',
    )
    ax.set_xscale('log')
    ax.set_xlabel('gradient noise σ (a.u.)')
    ax.set_ylabel('steps to stop')
    ax.set_title('Step count vs noise')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(levels, true_gmax, 'o-')
    ax.axhline(gmax, color='grey', ls=':', label='gradientmax threshold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('gradient noise σ (a.u.)')
    ax.set_ylabel('mean true gmax at stop (a.u.)')
    ax.set_title('Residual true gradient vs noise')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'noise_stability.png', dpi=130)
    plt.close(fig)


def plot_steps_box(meta, runs, out_dir):
    levels = meta['levels']
    data = [
        [r['steps'] for r in runs if r['sigma'] == s and r['seed'] is not None]
        for s in levels
    ]
    clean = [r['steps'] for r in runs if r['seed'] is None]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    positions = list(range(1, len(levels) + 1))
    ax.boxplot([clean, *data], positions=[0, *positions], widths=0.6)
    ax.set_xticks([0, *positions])
    ax.set_xticklabels(['clean', *(f'{s:.0e}' for s in levels)])
    ax.set_xlabel('gradient noise σ (a.u.)')
    ax.set_ylabel('steps to stop')
    ax.set_title('Step-count distribution across molecules & seeds')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_dir / 'noise_steps_box.png', dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out-dir', type=Path, default=Path('scripts/noise_stability_out'))
    args = ap.parse_args(argv)
    meta, runs = load(args.out_dir)

    ltable, _ = level_table(meta, runs)
    mtable = molecule_table(meta, runs)
    plot_summary(meta, runs, args.out_dir)
    plot_steps_box(meta, runs, args.out_dir)

    md = f"""# Baker benchmark: convergence stability under gradient noise

**Benchmark:** {meta['benchmark']} ({meta['n_molecules']} molecules) &nbsp;·&nbsp;
**Solver:** {meta['solver']} &nbsp;·&nbsp;
**Noise model:** {meta['noise_model']} &nbsp;·&nbsp;
**Seeds/level:** {meta['seeds']} &nbsp;·&nbsp; **maxsteps:** {meta['maxsteps']}

pyberny's convergence test fires when the gradient max/RMS fall below
`gradientmax = {meta['gradientmax']:.2e}` / `gradientrms = {meta['gradientrms']:.2e}`
(a.u.). Each optimization step the clean tblite gradient is perturbed by
additive Gaussian noise `N(0, σ²)` before being fed to the optimizer; the
energy is left clean so the experiment isolates the effect of *gradient* error.

## Aggregate stability vs noise level

{ltable}

\\* **False convergence** = pyberny reported convergence but the *true* (clean)
gradient max at the stopping geometry still exceeds `gradientmax`. The
"true minimum" curve in the plot is the reported rate minus this.

![stability](noise_stability.png)

![step distribution](noise_steps_box.png)

## Per-molecule convergence rate (most robust first)

Cells are the fraction of seeds that reported convergence at each σ.

{mtable}

*Total wall time for the sweep: {meta['wall_seconds']:.0f} s.*
"""
    (args.out_dir / 'summary.md').write_text(md)
    print(f'Wrote {args.out_dir / "summary.md"} and PNGs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
