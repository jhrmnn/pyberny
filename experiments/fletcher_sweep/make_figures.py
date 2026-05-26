#!/usr/bin/env python3
"""Render the figures embedded in README.md from results.json.

Re-run after re-running ``run_sweep.py``; commits the PNGs alongside the
JSON so the README displays without needing matplotlib at view time.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE / 'results.json').read_text())
    settings = [p['name'] for p in data['params']]
    benches = ['birkholz', 'baker']

    # ---- Figure 1: good-step fraction per setting per benchmark --------
    good = {b: [data['cells'][s][b]['summary']['good_step_fraction']
                for s in settings] for b in benches}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(settings))
    w = 0.38
    ax.bar(x - w / 2, good['birkholz'], w, label='birkholz (19 mols)', color='#3b6fbd')
    ax.bar(x + w / 2, good['baker'], w, label='baker (30 mols)', color='#d97a3e')
    base = good['birkholz'][0]
    ax.axhline(base, color='#3b6fbd', linestyle=':', linewidth=0.8, alpha=0.7)
    base_b = good['baker'][0]
    ax.axhline(base_b, color='#d97a3e', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=25, ha='right')
    ax.set_ylabel(r'good-step fraction  ($r \in [0.75, 1.25]$)')
    ax.set_title("Fletcher's parameter falls in the good band ~40 % of steps")
    ax.set_ylim(0, 0.55)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / 'good_step_fraction.png', dpi=140)
    plt.close(fig)

    # ---- Figure 2: histograms of r per setting, both benchmarks --------
    # Use clipped r values so outliers don't dominate the axes.
    clip = (-1.0, 4.0)
    fig, axes = plt.subplots(
        len(settings), 2, figsize=(9.0, 1.55 * len(settings)),
        sharex=True, sharey='row',
    )
    bins = np.linspace(*clip, 41)
    for i, s in enumerate(settings):
        for j, b in enumerate(benches):
            ax = axes[i, j]
            rs = [rec['fletcher']
                  for mol in data['cells'][s][b]['per_mol'].values()
                  for rec in mol['records']
                  if rec['fletcher'] is not None]
            rs_c = np.clip(rs, *clip)
            ax.hist(rs_c, bins=bins,
                    color=('#3b6fbd' if b == 'birkholz' else '#d97a3e'),
                    edgecolor='white', linewidth=0.3)
            ax.axvspan(0.75, 1.25, color='green', alpha=0.10, zorder=0)
            ax.axvline(1.0, color='black', linewidth=0.5, alpha=0.5)
            if i == 0:
                ax.set_title(b)
            if j == 0:
                ax.set_ylabel(s, rotation=0, ha='right', va='center', fontsize=9)
            ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"Fletcher's $r = \Delta E / \Delta E_{\rm pred}$  (clipped to [-1, 4])")
    fig.suptitle(r"Distribution of Fletcher's $r$ — green band is the good-step window",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(HERE / 'r_distribution.png', dpi=140)
    plt.close(fig)

    print(f'wrote {HERE / "good_step_fraction.png"} '
          f'and {HERE / "r_distribution.png"}')


if __name__ == '__main__':
    main()
