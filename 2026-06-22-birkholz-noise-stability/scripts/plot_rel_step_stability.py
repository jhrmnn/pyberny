#!/usr/bin/env python3
"""Plot per-system (RMSD-relative) step-count stability.

Left:  per-molecule step-count inflation (median steps / clean steps) under a
       fixed sigma vs per-system noise = 20 % of the start->minimum RMSD,
       sorted by the relative-noise value. Molecules that stay inflated
       (>1.5x) under relative noise are coloured red.
Right: per-molecule coefficient of variation of step count under 20 % relative
       noise.

Usage: plot_rel_step_stability.py rel_step_stability.json step_stability.json out.png [fixed_sigma_key]
"""

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INFLATED = 1.5  # threshold for the red "stays inflated under relative noise" colour


def main(rel_file, fix_file, outfile, fixed_sigma=None):
    rel_blob = json.loads(Path(rel_file).read_text())
    fix_blob = json.loads(Path(fix_file).read_text())
    rel = rel_blob['data']
    fix = fix_blob['data']
    bench = rel_blob.get('benchmark', '?')
    if fixed_sigma is None:
        sigmas = [str(s) for s in fix_blob['sigmas']]
        fixed_sigma = sigmas[len(sigmas) // 2]

    def infl_rel(n):
        s = rel[n]['by_frac'].get('0.2', {}).get('steps', [])
        return st.median(s) / rel[n]['clean_steps'] if s else float('nan')

    def infl_fix(n):
        s = fix.get(n, {}).get('by_sigma', {}).get(fixed_sigma, {}).get('steps', [])
        return st.median(s) / fix[n]['clean_steps'] if s and n in fix else float('nan')

    mols = sorted(
        [n for n in rel if rel[n]['by_frac'].get('0.2', {}).get('steps')], key=infl_rel
    )
    y = range(len(mols))
    colors = ['tab:red' if infl_rel(n) > INFLATED else 'tab:blue' for n in mols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, 0.32 * len(mols) + 2)))

    for i, n in enumerate(mols):
        a, b = infl_fix(n), infl_rel(n)
        if a == a:  # not NaN
            ax1.plot([a, b], [i, i], color='lightgray', zorder=1)
    ax1.scatter(
        [infl_fix(n) for n in mols],
        list(y),
        marker='x',
        color='gray',
        s=35,
        label=f'fixed sigma = {fixed_sigma} A',
        zorder=2,
    )
    ax1.scatter(
        [infl_rel(n) for n in mols],
        list(y),
        color=colors,
        s=55,
        label='per-system 20% of R$_{sm}$',
        zorder=3,
    )
    ax1.axvline(1.0, color='k', lw=0.8, ls=':')
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(mols, fontsize=8)
    ax1.set_xlabel('step-count inflation (median steps / clean steps)')
    ax1.set_title('Inflation: fixed vs per-system noise\n(red = stays >1.5x under relative noise)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)

    cvs = []
    for n in mols:
        s = rel[n]['by_frac']['0.2']['steps']
        cvs.append(100 * st.pstdev(s) / st.mean(s) if len(s) > 1 else 0)
    ax2.barh(list(y), cvs, color=colors)
    ax2.axvline(25, color='k', lw=0.8, ls=':')
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(mols, fontsize=8)
    ax2.set_xlabel('step-count CV under 20% relative noise (%)')
    ax2.set_title('Dispersion under per-system 20% noise\n(dotted line = 25%)')
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle(
        f'Per-system noise (RMSD = 20% of start->minimum distance): step-count stability\n'
        f'({bench} set, GFN2-xTB, frustrated-reference molecules excluded)',
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outfile, dpi=130)
    print(f'wrote {outfile}')


if __name__ == '__main__':
    main(*sys.argv[1:])
