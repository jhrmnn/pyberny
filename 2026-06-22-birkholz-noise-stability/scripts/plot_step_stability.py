#!/usr/bin/env python3
"""Plot step-count stability from a step_stability.json.

Two panels:
  (left)  per-molecule step-count box plots at a chosen sigma (default the
          middle amplitude), with the no-noise (clean) step count overlaid;
  (right) ECDF of step count normalised by each molecule's median, pooled over
          all molecules, one curve per noise amplitude, with a +/-25% band.

Usage: plot_step_stability.py step_stability.json out.png [sigma_key]
"""

import json
import statistics as st
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def main(infile, outfile, sigma_key=None):
    blob = json.loads(open(infile).read())
    d = blob['data']
    bench = blob.get('benchmark', '?')
    sigmas = [str(s) for s in blob['sigmas']]
    if sigma_key is None:
        sigma_key = sigmas[len(sigmas) // 2]
    # Only molecules with data at the chosen sigma.
    have = [n for n in d if d[n]['by_sigma'].get(sigma_key, {}).get('steps')]
    mols = sorted(have, key=lambda n: st.median(d[n]['by_sigma'][sigma_key]['steps']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(6, 0.32 * len(mols) + 2)))

    data = [d[n]['by_sigma'][sigma_key]['steps'] for n in mols]
    ax1.boxplot(
        data,
        orientation='horizontal',
        widths=0.6,
        showfliers=True,
        medianprops=dict(color='tab:blue'),
        flierprops=dict(marker='.', ms=4, mfc='gray', mec='gray'),
    )
    ax1.scatter(
        [d[n]['clean_steps'] for n in mols],
        range(1, len(mols) + 1),
        color='tab:red',
        zorder=5,
        s=40,
        label='no-noise (clean) steps',
    )
    ax1.set_yticks(range(1, len(mols) + 1))
    ax1.set_yticklabels(mols, fontsize=8)
    ax1.set_xlabel('steps to convergence')
    ax1.set_title(
        f'Step-count distribution at sigma={sigma_key} A\n'
        f'(same-minimum trials)'
    )
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)

    palette = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple']
    for sig, col in zip(sigmas, palette):
        norm = []
        for n in d:
            s = d[n]['by_sigma'].get(sig, {}).get('steps', [])
            if not s:
                continue
            m = st.median(s)
            norm += [x / m for x in s]
        if not norm:
            continue
        norm = np.sort(norm)
        ecdf = np.arange(1, len(norm) + 1) / len(norm)
        cvs = [
            100 * st.pstdev(d[n]['by_sigma'][sig]['steps'])
            / st.mean(d[n]['by_sigma'][sig]['steps'])
            for n in d
            if len(d[n]['by_sigma'].get(sig, {}).get('steps', [])) > 1
        ]
        cv = st.median(cvs) if cvs else 0
        ax2.plot(norm, ecdf, color=col, label=f'sigma={sig} A (median CV {cv:.0f}%)')
    ax2.axvspan(0.75, 1.25, color='gray', alpha=0.12, label='+/-25% of molecule median')
    ax2.axvline(1.0, color='k', lw=0.8, ls=':')
    ax2.set_xlabel('step count / molecule median')
    ax2.set_ylabel('cumulative fraction of trials')
    ax2.set_title('Relative step-count dispersion\n(pooled over molecules)')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 2.5)

    fig.suptitle(
        f'Trajectory step-count stability under small start-geometry noise\n'
        f'({bench} set, GFN2-xTB, frustrated-reference molecules excluded)',
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outfile, dpi=130)
    print(f'wrote {outfile}')


if __name__ == '__main__':
    main(*sys.argv[1:])
