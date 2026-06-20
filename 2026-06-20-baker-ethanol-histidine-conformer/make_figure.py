#!/usr/bin/env python3
"""Figure for the ethanol/histidine conformer report (#154).

Grouped bars: the energy of the noise-found ``lower_min`` conformer relative to
the no-noise ``reference_min`` (the bundled Baker-start basin minimum), at
GFN2-xTB vs the paper's HF/6-31G** reference method. Reads ``data/results.json``.
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, 'data', 'results.json')) as f:
    res = json.load(f)

mols = ['ethanol', 'histidine']
gfn2 = [res[m]['gfn2']['gap_kcal'] for m in mols]
hf = [res[m]['hf_631gpp']['gap_kcal'] for m in mols]

x = np.arange(len(mols))
w = 0.36
fig, ax = plt.subplots(figsize=(6.4, 4.2))
b1 = ax.bar(x - w / 2, gfn2, w, label='GFN2-xTB', color='#c44e52')
b2 = ax.bar(x + w / 2, hf, w, label='HF/6-31G** (reference method)',
            color='#4c72b0')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(mols)
ax.set_ylabel(r'$E$(lower conformer) $-\ E$(Baker-start min)  [kcal/mol]')
ax.set_title('The conformer gap is a GFN2 artefact:\n'
             'at the reference method it collapses to noise')
ax.legend(loc='lower left', frameon=False)
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:+.2f}', (b.get_x() + b.get_width() / 2, h),
                    ha='center', va='bottom' if h >= 0 else 'top',
                    xytext=(0, 3 if h >= 0 else -3), textcoords='offset points',
                    fontsize=9)
ax.margins(y=0.18)
fig.tight_layout()
out = os.path.join(here, 'conformer_gap.png')
fig.savefig(out, dpi=150)
print('wrote', out)
