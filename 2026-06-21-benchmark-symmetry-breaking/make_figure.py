#!/usr/bin/env python3
"""Figure for the benchmark symmetry-breaking report.

Reads ``data/results.json`` (the measured none-vs-break step counts and final
energies, GFN2-xTB) and renders ``symmetry_breaking.png``: a per-molecule
step-count panel for the Baker molecules that change, plus an energy-gap panel
flagging which of those reach a genuinely lower minimum (the issue #148
saddles) versus merely re-converging to the same symmetric minimum.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
data = json.loads((HERE / 'data' / 'results.json').read_text())

# Baker molecules whose step count actually changes under symmetry breaking.
changed = [
    r
    for r in data['baker']
    if r['break_steps'] is not None
    and r['none_steps'] is not None
    and r['break_steps'] != r['none_steps']
]
changed.sort(key=lambda r: (r['dE_kcal'] if r['dE_kcal'] is not None else 0))

names = [r['name'] for r in changed]
none_steps = [r['none_steps'] for r in changed]
break_steps = [r['break_steps'] for r in changed]
dE = [r['dE_kcal'] for r in changed]
y = range(len(names))

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11, 7), gridspec_kw={'width_ratios': [1.1, 1]}
)

ax1.barh([i + 0.2 for i in y], none_steps, height=0.4, label='symmetric start', color='#9ecae1')
ax1.barh([i - 0.2 for i in y], break_steps, height=0.4, label="symmetry='break'", color='#fb6a4a')
ax1.set_yticks(list(y))
ax1.set_yticklabels(names)
ax1.set_xlabel('pyberny steps (GFN2-xTB)')
ax1.set_title('Step count: symmetric start vs broken start')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)

colors = ['#cb181d' if d < -0.05 else '#bdbdbd' for d in dE]
ax2.barh(list(y), dE, color=colors)
ax2.set_yticks(list(y))
ax2.set_yticklabels([])
ax2.set_xlabel(r'$E_\mathrm{break} - E_\mathrm{sym}$  (kcal/mol)')
ax2.set_title('Energy gained by breaking symmetry')
ax2.axvline(0, color='k', lw=0.8)
ax2.grid(axis='x', alpha=0.3)
for i, d in zip(y, dE):
    if d < -0.05:
        ax2.text(d, i, f' {d:.2f} ', va='center', ha='right', fontsize=8, color='#cb181d')

fig.suptitle(
    'Switching the Baker benchmark to symmetry-breaking starts (GFN2-xTB)',
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(HERE / 'symmetry_breaking.png', dpi=130)
print('wrote symmetry_breaking.png with', len(names), 'molecules')
