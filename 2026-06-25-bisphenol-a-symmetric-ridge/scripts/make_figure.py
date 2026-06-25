#!/usr/bin/env python3
"""Build ``bisphenol_a_ridge.png`` from ``trajectory.py`` outputs.

    python scripts/trajectory.py out
    python scripts/make_figure.py out bisphenol_a_ridge.png

Panel (a) energy above each run's own minimum; (b) the ring-1 aryl torsion
(clean pinned near 0 deg on the near-Cs ridge, then breaking); (c) the trust
radius, with crosses marking steps whose Hessian has a negative eigenvalue.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HK = 627.5094740631
R1 = (10, 9, 0, 19)


def dihedral(C, t):
    p0, p1, p2, p3 = (C[i] for i in t)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def load(out, nm):
    tr = json.loads((out / f'{nm}.trace.json').read_text())
    traj = np.load(out / f'{nm}.traj.npy')
    return tr, traj


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'out')
    dest = sys.argv[2] if len(sys.argv) > 2 else 'bisphenol_a_ridge.png'
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for nm, label, col in [
        ('clean', 'clean (72 steps)', 'C3'),
        ('s0.05_seed3', 'σ=0.05 Å seed (26 steps)', 'C0'),
    ]:
        tr, traj = load(out, nm)
        E = np.array([r['energy'] for r in tr])
        st = np.arange(1, len(E) + 1)
        ax[0].semilogy(st, np.abs(E - E[-1]) * HK + 1e-4, '-o', ms=3, color=col, label=label)
        R = np.array([dihedral(c, R1) for c in traj])
        ax[1].plot(st, R, '-o', ms=3, color=col, label=label)
        tru = np.array([r['quadratic_step']['trust_radius'] for r in tr])
        neg = np.array([r['quadratic_step']['n_negative_eigenvalues'] for r in tr])
        ax[2].semilogy(st, tru, '-o', ms=3, color=col, label=label)
        ax[2].semilogy(st[neg > 0], tru[neg > 0], 'x', ms=7, mew=1.6, color=col)
    ax[0].set(xlabel='step', ylabel='|E − E_final|  (kcal/mol)', title='(a) energy above own minimum')
    ax[0].axhline(0.1, ls=':', c='grey', lw=0.8)
    ax[0].legend(fontsize=7, loc='upper right')
    ax[1].set(xlabel='step', ylabel='ring-1 torsion R1 (°)', title='(b) aryl-torsion coordinate')
    ax[1].axhline(0, ls=':', c='grey', lw=0.8)
    ax[1].legend(fontsize=7)
    ax[1].annotate('clean pinned on\nthe R1≈0 ridge', (18, 2), (28, -30), fontsize=7,
                   arrowprops=dict(arrowstyle='->', lw=0.8))
    ax[2].set(xlabel='step', ylabel='trust radius (a.u.)', title='(c) trust radius  (× = neg. Hessian eig.)')
    ax[2].legend(fontsize=7, loc='lower right')
    fig.suptitle('bisphenol_a GFN2-xTB: clean start crawls a near-Cs aryl-torsion ridge before descending', fontsize=10)
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    print('saved', dest)


if __name__ == '__main__':
    main()
