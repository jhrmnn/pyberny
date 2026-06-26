#!/usr/bin/env python3
"""Three-panel figure: the bisphenol_a torsional-ridge pathology (GFN2-xTB).

(a) energy above the final minimum vs step, clean (long plateau) vs noisy;
(b) the two aryl torsions along the clean run -- frozen near-equal for ~34
    steps, then the near-symmetry breaks and they split to the asymmetric
    minimum; (c) lowest Hessian eigenvalue per step -- the clean run repeatedly
    re-enters negative curvature (the ridge), the noisy run does not.

    python plot.py --data ../data --out ../bisphenol_a_ridge.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HARTREE_KCAL = 627.5094740631


def erel(tr):
    e = np.array([r['energy'] for r in tr])
    return (e - e[-1]) * HARTREE_KCAL


def loweig(tr):
    return [r.get('quadratic_step', {}).get('lowest_eigenvalue') for r in tr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('../data'))
    ap.add_argument('--out', type=Path, default=Path('../bisphenol_a_ridge.png'))
    args = ap.parse_args()

    clean = json.load(open(args.data / 'xtb_trace_clean.json'))
    noisy = json.load(open(args.data / 'xtb_trace_noisy.json'))
    tors = json.load(open(args.data / 'xtb_torsions.json'))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    ec, en = erel(clean), erel(noisy)
    ax[0].semilogy(range(1, len(ec) + 1), ec + 1e-4, '-o', ms=3, color='C3',
                   label=f'clean ({len(ec)} steps)')
    ax[0].semilogy(range(1, len(en) + 1), en + 1e-4, '-o', ms=3, color='C0',
                   label=f'noisy σ=0.02 Å ({len(en)} steps)')
    ax[0].set_xlabel('optimization step')
    ax[0].set_ylabel('E − E$_{final}$ (kcal/mol)')
    ax[0].set_title('(a) Descent: clean grinds on a plateau')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    t1, t2 = tors['clean']['torsions']['t1'], tors['clean']['torsions']['t2']
    x = range(1, len(t1) + 1)
    ax[1].plot(x, t1, '-o', ms=3, color='C2', label='aryl torsion 1')
    ax[1].plot(x, t2, '-o', ms=3, color='C4', label='aryl torsion 2')
    ax[1].plot(x, np.array(t1) - np.array(t2), '--', color='k', lw=1.2,
               label='torsion 1 − 2 (asymmetry)')
    ax[1].axvspan(1, 45, color='C3', alpha=0.10)
    ax[1].text(23, 52, 'near-symmetric\nridge plateau', ha='center',
               fontsize=8, color='C3')
    ax[1].set_xlabel('optimization step (clean run)')
    ax[1].set_ylabel('dihedral (deg)')
    ax[1].set_title('(b) Clean run: torsions locked ~45 steps,\nthen symmetry breaks')
    ax[1].legend(fontsize=8, loc='center right'); ax[1].grid(alpha=0.3)

    lc, ln = loweig(clean), loweig(noisy)
    ax[2].plot(range(1, len(lc) + 1), lc, '-o', ms=3, color='C3', label='clean')
    ax[2].plot(range(1, len(ln) + 1), ln, '-o', ms=3, color='C0', label='noisy')
    ax[2].axhline(0, color='k', lw=0.8)
    ax[2].set_ylim(-1.2, 0.02)
    ax[2].set_xlabel('optimization step')
    ax[2].set_ylabel('lowest Hessian eigenvalue (a.u.)')
    ax[2].set_title('(c) Clean run repeatedly enters\nnegative curvature (ridge)')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
