#!/usr/bin/env python3
"""Re-draw the benzene -> pseudo-minimum interpolation path, extended past the
endpoint so the well at the pseudo-minimum (fulvene) becomes visible.

The #147 ``minima_paths.png`` plots a linear-Cartesian interpolation from
benzene (t=0) to the noise-found structure (t=1) and *stops* at t=1, so the
pseudo-minimum shows up as the end of a ramp with no well -- an artefact of
only ever sampling the approach side of an endpoint. Here we Kabsch-align the
two structures (as #147 did), interpolate, and continue the *same* straight
line past t=1; the energy turns over and rises again, revealing the local
minimum. A linear-scale inset zooms the basin.

    ./interpolation_plot.py        # writes interpolation_path.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tblite.interface import Calculator

from berny.species_data import get_property

A2B = 1.8897261246257702
HA2KCAL = 627.509474


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    syms, xyz = [], []
    for line in lines[2 : 2 + n]:
        s, x, y, z = line.split()[:4]
        syms.append(s)
        xyz.append([float(x), float(y), float(z)])
    return syms, np.array(xyz)


def energy(syms, xyz_ang):
    num = np.array([int(get_property(s, 'number')) for s in syms])
    calc = Calculator('GFN2-xTB', num, xyz_ang * A2B, charge=0.0, uhf=0)
    calc.set('verbosity', 0)
    calc.set('accuracy', 1e-3)
    return float(calc.singlepoint().get('energy'))


def kabsch(P, Q):
    """Rotate/translate Q onto P (both N-by-3); return aligned Q."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Qc.T @ Pc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    return Qc @ R + P.mean(0)


def main():
    here = Path(__file__).parent
    syms, xb = read_xyz(here / 'benzene_reference_min.xyz')
    _, xf = read_xyz(here / 'benzene_pseudo_min.xyz')
    xf = kabsch(xb, xf)  # align as #147 did, so the line is the physical one

    eb = energy(syms, xb)
    rmsd_full = np.sqrt(((xf - xb) ** 2).sum(1).mean())  # benzene->fulvene RMSD

    ts = np.linspace(-0.05, 1.25, 79)
    de, rmsd = [], []
    for t in ts:
        x = xb + t * (xf - xb)
        de.append((energy(syms, x) - eb) * HA2KCAL)
        rmsd.append(np.sqrt(((x - xb) ** 2).sum(1).mean()))
    de = np.array(de)
    rmsd = np.array(rmsd)
    i1 = int(np.argmin(np.abs(ts - 1.0)))  # endpoint (pseudo-min) index
    i0 = int(np.argmin(np.abs(ts - 0.0)))  # benzene index (t=0, E-Eb≈0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- left: full path, log scale, #147 style, extended past the endpoint ---
    upto = ts <= 1.0
    beyond = ts >= 1.0
    axL.semilogy(rmsd[upto], np.clip(de[upto], 1e-2, None), '-', color='C0',
                 label='benzene → pseudo-min (as in #147)')
    axL.semilogy(rmsd[beyond], np.clip(de[beyond], 1e-2, None), '--', color='C3',
                 label='extension past the endpoint (t > 1)')
    axL.plot(rmsd[i0], max(de[i0], 1e-2), '*', ms=16, color='gold', mec='k',
             zorder=6, label='benzene (true min)')
    axL.plot(rmsd[i1], de[i1], 'o', ms=9, color='C3', zorder=5,
             label=f'pseudo-min (fulvene, +{de[i1]:.1f})')
    ipk = int(np.argmax(de[ts <= 1.0]))
    axL.annotate('linear-interp. clash\n(bond-breaking, barrier = upper bound)',
                 xy=(rmsd[ipk], de[ipk]), xytext=(rmsd[ipk] + 0.04, de[ipk] * 0.12),
                 fontsize=7.5, ha='left', color='dimgray',
                 arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.7))
    axL.set_xlabel('path coord. (RMSD from benzene, Å)')
    axL.set_ylabel('E − E$_{benzene}$ (kcal/mol, log)')
    axL.set_title('Full interpolation — endpoint looks like a ramp')
    axL.legend(fontsize=8, loc='lower right')
    axL.grid(True, which='both', alpha=0.25)

    # --- right: linear zoom on the basin around the endpoint ---
    zoom = (ts >= 0.82) & (ts <= 1.18)
    axR.plot(ts[zoom], de[zoom], '-o', ms=4, color='C0')
    axR.plot(1.0, de[i1], 'o', ms=10, color='C3', zorder=5,
             label=f'pseudo-min, t=1 (+{de[i1]:.1f})')
    axR.axvline(1.0, color='C3', ls=':', alpha=0.6)
    axR.set_xlabel('interpolation parameter t')
    axR.set_ylabel('E − E$_{benzene}$ (kcal/mol, linear)')
    axR.set_title('Zoom: the endpoint is the bottom of a well')
    axR.legend(fontsize=9)
    axR.grid(True, alpha=0.3)

    fig.suptitle(
        'benzene ⇄ noise-found pseudo-minimum (fulvene): the well is real, '
        'just truncated in the #147 one-sided path',
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = here / 'interpolation_path.png'
    fig.savefig(out, dpi=140)
    print(f'benzene→fulvene RMSD = {rmsd_full:.3f} Å, pseudo-min = +{de[i1]:.1f} kcal/mol')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
