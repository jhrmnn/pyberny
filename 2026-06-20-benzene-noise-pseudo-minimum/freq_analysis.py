#!/usr/bin/env python3
"""Vibrational (Hessian) analysis of the benzene noise-found "pseudo-minimum".

Issue #149: a large-amplitude noise perturbation drives pyberny to a structure
it reports as *converged* (gmax ~ 2.3e-5, tighter than the real planar minimum)
yet which sits ~+32 kcal/mol above the planar D6h minimum. The first-order
convergence test (gradient + step norms) cannot tell a minimum from a saddle
point, so we settle the question directly: build the Hessian by finite
differences of the analytic gradient, project out translations/rotations, and
read off the vibrational frequencies. A genuine minimum has all-real
frequencies; a saddle point has one or more imaginary ones.

Runs with GFN2-xTB (the benchmark surface). For the HF cross-check — which
needs an HF *re*-optimization first, since this geometry is only a GFN2
stationary point — see ``hf_crosscheck.py``.

    investigations/noise_minima/freq_analysis.py            # GFN2-xTB
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HARTREE_BOHR2_AMU_TO_CM1 = 5140.4838  # sqrt(Eh / bohr^2 / amu) -> cm^-1
ANGSTROM_TO_BOHR = 1.8897261246257702

# Atomic masses (amu) for the only elements that appear here.
MASS = {'H': 1.007825, 'C': 12.0, 'N': 14.003074, 'O': 15.994915}


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, str]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    comment = lines[1]
    syms, xyz = [], []
    for line in lines[2 : 2 + n]:
        sym, x, y, z = line.split()[:4]
        syms.append(sym)
        xyz.append([float(x), float(y), float(z)])
    return syms, np.array(xyz), comment


def gfn2_gradient(syms: list[str], xyz_ang: np.ndarray) -> tuple[float, np.ndarray]:
    """Energy (Hartree) and gradient (Hartree/bohr) from GFN2-xTB via tblite."""
    from tblite.interface import Calculator

    from berny.species_data import get_property

    numbers = np.array([int(get_property(s, 'number')) for s in syms])
    positions = xyz_ang * ANGSTROM_TO_BOHR
    calc = Calculator('GFN2-xTB', numbers, positions, charge=0.0, uhf=0)
    calc.set('verbosity', 0)
    calc.set('accuracy', 1e-3)  # tight, for clean finite differences
    res = calc.singlepoint()
    return float(res.get('energy')), np.array(res.get('gradient'))


def hessian(syms, xyz_ang, grad_fn, delta_bohr=0.01):
    """Cartesian Hessian (Hartree/bohr^2) by central differences of the gradient."""
    natom = len(syms)
    xyz_bohr = xyz_ang * ANGSTROM_TO_BOHR
    ncoord = 3 * natom
    H = np.zeros((ncoord, ncoord))
    for i in range(ncoord):
        a, c = divmod(i, 3)
        xp = xyz_bohr.copy()
        xp[a, c] += delta_bohr
        xm = xyz_bohr.copy()
        xm[a, c] -= delta_bohr
        _, gp = grad_fn(syms, xp / ANGSTROM_TO_BOHR)
        _, gm = grad_fn(syms, xm / ANGSTROM_TO_BOHR)
        H[i] = ((gp - gm) / (2 * delta_bohr)).reshape(-1)
    return 0.5 * (H + H.T)  # symmetrize


def projector(syms, xyz_ang):
    """Orthonormal basis of the 6 (or 5) translation/rotation modes, mass-weighted."""
    masses = np.array([MASS[s] for s in syms])
    xyz_bohr = xyz_ang * ANGSTROM_TO_BOHR
    com = (masses[:, None] * xyz_bohr).sum(0) / masses.sum()
    r = xyz_bohr - com
    sm = np.sqrt(np.repeat(masses, 3))
    natom = len(syms)
    D = np.zeros((3 * natom, 6))
    for a in range(natom):
        for c in range(3):
            D[3 * a + c, c] = 1.0  # translations
    for a in range(natom):
        x, y, z = r[a]
        D[3 * a : 3 * a + 3, 3] = [0, -z, y]  # Rx
        D[3 * a : 3 * a + 3, 4] = [z, 0, -x]  # Ry
        D[3 * a : 3 * a + 3, 5] = [-y, x, 0]  # Rz
    D = D * sm[:, None]  # mass-weight
    # orthonormalize, dropping null columns (linear molecules -> 5 modes)
    q, rdiag = np.linalg.qr(D)
    keep = np.abs(np.diag(rdiag)) > 1e-8
    return q[:, keep]


def frequencies(syms, xyz_ang, grad_fn, delta_bohr=0.01):
    H = hessian(syms, xyz_ang, grad_fn, delta_bohr)
    masses = np.array([MASS[s] for s in syms])
    sm = np.sqrt(np.repeat(masses, 3))
    Hmw = H / np.outer(sm, sm)  # mass-weighted, units Eh/bohr^2/amu
    P = projector(syms, xyz_ang)
    proj = np.eye(Hmw.shape[0]) - P @ P.T
    Hint = proj @ Hmw @ proj
    evals = np.linalg.eigvalsh(Hint)
    # drop the modes we projected out (eigenvalues ~0)
    nrot = P.shape[1]
    evals = evals[nrot:]
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * HARTREE_BOHR2_AMU_TO_CM1
    return np.sort(freqs)


def describe(label, syms, xyz_ang, grad_fn):
    e, _ = grad_fn(syms, xyz_ang)
    freqs = frequencies(syms, xyz_ang, grad_fn)
    imag = freqs[freqs < -1.0]  # treat |nu|<1 cm^-1 as numerical zero
    print(f'\n=== {label} ===')
    print(f'  energy           : {e:.8f} Ha')
    print('  lowest 8 freqs   : ' + ' '.join(f'{f:8.1f}' for f in freqs[:8]))
    if imag.size:
        print(
            f'  IMAGINARY modes  : {imag.size} ->', ' '.join(f'{f:.1f}i' for f in -imag)
        )
        print('  VERDICT: SADDLE POINT (not a minimum)')
    else:
        print('  IMAGINARY modes  : none')
        print('  VERDICT: genuine minimum (all-real frequencies)')
    return e, freqs


def planarity(syms, xyz_ang):
    """RMS out-of-plane deviation (Angstrom) from the best-fit plane of C atoms."""
    mask = np.array([s == 'C' for s in syms])
    pts = xyz_ang[mask]
    centroid = pts.mean(0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    d = (xyz_ang - centroid) @ normal
    return float(np.sqrt((d**2).mean())), float(np.abs(d).max())


def main():
    here = Path(__file__).parent
    ref = here / 'benzene_reference_min.xyz'
    pseudo = here / 'benzene_pseudo_min.xyz'

    print(
        'GFN2-xTB vibrational analysis (finite-difference Hessian, '
        'translations/rotations projected out)'
    )
    print('\n########## GFN2-xTB ##########')
    for label, path in [
        ('benzene reference (planar D6h)', ref),
        ('benzene pseudo-minimum (noise-found)', pseudo),
    ]:
        syms, xyz, comment = read_xyz(path)
        rms_oop, max_oop = planarity(syms, xyz)
        print(f'\n[{path.name}] {comment}')
        print(f'  C-frame out-of-plane RMS={rms_oop:.4f} A, max={max_oop:.4f} A')
        describe(label, syms, xyz, gfn2_gradient)


if __name__ == '__main__':
    sys.exit(main())
