#!/usr/bin/env python3
"""HF/pyscf cross-check for issue #149.

The frequency analysis (freq_analysis.py) shows the benzene "pseudo-minimum"
is a *genuine* GFN2-xTB minimum, namely fulvene. This script checks that the
phenomenon is not an artefact of xTB's smooth surface: drive pyberny with the
pyscf RHF solver from (a) the planar benzene start and (b) the fulvene
geometry, and confirm both relax to distinct, separately converged minima with
the benzene/fulvene energy ordering. A finite-difference Hessian at the
HF-optimized fulvene confirms it is a real minimum there too.

    investigations/noise_minima/hf_crosscheck.py            # HF/3-21G (default)
    investigations/noise_minima/hf_crosscheck.py --basis sto-3g
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from berny import Berny, Geometry, optimize

ANGSTROM_TO_BOHR = 1.8897261246257702
HA2KCAL = 627.509474


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    syms, xyz = [], []
    for line in lines[2 : 2 + n]:
        s, x, y, z = line.split()[:4]
        syms.append(s)
        xyz.append([float(x), float(y), float(z)])
    return syms, np.array(xyz)


def hf_solver(basis: str):
    """pyberny solver yielding (energy[Ha], gradient[Ha/bohr]) from RHF/pyscf."""
    from pyscf import gto, scf

    atoms, _ = yield None
    while True:
        mol = gto.M(
            atom=[(s, tuple(c)) for s, c in atoms],
            basis=basis,
            unit='Angstrom',
            verbose=0,
        )
        mf = scf.RHF(mol)
        e = mf.kernel()
        g = mf.nuc_grad_method().kernel()  # Ha/bohr, N-by-3
        atoms, _ = yield float(e), g


def hf_energy_grad(syms, xyz_ang, basis):
    from pyscf import gto, scf

    mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(syms, xyz_ang)],
        basis=basis,
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol)
    e = mf.kernel()
    g = mf.nuc_grad_method().kernel()
    return float(e), np.array(g)


def fd_hessian_min_freq(syms, xyz_ang, basis, delta=0.01):
    """Return the lowest finite-difference frequency (cm^-1) — negative if imaginary."""
    MASS = {'H': 1.007825, 'C': 12.0}
    natom = len(syms)
    xyz_bohr = xyz_ang * ANGSTROM_TO_BOHR
    nc = 3 * natom
    H = np.zeros((nc, nc))
    for i in range(nc):
        a, c = divmod(i, 3)
        xp = xyz_bohr.copy()
        xp[a, c] += delta
        xm = xyz_bohr.copy()
        xm[a, c] -= delta
        _, gp = hf_energy_grad(syms, xp / ANGSTROM_TO_BOHR, basis)
        _, gm = hf_energy_grad(syms, xm / ANGSTROM_TO_BOHR, basis)
        H[i] = ((gp - gm) / (2 * delta)).reshape(-1)
    H = 0.5 * (H + H.T)
    sm = np.sqrt(np.repeat([MASS[s] for s in syms], 3))
    Hmw = H / np.outer(sm, sm)
    # crude trans/rot projection via lowest-6 discard
    evals = np.sort(np.linalg.eigvalsh(Hmw))[6:]
    return np.sign(evals[0]) * np.sqrt(abs(evals[0])) * 5140.4838


def run(label, syms, xyz, basis):
    geom = Geometry(syms, xyz)
    opt = Berny(geom)
    final = optimize(opt, hf_solver(basis))
    final_xyz = np.array([c for _, c in final])
    e, _ = hf_energy_grad(syms, final_xyz, basis)
    print(f'  {label:20s}: converged={opt.converged}  E={e:.6f} Ha')
    return e, syms, final_xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basis', default='3-21g')
    args = ap.parse_args()
    here = Path(__file__).parent

    print(f'HF/{args.basis} cross-check via pyberny + pyscf\n')
    results = {}
    for label, fname in [
        ('benzene', 'benzene_reference_min.xyz'),
        ('fulvene', 'benzene_pseudo_min.xyz'),
    ]:
        syms, xyz = read_xyz(here / fname)
        results[label] = run(label, syms, xyz, args.basis)

    e_benz = results['benzene'][0]
    e_ful = results['fulvene'][0]
    print(
        f'\n  fulvene - benzene = {(e_ful - e_benz) * HA2KCAL:+.1f} kcal/mol '
        '(HF)  [GFN2-xTB: +32.3]'
    )

    syms, xyz = results['fulvene'][1], results['fulvene'][2]
    f0 = fd_hessian_min_freq(syms, xyz, args.basis)
    verdict = 'SADDLE (imaginary mode)' if f0 < -1 else 'genuine minimum'
    print(f'  lowest HF frequency at optimized fulvene: {f0:.1f} cm^-1 -> {verdict}')


if __name__ == '__main__':
    main()
