#!/usr/bin/env python3
"""Re-optimize ethanol/histidine at HF/6-31G** from both starts, save the
optimized geometries, and report the ethanol H-O-C-C torsion so we can confirm
the anti/gauche pair stays distinct (the ordering genuinely flips, rather than
gauche collapsing onto anti)."""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
from pyscf import gto, scf
from pyscf.geomopt import berny_solver

HARTREE2KCAL = 627.50947
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def opt(path):
    mol = gto.M(atom=path, basis='6-31G**', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    conv, mol_eq = berny_solver.kernel(mf)
    e = scf.RHF(mol_eq).run(verbose=0).e_tot
    return e, mol_eq


def save(mol_eq, comment, out):
    sym = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    xyz = mol_eq.atom_coords(unit='Angstrom')
    with open(out, 'w') as f:
        f.write(f'{len(sym)}\n{comment}\n')
        for s, c in zip(sym, xyz):
            f.write(f'{s:2s} {c[0]:18.10f} {c[1]:18.10f} {c[2]:18.10f}\n')
    return xyz


for mol in ['ethanol', 'histidine']:
    print(f'== {mol} (HF/6-31G**) ==')
    res = {}
    for tag in ['reference_min', 'lower_min']:
        e, meq = opt(os.path.join(here, f'{mol}_{tag}.xyz'))
        xyz = save(meq, f'{mol} HF/6-31G** opt from {tag}: E={e:.8f} Ha',
                   os.path.join(here, f'{mol}_{tag}_hf.xyz'))
        res[tag] = (e, xyz)
        extra = ''
        if mol == 'ethanol':
            extra = f'  HOCC={dih(xyz[3], xyz[0], xyz[1], xyz[2]):.1f} deg'
        print(f'  {tag:14s} E={e:.8f} Ha{extra}')
    d = (res['lower_min'][0] - res['reference_min'][0]) * HARTREE2KCAL
    print(f'  dE(lower - reference) = {d:+.3f} kcal/mol\n')
