#!/usr/bin/env python3
"""Cross-check the ethanol/histidine conformer gap at GFN2-xTB and HF/6-31G**.

For each molecule we optimize from three starts -- the Baker start, the
no-noise reference minimum, and the noise-found lower minimum -- at GFN2-xTB,
and from the reference and lower minima at HF/6-31G**. The question (#154 Q3)
is whether the *lower* conformer stays below the no-noise reference at the
paper's reference method, i.e. whether the gap is real conformer ordering and
not a GFN2 artefact.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np

from berny import Berny, geomlib
from berny.solvers import XTBSolver

HARTREE2KCAL = 627.50947


def run_xtb(path):
    geom = geomlib.readfile(path)
    berny = Berny(geom)
    solver = XTBSolver(charge=0, mult=1)
    next(solver)
    energy = None
    gmax = None
    for g in berny:
        energy, gradients = solver.send((list(g), g.lattice))
        gmax = np.abs(np.array(gradients)).max()
        berny.send((energy, gradients))
    return berny.converged, berny._n, energy, gmax


def run_hf(path):
    from pyscf import gto, scf
    from pyscf.geomopt import berny_solver

    mol = gto.M(atom=path, basis='6-31G**', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    state = {'energies': []}

    def cb(loc):
        e = loc.get('energy')
        if e is not None:
            state['energies'].append(float(e))

    converged, mol_eq = berny_solver.kernel(mf, callback=cb)
    e_final = state['energies'][-1] if state['energies'] else None
    return converged, len(state['energies']), e_final, mol_eq


def main():
    cases = {
        'ethanol': {
            'initial': 'ethanol_initial.xyz',
            'reference_min': 'ethanol_reference_min.xyz',
            'lower_min': 'ethanol_lower_min.xyz',
        },
        'histidine': {
            'initial': 'histidine_initial.xyz',
            'reference_min': 'histidine_reference_min.xyz',
            'lower_min': 'histidine_lower_min.xyz',
        },
    }
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    for mol, starts in cases.items():
        print(f'\n========== {mol} ==========')
        # --- GFN2-xTB ---
        print('--- GFN2-xTB (reoptimize from each start) ---')
        xtb = {}
        for label, fn in starts.items():
            conv, n, e, gmax = run_xtb(os.path.join(here, fn))
            xtb[label] = e
            print(
                f'  {label:14s} conv={conv} steps={n:3d} '
                f'E={e:.8f} Ha gmax={gmax:.2e}'
            )
        if 'reference_min' in xtb and 'lower_min' in xtb:
            d = (xtb['lower_min'] - xtb['reference_min']) * HARTREE2KCAL
            print(f'  GFN2 dE(lower - reference) = {d:+.2f} kcal/mol')

        # --- HF/6-31G** ---
        print('--- HF/6-31G** (reoptimize from reference_min and lower_min) ---')
        hf = {}
        for label in ('reference_min', 'lower_min'):
            conv, n, e, _ = run_hf(os.path.join(here, starts[label]))
            hf[label] = e
            print(f'  {label:14s} conv={conv} steps={n:3d} E={e:.8f} Ha')
        if 'reference_min' in hf and 'lower_min' in hf:
            d = (hf['lower_min'] - hf['reference_min']) * HARTREE2KCAL
            print(f'  HF   dE(lower - reference) = {d:+.2f} kcal/mol')


if __name__ == '__main__':
    main()
