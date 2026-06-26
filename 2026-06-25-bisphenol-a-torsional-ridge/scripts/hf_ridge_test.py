#!/usr/bin/env python3
"""Is the *graded* HF/3-21G step count for bisphenol_a ridge-inflated too?

Runs the benchmark's exact PySCF path (``pyscf.geomopt.berny_solver``,
HF/3-21G -- the method/basis behind ``reference.json``'s ``pyberny_steps``)
on the clean start and on small symmetry-breaking perturbations, recording
step counts and final energies. Contrast with ``scatter.py`` (GFN2-xTB): if
the HF count barely moves under perturbation while the xTB count roughly
halves, the dramatic ridge inflation is specific to the semi-empirical surface
(which Birkholz & Schlegel predicted would be "less regular").

    OMP_NUM_THREADS=4 python hf_ridge_test.py --out-dir ../data

One HF/3-21G energy+gradient on this 33-atom molecule is ~10-40 s, and the
optimization takes ~50 steps, so each run is ~30-40 min; budget accordingly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from pyscf.geomopt import berny_solver

HARTREE_KCAL = 627.5094740631
XYZ = ('../../../src/berny/benchmarks/birkholz_schlegel/bisphenol_a.xyz')


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    sp, co = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sp.append(p[0]); co.append([float(x) for x in p[1:4]])
    return sp, np.array(co)


def write_xyz(path, sp, co):
    with open(path, 'w') as f:
        f.write(f'{len(sp)}\nperturbed\n')
        for s, (x, y, z) in zip(sp, co):
            f.write(f'{s} {x:18.10f} {y:18.10f} {z:18.10f}\n')


def run(xyz_path, trace_path):
    mol = gto.M(atom=str(xyz_path), basis='3-21G', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    state = {'n': 0, 'energies': []}

    def cb(loc):
        state['n'] = loc['cycle'] + 1
        if loc.get('energy') is not None:
            state['energies'].append(float(loc['energy']))

    converged, _ = berny_solver.kernel(mf, callback=cb, trace=str(trace_path))
    return bool(converged), state['n'], state['energies']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=Path('../data'))
    ap.add_argument('--xyz', default=XYZ)
    ap.add_argument('--perturbations', type=str, default='0.02:1,0.02:2,0.05:3',
                    help='comma list of sigma:seed')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp, co = read_xyz(args.xyz)

    results = {}
    conv, n, en = run(args.xyz, args.out_dir / 'hf_trace_clean.json')
    e_clean = en[-1]
    results['clean'] = {'converged': conv, 'steps': n, 'E': e_clean}
    print(f'clean: conv={conv} steps={n} E={e_clean:.6f}', flush=True)

    for spec in args.perturbations.split(','):
        sigma, seed = float(spec.split(':')[0]), int(spec.split(':')[1])
        rng = np.random.default_rng(seed)
        p = args.out_dir / f'_pert_{sigma}_{seed}.xyz'
        write_xyz(p, sp, co + rng.normal(0.0, sigma, size=co.shape))
        conv, n, en = run(p, args.out_dir / f'hf_trace_n{sigma}_{seed}.json')
        de = (en[-1] - e_clean) * HARTREE_KCAL
        results[f'sigma{sigma}_seed{seed}'] = {'converged': conv, 'steps': n,
                                               'E': en[-1], 'dE_kcal': de}
        print(f'sigma={sigma} seed={seed}: conv={conv} steps={n} dE={de:+.3f} kcal',
              flush=True)

    (args.out_dir / 'hf_results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
