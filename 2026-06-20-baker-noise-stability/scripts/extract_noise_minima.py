#!/usr/bin/env python3
"""Extract reference and noise-found minima as structures for follow-up issues.

For the Baker molecules where start-geometry noise relaxes into a *lower*
minimum than the unperturbed run (see scripts/baker_noise_stability_findings.md),
this writes three geometries per molecule into ``investigations/noise_minima/``:
the Baker starting geometry (``*_initial.xyz``), the minimum the no-noise run
reaches (``*_reference_min.xyz``), and the lower minimum a noisy start reaches
(``*_lower_min.xyz``).

For benzene it additionally writes the higher-energy ``benzene_pseudo_min.xyz``
structure a noisy start converges to -- the "minimum" whose status is the
subject of a follow-up issue.

This only *produces* the structures (and a ``manifest.md`` of their energies);
the analysis of why the lower minima exist, and whether the benzene pseudo
minimum is a genuine minimum, is left to the issues that reference these files.

Run from the repo root.
"""

from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules, load_reference
from berny.solvers import XTBSolver

H2KCAL = 627.5094740631

OUT = Path('investigations/noise_minima')

# Molecules whose noisy starts reach a minimum below the no-noise reference,
# with the approximate drop (kcal/mol) seen in the sweep / path analysis.
LOWER_CASES = {
    'methylamine': -6.24,
    'ethanol': -1.55,
    'histidine': -1.98,
    'mesityl_oxide': -1.30,
    'benzidine': -1.28,
    'acanil01': -0.75,
    'caffeine': -0.42,
}

SCHEDULE = [(0.05, 8), (0.1, 8), (0.2, 12), (0.3, 16)]


def optimize(geom, ref, maxsteps=200):
    """Return (converged, energy, final_geom, max_grad_component[a.u.])."""
    berny = Berny(geom, maxsteps=maxsteps)
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    energy = None
    grad = None
    last = geom
    for g in berny:
        energy, grad = solver.send((list(g), g.lattice))
        berny.send((energy, grad))
        last = g
    gmax = float(np.abs(grad).max()) if grad is not None else float('nan')
    return berny.converged, energy, last, gmax


def perturb(geom, sigma, seed):
    rng = np.random.default_rng((int(sigma * 1000), seed))
    return geomlib.Geometry(
        list(geom.species),
        geom.coords + rng.normal(0.0, sigma, size=geom.coords.shape),
        geom.lattice,
    )


def find_noise_minimum(geom, ref, e_clean, accept):
    """Sweep noisy starts; return the first converged ``(e, geom, gmax)`` for
    which ``accept(e - e_clean)`` (energy difference in Hartree) holds, else the
    best (lowest-energy) converged candidate seen, or ``None``."""
    best = None
    for sigma, n_seed in SCHEDULE:
        for seed in range(n_seed):
            try:
                conv, e, g, gmax = optimize(perturb(geom, sigma, seed), ref)
            except Exception:
                continue
            if not conv:
                continue
            if best is None or e < best[0]:
                best = (e, g, gmax)
            if accept(e - e_clean):
                return (e, g, gmax)
    return best


def write_xyz(path, species, coords, comment):
    lines = [str(len(species)), comment]
    for sp, (x, y, z) in zip(species, coords):
        lines.append(f'{sp:>2} {x:18.10f} {y:18.10f} {z:18.10f}')
    path.write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = load_reference('baker')
    names = sorted(set(LOWER_CASES) | {'benzene'})
    geoms = {name: (geom, ref) for name, geom, ref in iter_molecules('baker', names)}
    manifest = ['# Noise-found minima for follow-up issues', '']

    for name in names:
        geom, ref = geoms[name]
        print(f'==> {name}', flush=True)
        _conv0, e0, g0, gmax0 = optimize(geom, ref)
        write_xyz(
            OUT / f'{name}_initial.xyz',
            geom.species,
            geom.coords,
            f'{name}: Baker reference start geometry',
        )
        write_xyz(
            OUT / f'{name}_reference_min.xyz',
            g0.species,
            g0.coords,
            f'{name}: no-noise minimum E={e0:.8f} Ha gmax={gmax0:.2e}',
        )

        if name == 'benzene':
            # A converged structure well above the planar reference minimum.
            found = find_noise_minimum(geom, ref, e0, accept=lambda de: de * H2KCAL > 5)
            e1, g1, gmax1 = found
            write_xyz(
                OUT / f'{name}_pseudo_min.xyz',
                g1.species,
                g1.coords,
                f'{name}: noise-found "minimum" E={e1:.8f} Ha '
                f'(+{(e1 - e0) * H2KCAL:.1f} kcal/mol) gmax={gmax1:.2e}',
            )
            manifest += [
                '## benzene',
                f'- reference (planar) min: E={e0:.8f} Ha, gmax={gmax0:.2e}',
                f'- pseudo "min": E={e1:.8f} Ha (+{(e1 - e0) * H2KCAL:.1f} kcal/mol), '
                f'gmax={gmax1:.2e}',
                '',
            ]
            print(f'    benzene pseudo min +{(e1 - e0) * H2KCAL:.1f} kcal/mol')
            continue

        found = find_noise_minimum(geom, ref, e0, accept=lambda de: de * H2KCAL < -0.15)
        if found is None or (found[0] - e0) * H2KCAL >= -0.15:
            print(f'    WARNING: no lower minimum recovered for {name}')
            manifest += [f'## {name}', '- WARNING: no lower minimum recovered', '']
            continue
        e1, g1, gmax1 = found
        write_xyz(
            OUT / f'{name}_lower_min.xyz',
            g1.species,
            g1.coords,
            f'{name}: noise-found lower minimum E={e1:.8f} Ha '
            f'({(e1 - e0) * H2KCAL:+.2f} kcal/mol) gmax={gmax1:.2e}',
        )
        manifest += [
            f'## {name} ({reference[name]["atoms"]} atoms)',
            f'- reference min: E={e0:.8f} Ha, gmax={gmax0:.2e}',
            f'- lower min: E={e1:.8f} Ha ({(e1 - e0) * H2KCAL:+.2f} kcal/mol), '
            f'gmax={gmax1:.2e}',
            '',
        ]
        print(f'    {name}: lower min {(e1 - e0) * H2KCAL:+.2f} kcal/mol')

    (OUT / 'manifest.md').write_text('\n'.join(manifest) + '\n')
    print(f'\nwrote structures + manifest to {OUT}')


if __name__ == '__main__':
    main()
