#!/usr/bin/env python3
"""Interpolated energy paths through the multiple minima a Baker molecule finds
under start-geometry noise.

The companion sweep (``scripts/noise_stability.py``) shows that for some Baker
molecules noisy starts relax into a *different* minimum than the unperturbed
reference -- sometimes lower, sometimes (at large noise) a broken higher-energy
structure. This script, for every such molecule:

1. **Rediscovers the distinct minima with their geometries.** It optimizes the
   clean start and a sweep of noisy starts (GFN2-xTB), and clusters the
   converged structures by Kabsch-aligned RMSD (not just energy) so that
   ordinary convergence scatter is not mistaken for a separate basin.
2. **Connects them with a single piecewise-linear path.** The distinct minima
   are ordered by energy and aligned head-to-tail (each onto the previous, to
   remove rigid-body rotation/translation); consecutive minima are joined by
   linear interpolation of the Cartesian coordinates.
3. **Evaluates the energy along the path** with single-point GFN2-xTB and plots
   it against a cumulative-RMSD path coordinate, with the minima marked.

The result is one energy-vs-path panel per molecule, gathered into a single
figure. Linear Cartesian interpolation is a crude connector (it is not a
minimum-energy path), so the inter-minimum maxima are upper bounds on the real
barriers; the point is to *visualize* that the located structures are genuinely
distinct basins separated by energy rises, and how large those rises are.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules
from berny.solvers import XTBSolver

H2KCAL = 627.5094740631


def optimize(geom, ref, maxsteps=150):
    """Optimize ``geom`` with GFN2-xTB; return (converged, energy, final_geom)."""
    berny = Berny(geom, maxsteps=maxsteps)
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    energy = None
    last = geom
    for g in berny:
        energy, grad = solver.send((list(g), g.lattice))
        berny.send((energy, grad))
        last = g
    return berny.converged, energy, last


def singlepoint(species, coords, ref):
    solver = XTBSolver(charge=ref['charge'], mult=ref['mult'])
    next(solver)
    e, _ = solver.send((list(zip(species, coords)), None))
    return e


def kabsch(mobile, ref):
    """Rigid-body align ``mobile`` onto ``ref`` (both ``(N, 3)``); return the
    aligned coordinates and the resulting RMSD."""
    mc = mobile.mean(0)
    rc = ref.mean(0)
    m = mobile - mc
    r = ref - rc
    h = m.T @ r
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = m @ rot.T + rc
    rmsd = float(np.sqrt(((aligned - ref) ** 2).sum(1).mean()))
    return aligned, rmsd


def rmsd_to(coords, ref):
    return kabsch(coords, ref)[1]


def target_minima_counts(json_path, tol_kcal=0.3):
    """Per-molecule count of distinct converged energy levels in the recorded
    sweep (clustered at ``tol_kcal``); used to know how many basins to chase."""
    data = json.loads(Path(json_path).read_text())
    out = {}
    for mol, trials in data['raw'].items():
        es = sorted(
            t['energy']
            for t in trials
            if t['status'] == 'converged' and t['energy'] is not None
        )
        clusters = 0
        prev = None
        for e in es:
            if prev is None or (e - prev) * H2KCAL > tol_kcal:
                clusters += 1
            prev = e
        out[mol] = clusters
    return out


def discover_minima(name, geom, ref, expected_n, rmsd_tol, schedule):
    """Discover structurally distinct minima for ``name``.

    Returns ``(minima, clean_energy)`` where ``minima`` is a list of distinct
    ``(energy, geom)`` sorted by energy and ``clean_energy`` is the energy of
    the unperturbed (no-noise) optimization (or ``None`` if it didn't
    converge).

    Seeds the clean optimization, then sweeps noisy starts per ``schedule`` (a
    list of ``(sigma, n_seed)``), adding any converged structure whose aligned
    RMSD to every known minimum exceeds ``rmsd_tol``. Stops once ``expected_n``
    distinct minima are in hand."""
    minima = []
    clean_energy = None
    conv, e, g = optimize(geom, ref)
    if conv:
        clean_energy = e
        minima.append((e, g))
    for sigma, n_seed in schedule:
        if len(minima) >= expected_n:
            break
        for seed in range(n_seed):
            if len(minima) >= expected_n:
                break
            rng = np.random.default_rng((hash(name) & 0xFFFF, int(sigma * 1000), seed))
            noisy = geomlib.Geometry(
                list(geom.species),
                geom.coords + rng.normal(0.0, sigma, size=geom.coords.shape),
                geom.lattice,
            )
            try:
                conv, e, g = optimize(noisy, ref)
            except Exception:
                continue
            if not conv:
                continue
            if all(rmsd_to(g.coords, m[1].coords) > rmsd_tol for m in minima):
                minima.append((e, g))
    minima.sort(key=lambda m: m[0])
    return minima, clean_energy


def build_path(minima, per_segment):
    """Order-by-energy minima -> aligned piecewise-linear path.

    Returns ``(images, node_idx, energies_placeholder)`` where ``images`` is a
    list of ``(N, 3)`` coordinate arrays and ``node_idx`` the indices of the
    minima within ``images``."""
    coords = [m[1].coords for m in minima]
    aligned = [coords[0]]
    for k in range(1, len(coords)):
        a, _ = kabsch(coords[k], aligned[k - 1])
        aligned.append(a)
    images = [aligned[0]]
    node_idx = [0]
    for k in range(len(aligned) - 1):
        a, b = aligned[k], aligned[k + 1]
        for j in range(1, per_segment + 1):
            t = j / per_segment
            images.append((1 - t) * a + t * b)
        node_idx.append(len(images) - 1)
    return images, node_idx


def path_coordinate(images):
    x = [0.0]
    for i in range(1, len(images)):
        x.append(
            x[-1] + float(np.sqrt(((images[i] - images[i - 1]) ** 2).sum(1).mean()))
        )
    return np.array(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--benchmark', default='baker')
    ap.add_argument(
        '--json',
        type=Path,
        default=Path('artifacts/baker_noise_stability.json'),
        help='sweep output used to decide how many basins to chase per molecule',
    )
    ap.add_argument('--rmsd-tol', type=float, default=0.1, help='Angstrom')
    ap.add_argument('--per-segment', type=int, default=16)
    ap.add_argument('--molecules', nargs='*', default=None)
    ap.add_argument('--out-fig', type=Path, default=Path('artifacts/minima_paths.png'))
    ap.add_argument(
        '--out-json', type=Path, default=Path('artifacts/minima_paths.json')
    )
    args = ap.parse_args(argv)

    counts = target_minima_counts(args.json)
    candidates = args.molecules or sorted(m for m, n in counts.items() if n >= 2)

    schedule = [(0.05, 4), (0.1, 4), (0.2, 6), (0.3, 14)]
    results = {}
    for name, geom, ref in iter_molecules(args.benchmark, candidates):
        expected = counts.get(name, 2)
        print(f'==> {name} (expect up to {expected} minima)', flush=True)
        minima, clean_energy = discover_minima(
            name, geom, ref, expected, args.rmsd_tol, schedule
        )
        if len(minima) < 2:
            print(f'    only {len(minima)} distinct minimum found; skipping')
            continue
        images, node_idx = build_path(minima, args.per_segment)
        energies = np.array([singlepoint(geom.species, im, ref) for im in images])
        x = path_coordinate(images)
        results[name] = {
            'atoms': ref['atoms'],
            'n_minima': len(minima),
            'minimum_energies': [float(m[0]) for m in minima],
            'clean_energy': None if clean_energy is None else float(clean_energy),
            'node_idx': node_idx,
            'x': x.tolist(),
            'energy': energies.tolist(),
        }
        rel = (np.array([m[0] for m in minima]) - minima[0][0]) * H2KCAL
        print(f'    {len(minima)} minima; rel E (kcal/mol) = {np.round(rel, 2)}')

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))
    plot(results, args.out_fig)
    print(f'\nwrote {args.out_fig} and {args.out_json} ({len(results)} molecules)')
    return 0


def plot(results, out_fig):
    import math

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Log y-axis spans the >4 orders of magnitude between the sub-kcal/mol
    # conformer barriers and the >100 kcal/mol broken-aromatic ones. Energies
    # are measured from the lowest located minimum, so that minimum sits at
    # zero; floor it (and any other exactly-degenerate node) to a small
    # positive value so it still renders on a log axis.
    floor = 1e-2  # kcal/mol

    names = sorted(results, key=lambda n: -results[n]['n_minima'])
    n = len(names)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    clean_handle = None
    for ax, name in zip(axes, names):
        r = results[name]
        x = np.array(r['x'])
        e = np.array(r['energy'])
        nodes = np.array(r['node_idx'])
        node_e = e[nodes]
        e0 = node_e.min()
        y = np.clip((e - e0) * H2KCAL, floor, None)
        ax.plot(x, y, '-', color='tab:blue', lw=1.5)
        ax.plot(x[nodes], y[nodes], 'o', color='tab:red', ms=6, zorder=5)
        # Highlight the minimum reached by the unperturbed (no-noise) run.
        clean_energy = r.get('clean_energy')
        clean_k = None
        if clean_energy is not None:
            clean_k = int(np.argmin(np.abs(node_e - clean_energy)))
        for k, (xi, yi) in enumerate(zip(x[nodes], y[nodes])):
            if k == clean_k:
                h = ax.plot(
                    xi,
                    yi,
                    '*',
                    color='gold',
                    markeredgecolor='black',
                    markeredgewidth=0.6,
                    ms=15,
                    zorder=6,
                )[0]
                clean_handle = clean_handle or h
            ax.annotate(
                f'{yi:.2f}' if yi < 1 else f'{yi:.1f}',
                (xi, yi),
                textcoords='offset points',
                xytext=(0, 7),
                ha='center',
                fontsize=7,
                color='black' if k == clean_k else 'tab:red',
            )
        ax.set_yscale('log')
        ax.set_ylim(bottom=floor)
        ax.set_title(f"{name}  ({r['n_minima']} minima, {r['atoms']} at.)", fontsize=9)
        ax.set_xlabel('path coord. (cum. RMSD, A)', fontsize=8)
        ax.set_ylabel('E - E_min (kcal/mol, log)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, which='both')
    for ax in axes[n:]:
        ax.set_visible(False)
    if clean_handle is not None:
        fig.legend(
            [clean_handle],
            ['no-noise (reference) minimum'],
            loc='upper right',
            fontsize=9,
        )
    fig.suptitle(
        'Linearly interpolated energy paths through Baker minima found under '
        'start-geometry noise (GFN2-xTB)',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=130)


if __name__ == '__main__':
    raise SystemExit(main())
