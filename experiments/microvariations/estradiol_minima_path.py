#!/usr/bin/env python3
"""Interpolate between estradiol's distinct PM7 minima and plot the energy path.

The microvariation experiment uncovered 5 energy-distinct minima of estradiol on
the PM7 PES (clustered at 0.5 kcal/mol tolerance). This script:

1. Picks one representative seed per basin from experiments/microvariations/.
2. Kabsch-aligns all five geometries to the deepest one.
3. For each consecutive pair (sorted by energy), interpolates the Cartesians
   linearly in N steps and runs a single-point MOPAC PM7 calculation at each.
4. Plots the resulting energy trajectory.

The linear path is generally NOT the minimum-energy path between minima - it's
a straight line in 3N-dimensional configuration space, so the barriers it
crosses are upper bounds on the true barriers. What this picture is good for:
seeing that the minima really are separate (peaks between them) versus a flat
trough (no peaks), and getting a rough sense of how high the barriers are.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from berny import geomlib  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / 'src' / 'berny' / 'benchmarks' / 'birkholz_schlegel'
OUT_DIR = REPO_ROOT / 'experiments' / 'microvariations'
KCAL_PER_HA = 627.503

# Five basin representatives identified by 0.5-kcal/mol clustering of the
# default-tolerance microvariation run.
BASINS = [
    {
        'label': 'basin 0 (deepest)',
        'sigma': 0.05,
        'seed': 6,
        'E_expected': -0.158928,
    },
    {'label': 'basin 1', 'sigma': 0.05, 'seed': 2, 'E_expected': -0.158116},
    {'label': 'basin 2', 'sigma': 0.05, 'seed': 5, 'E_expected': -0.155528},
    {'label': 'basin 3', 'sigma': 0.01, 'seed': 8, 'E_expected': -0.153738},
    {
        'label': 'basin 4 (published start)',
        'sigma': 0.01,
        'seed': 0,
        'E_expected': -0.151174,
    },
]
POINTS_PER_SEGMENT = 25


def mopac_single_point(species, coords, cmd='mopac', method='PM7'):
    """Run a single-point MOPAC PM7 calculation; return energy in Hartree."""
    tmpdir = tempfile.mkdtemp()
    try:
        keyword_line = f'{method} 1SCF'
        body = '\n'.join(
            f'{el} {x} 1 {y} 1 {z} 1' for el, (x, y, z) in zip(species, coords)
        )
        with open(os.path.join(tmpdir, 'job.mop'), 'w') as f:
            f.write(f'{keyword_line}\n\n\n{body}\n')
        subprocess.check_call(
            [cmd, os.path.join(tmpdir, 'job.mop')],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(os.path.join(tmpdir, 'job.out')) as f:
            for line in f:
                if 'FINAL HEAT OF FORMATION' in line:
                    return float(line.split()[5]) / KCAL_PER_HA
        raise RuntimeError('no energy found')
    finally:
        shutil.rmtree(tmpdir)


def kabsch_align(target, mobile):
    """Rotate+translate ``mobile`` to best overlap ``target``."""
    t_c = target.mean(axis=0)
    m_c = mobile.mean(axis=0)
    h = (target - t_c).T @ (mobile - m_c)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (mobile - m_c) @ rot.T + t_c


def main():
    final_coords = json.load(open(OUT_DIR / 'final_coords.json'))
    g0 = geomlib.readfile(str(DATA / 'estradiol.xyz'))
    species = g0.species

    # Load and align all basin geometries to the deepest one.
    raw = [
        np.array(final_coords[f'estradiol|{b["sigma"]}|{b["seed"]}']) for b in BASINS
    ]
    aligned = [raw[0]]
    for c in raw[1:]:
        aligned.append(kabsch_align(raw[0], c))

    # Interpolate consecutive pairs and run single points.
    xs = []  # path coordinate
    es = []  # energies (Ha)
    basin_marks = [0]  # x-positions where each basin sits
    x_running = 0.0
    for i in range(len(BASINS) - 1):
        a = aligned[i]
        b = aligned[i + 1]
        print(f'\nSegment {i}: {BASINS[i]["label"]} -> {BASINS[i + 1]["label"]}')
        for j, t in enumerate(np.linspace(0, 1, POINTS_PER_SEGMENT)):
            coords = (1 - t) * a + t * b
            energy = mopac_single_point(species, coords)
            x = x_running + t
            xs.append(x)
            es.append(energy)
            if j % 5 == 0:
                kcal = energy * KCAL_PER_HA
                print(f'  t={t:.3f}  E={energy:.6f} Ha  ({kcal:.3f} kcal/mol)')
        x_running += 1.0
        basin_marks.append(x_running)

    xs = np.array(xs)
    es = np.array(es)
    e_ref = es.min()

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, (es - e_ref) * KCAL_PER_HA, '-', color='#1f77b4', lw=1.5)
    ax.scatter(
        xs,
        (es - e_ref) * KCAL_PER_HA,
        s=12,
        color='#1f77b4',
        zorder=3,
    )
    for x, b in zip(basin_marks, BASINS):
        ax.axvline(x, color='gray', lw=0.5, ls='--', alpha=0.5)
        e_basin = (b['E_expected'] - e_ref) * KCAL_PER_HA
        ax.scatter(
            [x],
            [e_basin],
            s=80,
            marker='o',
            edgecolor='black',
            facecolor='#ff7f0e',
            zorder=4,
        )
        ax.annotate(
            b['label'].replace(' (', '\n('),
            (x, e_basin),
            textcoords='offset points',
            xytext=(0, -25),
            ha='center',
            fontsize=8,
        )
    ax.set_xlabel('linear-interpolation path coordinate')
    ax.set_ylabel('energy above deepest minimum (kcal/mol)')
    ax.set_title('Estradiol PM7 PES: linear-interpolation path through 5 minima')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / 'minima_interpolation.png'
    fig.savefig(out_path, dpi=140)
    print(f'\nWrote {out_path}')

    # Also write the raw data.
    data = {
        'basins': BASINS,
        'points_per_segment': POINTS_PER_SEGMENT,
        'path_coordinate': xs.tolist(),
        'energy_Ha': es.tolist(),
        'basin_marks': basin_marks,
    }
    (OUT_DIR / 'minima_interpolation.json').write_text(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
