#!/usr/bin/env python3
"""Root-cause analysis of the PPE ``CoordinateError`` reported in pyberny#173.

Regenerates every number and the figure used in the report's ``README.md``:

1. **Canonical case** -- finds the first PPE_n5 start perturbed at sigma=0.2 A
   that raises ``CoordinateError``, rebuilds the connectivity exactly as
   ``InternalCoords`` does, and measures the distances/angles at the offending
   centre (showing the offending "bond" is a non-covalent fragment-bridge).
2. **Failure requires fragmentation** -- sweeps the PPE length series over a
   range of noise amplitudes and records, for every trial that raises, the
   number of *covalent* fragments of the perturbed geometry. No raise ever
   happens on an intact (single-fragment) molecule.
3. **Fragmentation / error onset vs amplitude** -- per-sigma rates of
   fragmentation and of ``CoordinateError`` (figure ``fragmentation_onset.png``).
4. **Threshold proximity vs chain length** -- in the *clean* geometries, how
   many backbone angles sit near the 175 deg linear-detection threshold, and
   how close, as a function of PPE length.

Deterministic: all randomness uses ``np.random.default_rng(seed)`` with fixed
seeds. Outputs land next to this script's ``../`` report folder.
"""

import json
from math import pi
from pathlib import Path

import numpy as np

from berny.benchmarks import iter_molecules
from berny.coords import Angle, CoordinateError, InternalCoords, get_clusters
from berny.species_data import get_property

DEG = 180 / pi
LIN_THRE_DEG = 175.0  # berny's 5 deg linear gate (>=175 deg or <=5 deg)
OUT = Path(__file__).resolve().parents[1]
DATA = OUT / 'data'
DATA.mkdir(exist_ok=True)

PPE = [(n, g, r) for n, g, r in iter_molecules('oligomers') if r.get('family') == 'ppe']
PPE_BY_NAME = {n: (g, r) for n, g, r in PPE}


def covalent_bondmatrix(geom):
    """The covalent connectivity InternalCoords starts from (before bridging)."""
    sp = geom.species
    dist = geom.dist(geom)
    radii = np.array([get_property(s, 'covalent_radius') for s in sp])
    bm = dist < 1.3 * (radii[None, :] + radii[:, None])
    np.fill_diagonal(bm, False)
    return bm, dist, radii


def n_fragments(geom):
    bm, _, _ = covalent_bondmatrix(geom)
    return len(get_clusters(bm)[0])


def perturb(geom, sigma, seed):
    rng = np.random.default_rng(seed)
    g = geom.copy()
    g.coords = g.coords + rng.normal(0.0, sigma, g.coords.shape)
    return g


# ---------------------------------------------------------------------------
# 1. Canonical measured case
# ---------------------------------------------------------------------------
def measure_canonical():
    geom, _ = PPE_BY_NAME['PPE_n5']
    sp = geom.species
    err_msg = None
    for seed in range(1000):
        g = perturb(geom, 0.2, seed)
        try:
            InternalCoords(g)
        except CoordinateError as e:
            err_msg = str(e)
            break
    if err_msg is None:
        raise RuntimeError('no CoordinateError found for PPE_n5 at sigma=0.2')

    # rebuild the connectivity InternalCoords actually used: covalent bonds
    # plus the fragment-reconnection bridges (vdW radius + growing shift).
    bm, dist, _radii = covalent_bondmatrix(g)
    clusters_cov = get_clusters(bm)[0]
    vdw = np.array([get_property(s, 'vdw_radius') for s in sp])
    C_total = get_clusters(bm)[1]
    shift = 0.0
    bridges = []
    while not C_total.all():
        new = ~C_total & (dist < vdw[None, :] + vdw[:, None] + shift)
        for i, j in zip(*np.where(np.triu(new & ~bm))):
            bridges.append((int(i), int(j)))
        bm = bm | new
        C_total = get_clusters(bm)[1]
        shift += 1.0

    # parse the centre out of the error message
    msg = err_msg
    import re

    m = re.search(r'center=\[([0-9,\s]+)\].*linear_l=\[([0-9,\s]*)\].*linear_r=\[([0-9,\s]*)\]', msg)
    center = [int(x) for x in m.group(1).split(',') if x.strip()]
    a, b = center[0], center[1]
    nbrs_a = [int(n) for n in np.flatnonzero(bm[a]) if n not in center]

    def cov_cut(i, j):
        ri = get_property(sp[i], 'covalent_radius')
        rj = get_property(sp[j], 'covalent_radius')
        return 1.3 * (ri + rj)

    rec = {
        'molecule': 'PPE_n5',
        'sigma': 0.2,
        'seed': seed,
        'rmsd_A': float(np.sqrt(((g.coords - geom.coords) ** 2).sum() / len(sp))),
        'n_covalent_fragments': len(clusters_cov),
        'error_center': center,
        'bridge_bonds_added': bridges,
        'center_bond': {
            'pair': [f'{sp[a]}{a}', f'{sp[b]}{b}'],
            'distance_A': float(dist[a, b]),
            'covalent_cutoff_A': float(cov_cut(a, b)),
            'covalently_bonded': bool(dist[a, b] < cov_cut(a, b)),
        },
        'left_terminus_neighbours': [],
    }
    for n in nbrs_a:
        ang_noisy = Angle(n, a, b).eval(g.coords) * DEG
        ang_clean = Angle(n, a, b).eval(geom.coords) * DEG
        rec['left_terminus_neighbours'].append({
            'neighbour': f'{sp[n]}{n}',
            'distance_to_center0_A': float(dist[a, n]),
            'covalently_bonded_to_center0': bool(dist[a, n] < cov_cut(a, n)),
            'angle_nbr_c0_c1_deg_noisy': float(ang_noisy),
            'angle_nbr_c0_c1_deg_clean': float(ang_clean),
            'classed_linear': bool(ang_noisy >= LIN_THRE_DEG or ang_noisy <= 5.0),
        })
    return rec


# ---------------------------------------------------------------------------
# 2 + 3. Fragmentation correlation and per-sigma onset
# ---------------------------------------------------------------------------
def sweep(sigmas, n_seeds):
    per_sigma = {}
    frag_hist = {}  # n_fragments -> count of CoordinateError trials
    for sigma in sigmas:
        n_trials = n_err = n_frag = 0
        for name, geom, _ in PPE:
            for seed in range(n_seeds):
                g = perturb(geom, sigma, seed * 100003 + int(sigma * 1000))
                n_trials += 1
                nf = n_fragments(g)
                if nf > 1:
                    n_frag += 1
                try:
                    InternalCoords(g)
                except CoordinateError:
                    n_err += 1
                    frag_hist[nf] = frag_hist.get(nf, 0) + 1
                except Exception:
                    pass
        per_sigma[sigma] = {
            'trials': n_trials,
            'coordinate_errors': n_err,
            'fragmented_geoms': n_frag,
        }
    return per_sigma, frag_hist


# ---------------------------------------------------------------------------
# 4. Threshold proximity vs chain length (clean geometries)
# ---------------------------------------------------------------------------
def threshold_table():
    from itertools import combinations

    rows = []
    for name in [f'PPE_n{i}' for i in range(1, 9)]:
        geom, _ = PPE_BY_NAME[name]
        co = geom.coords
        bm, _, _ = covalent_bondmatrix(geom)
        angs = []
        for j in range(len(bm)):
            nbrs = [int(n) for n in np.flatnonzero(bm[j])]
            for i, k in combinations(nbrs, 2):
                angs.append(Angle(i, j, k).eval(co) * DEG)
        angs = np.array(angs)
        near = angs[angs > 170]
        rows.append({
            'molecule': name,
            'atoms': len(geom),
            'n_angles_gt_170': int((angs > 170).sum()),
            'n_within_2deg_of_175': int((np.abs(angs - 175) < 2).sum()),
            'min_gap_to_175_deg': float(np.abs(near - 175).min()) if len(near) else None,
        })
    return rows


def make_figure(per_sigma):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sig = sorted(per_sigma)
    frac_frag = [per_sigma[s]['fragmented_geoms'] / per_sigma[s]['trials'] for s in sig]
    frac_err = [per_sigma[s]['coordinate_errors'] / per_sigma[s]['trials'] for s in sig]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(sig, [100 * x for x in frac_frag], 'o-', label='geometry fragmented (covalent)')
    ax.plot(sig, [100 * x for x in frac_err], 's-', label='CoordinateError raised')
    ax.axvspan(0, 0.06, color='0.9', label='physically reasonable noise')
    ax.set_xlabel('start-geometry noise $\\sigma$ (Å, per Cartesian)')
    ax.set_ylabel('% of PPE n1–n8 trials')
    ax.set_title('CoordinateError tracks fragmentation, not noise')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(-2, 102)
    fig.tight_layout()
    fig.savefig(OUT / 'fragmentation_onset.png', dpi=140)


def main():
    canonical = measure_canonical()
    sigmas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    per_sigma, frag_hist = sweep(sigmas, n_seeds=40)
    table = threshold_table()
    result = {
        'canonical_case': canonical,
        'per_sigma': {str(k): v for k, v in per_sigma.items()},
        'coordinate_error_by_fragment_count': {str(k): v for k, v in sorted(frag_hist.items())},
        'threshold_vs_length': table,
    }
    (DATA / 'analysis.json').write_text(json.dumps(result, indent=2))
    make_figure(per_sigma)

    # console summary
    print('canonical center bond:', canonical['center_bond'])
    print('CoordinateError by covalent-fragment-count:',
          result['coordinate_error_by_fragment_count'])
    for s in sigmas:
        ps = per_sigma[s]
        print(f"sigma={s}: {ps['coordinate_errors']}/{ps['trials']} errors, "
              f"{ps['fragmented_geoms']}/{ps['trials']} fragmented")


if __name__ == '__main__':
    main()
