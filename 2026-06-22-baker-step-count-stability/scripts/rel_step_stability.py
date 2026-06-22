#!/usr/bin/env python3
"""Step-count stability under *per-system* noise scaled to the start->minimum RMSD.

Companion to `step_stability.py`, which used fixed absolute amplitudes σ. That
design over-perturbs molecules whose pre-relaxed Baker start already sits almost
exactly at the minimum (tiny start->min RMSD), inflating their step count purely
because the noise dwarfs the natural travel distance. This script removes that
confound: for each molecule the noise amplitude is derived so that the
perturbation RMSD is a fixed *fraction* of that molecule's own start->minimum
RMSD (R_sm).

For a fraction f, the per-coordinate Gaussian σ is `f * R_sm / sqrt(3)` (raw
RMSD of iid N(0,σ) noise over 3N coordinates is sqrt(3)·σ), so the perturbation
RMSD is ≈ f · R_sm. Run with f ∈ {0.1, 0.2, 0.4}; the 0.2 (20 %) level is the
headline. Same molecule set and same-minimum filtering as `step_stability.py`.

GFN2-xTB via tblite; run from a checkout with pyberny installed.
"""

import json
import statistics as st
import time

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules, load_reference
from berny.solvers import XTBSolver

H2KCAL = 627.5094740631
FRUSTRATED = {
    "methylamine",
    "mesityl_oxide",
    "benzidine",
    "acanil01",
    "caffeine",
    "ethanol",
    "histidine",
}
FRACS = [0.1, 0.2, 0.4]
NSEED = 30


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((a @ rot.T - b) ** 2).sum(1).mean()))


def optimize(geom, ref, maxsteps=150):
    berny = Berny(geom, maxsteps=maxsteps)
    solver = XTBSolver(charge=ref["charge"], mult=ref["mult"])
    next(solver)
    energy = None
    g = geom
    for g in berny:
        energy, gradients = solver.send((list(g), g.lattice))
        berny.send((energy, gradients))
    return berny.converged, berny._n, energy, g


def main():
    reference = load_reference("baker")
    names = sorted(n for n in reference if n not in FRUSTRATED)
    geoms = {n: (g, r) for n, g, r in iter_molecules("baker", names)}

    t0 = time.time()
    out = {}
    for name in names:
        geom, ref = geoms[name]
        _conv0, n0, e0, gmin = optimize(geom, ref)
        r_sm = kabsch_rmsd(geom.coords, gmin.coords)
        rec = {"clean_steps": n0, "Rsm": r_sm, "by_frac": {}}
        for frac in FRACS:
            sigma = frac * r_sm / np.sqrt(3)
            steps = []
            noise_rmsd = []
            diff = 0
            for seed in range(NSEED):
                rng = np.random.default_rng((int(frac * 1000), seed, 7))
                disp = rng.normal(0.0, sigma, geom.coords.shape)
                noisy = geomlib.Geometry(
                    list(geom.species), geom.coords + disp, geom.lattice
                )
                noise_rmsd.append(kabsch_rmsd(noisy.coords, geom.coords))
                try:
                    conv, n, e, _ = optimize(noisy, ref)
                except Exception:
                    continue
                if not conv or (e is not None and abs((e - e0) * H2KCAL) > 0.1):
                    diff += 1
                    continue
                steps.append(n)
            rec["by_frac"][str(frac)] = {
                "sigma": sigma,
                "steps": steps,
                "noise_rmsd_mean": float(np.mean(noise_rmsd)),
                "diff_basin": diff,
            }
        out[name] = rec
        s = rec["by_frac"]["0.2"]["steps"]
        cv = 100 * st.pstdev(s) / st.mean(s) if len(s) > 1 else 0
        infl = st.median(s) / n0 if s else 0
        print(
            f"{name:24s} Rsm={r_sm:.3f} clean={n0:>2}  20%: "
            f"med={int(st.median(s)) if s else 0:>2} CV={cv:>3.0f}% infl={infl:.2f}x",
            flush=True,
        )

    with open("rel_step_stability.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDONE {time.time() - t0:.0f}s -> rel_step_stability.json")


if __name__ == "__main__":
    main()
