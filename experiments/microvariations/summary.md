# Geometric micro-variation experiment

Each row reports the optimizer outcome for one molecule under Gaussian noise of standard deviation `sigma` (angstrom) applied to every Cartesian coordinate of the Birkholz-Schlegel starting geometry. The PES is MOPAC PM7 (the same backend that produced the `mopac_pm7_steps` column of `tests/data/birkholz_schlegel/reference.json`). Each non-zero sigma cell aggregates 10 seeds.

`conv` is the fraction of seeds that converged within `--maxsteps`. `steps` is the median step count over converged seeds (parenthetical min/max). `dE` is the maximum |final_energy - baseline_energy| in kcal/mol over converged seeds (kept in MOPAC's natural unit so the numbers are easy to read). `RMSD` is the maximum Kabsch-aligned final-structure RMSD vs the sigma=0 minimum in angstrom.


## estradiol

Baseline (sigma=0): converged in 11 steps, E = -0.151 hartree.

| sigma (A) | conv | steps (min/max) | max dE (kcal/mol) | max RMSD (A) |
|---:|---:|---:|---:|---:|
| 0.001 | 10/10 | 11 (11/11) | 0.066 | 0.0017 |
| 0.005 | 10/10 | 11 (8/55) | 5.088 | 0.2371 |
| 0.01 | 10/10 | 11 (8/55) | 5.088 | 0.2371 |
| 0.05 | 9/10 | 25 (11/60) | 5.088 | 0.2373 |

## artemisinin

Baseline (sigma=0): converged in 25 steps, E = -0.264 hartree.

| sigma (A) | conv | steps (min/max) | max dE (kcal/mol) | max RMSD (A) |
|---:|---:|---:|---:|---:|
| 0.001 | 10/10 | 25 (24/25) | 0.000 | 0.0008 |
| 0.005 | 10/10 | 25 (24/26) | 0.000 | 0.0009 |
| 0.01 | 10/10 | 25 (24/29) | 0.000 | 0.0021 |
| 0.05 | 10/10 | 27 (25/29) | 0.000 | 0.0021 |

## vitamin_c

Baseline (sigma=0): converged in 29 steps, E = -0.366 hartree.

| sigma (A) | conv | steps (min/max) | max dE (kcal/mol) | max RMSD (A) |
|---:|---:|---:|---:|---:|
| 0.001 | 10/10 | 29 (29/30) | 0.000 | 0.0005 |
| 0.005 | 10/10 | 30 (29/31) | 0.000 | 0.0015 |
| 0.01 | 10/10 | 30 (29/35) | 0.000 | 0.0020 |
| 0.05 | 10/10 | 32 (29/38) | 0.000 | 0.0015 |

## codeine

Baseline (sigma=0): converged in 31 steps, E = -0.117 hartree.

| sigma (A) | conv | steps (min/max) | max dE (kcal/mol) | max RMSD (A) |
|---:|---:|---:|---:|---:|
| 0.001 | 10/10 | 31 (31/32) | 0.000 | 0.0012 |
| 0.005 | 10/10 | 31 (30/33) | 0.000 | 0.0018 |
| 0.01 | 9/10 | 33 (31/35) | 0.000 | 0.0018 |
| 0.05 | 8/10 | 32 (29/36) | 0.000 | 0.0020 |

## mg_porphin

Baseline (sigma=0): converged in 34 steps, E = 0.379 hartree.

| sigma (A) | conv | steps (min/max) | max dE (kcal/mol) | max RMSD (A) |
|---:|---:|---:|---:|---:|
| 0.001 | 10/10 | 33 (29/40) | 0.001 | 0.0426 |
| 0.005 | 10/10 | 33 (25/44) | 0.128 | 0.0426 |
| 0.01 | 10/10 | 34 (25/58) | 0.000 | 0.0426 |
| 0.05 | 10/10 | 36 (11/60) | 1.892 | 0.4918 |

## easc

Baseline (sigma=0): converged in 53 steps, E = -0.368 hartree.

| sigma (A) | conv | steps (min/max) | max dE (kcal/mol) | max RMSD (A) |
|---:|---:|---:|---:|---:|
| 0.001 | 10/10 | 53 (52/56) | 0.001 | 0.0026 |
| 0.005 | 10/10 | 54 (51/57) | 0.001 | 0.0027 |
| 0.01 | 10/10 | 54 (52/61) | 6.111 | 0.9939 |
| 0.05 | 10/10 | 48 (40/62) | 6.177 | 1.0960 |

## How to read this

A flat row across `sigma` columns means the optimizer is insensitive to that scale of starting-geometry noise. Step count creeping up with sigma is the expected behaviour: a larger perturbation is farther from the minimum and farther outside the initial trust region. A drop in `conv` indicates seeds that hit the `--maxsteps` ceiling. A large `max dE` paired with a large `max RMSD` indicates at least one seed converged to a different minimum (or hit a saddle); the baseline column tells you what the "intended" minimum was.

## Findings

The 246 runs split cleanly into three regimes by molecule:

1. **Single-basin, robust** — `artemisinin`, `vitamin_c`. 100% convergence at every sigma; the median step count grows by at most 2 between sigma=0 and sigma=0.05; every converged seed lands within 10^-3 hartree of the baseline minimum. The optimizer is essentially indifferent to noise on this scale for these molecules.

2. **Multiple nearby basins** — `estradiol`, `mg_porphin`, `easc`. These converge with high probability but a non-trivial fraction of seeds land in an alternate conformer:
   - `estradiol`: starting at sigma=0.005, max dE jumps to 5.09 kcal/mol with RMSD 0.24 A and the per-seed step-count range widens to 8-55. The PM7 PES therefore has a secondary minimum within a 0.005-A noise ball of the published starting structure.
   - `mg_porphin`: at sigma=0.05, max dE 1.89 kcal/mol, RMSD 0.49 A; the porphyrin ring puckers into an alternate distortion. Convergence still 10/10.
   - `easc`: at sigma >= 0.01, max RMSD ~1.0 A and dE ~6.1 kcal/mol — the most dramatic basin-hopping in the set, even though 10/10 seeds reach a stationary point.

3. **Slow-to-converge under noise** — `codeine`. Same basin everywhere (dE = 0.000, RMSD <= 0.002 A), but at sigma=0.05 two seeds out of ten exhaust the 120-step budget; the surviving median is still 32 steps.

Two practical takeaways for the pyberny benchmark:
- The single-point step counts in `reference.json` are reasonably representative for `artemisinin`/`vitamin_c`/`mg_porphin`/`easc` (within a couple of steps of the perturbed median) but understate the spread for `estradiol` (8-55 once you wiggle the start) and `codeine` at sigma=0.05.
- "Converged" is not synonymous with "same minimum": `estradiol`, `mg_porphin`, and `easc` all show seeds that finish at a different conformer of the molecule. A future benchmark gate that compared *final structures* (not just step counts) would catch this.

## Tight-threshold follow-up

The within-basin energy spreads of 50-200 µHa observed at small sigma for
estradiol and mg_porphin sit ~100x above pyberny's default gradient-implied
limit (`gradmax = 0.45e-3 a.u.`, `stepmax = 1.8e-3 a.u.`). To test whether
this is loose-on-soft-modes termination or something else, the four
suspicious cells were rerun with all four convergence thresholds tightened
10x (`experiments/microvariations/microvariation_tight.py`, results in
`results_tight.json`).

| Cell | Default spread | Tight spread | Default median steps | Tight median steps | Verdict |
|---|---:|---:|---:|---:|---|
| vitamin_c sigma=0.001 (control) | 0.08 µHa | **0.00 µHa** | 29 | 34 | Already at noise; tightens cleanly. |
| mg_porphin sigma=0.001 | 1.3 µHa | **0.05 µHa** | 34 | 42 | Confirms soft-mode termination: ~27x collapse. |
| mg_porphin sigma=0.005 | 205 µHa | **0.05 µHa** | 35 | 47 | Confirms soft-mode termination: ~4000x collapse, driven by a single outlier seed that had stopped at the wrong amplitude of the macrocycle out-of-plane mode. |
| estradiol sigma=0.001 | 154 µHa | **154 µHa** | 11 | 11 | Tightening had **zero** effect: all 10 seeds produced bit-identical trajectories. |

The estradiol result is the interesting one. The 10 seeds all converge in
exactly 11 steps under both the default and 10x-tighter criteria, with
identical final coordinates and identical 154 µHa cross-seed spread. That
means *the loose criteria were not the binding constraint*: by step 11 the
gradient and step magnitudes were already well below the tight thresholds.
The energy spread is intrinsic to how the sigma=0.001 starting perturbation
propagates through 11 BFGS steps - presumably the BFGS-projected step in
estradiol's softest internal coordinate (a phenolic OH or methyl torsion)
shrinks below `stepmax` quickly enough to stop the optimizer at slightly
displaced points, with no remaining driver to push them together.

So the 100-200 µHa "loose convergence" picture from the first run is
**half right**: it applies to mg_porphin (and presumably any molecule with
genuinely soft macrocycle modes), but for estradiol the spread is a
BFGS-trajectory phenomenon that the gradient threshold cannot influence.
Fixing the latter would require either a stricter step criterion that
specifically responds to the BFGS Hessian's projection onto soft modes, or
forcing additional polish iterations beyond the standard convergence test.

### MOPAC-side tightening does not help either

Re-running estradiol sigma=0.001 with the MOPAC `PRECISE` keyword
(tightens SCF criterion 100x) had no effect on the cross-seed energy
spread:

| condition | spread (uHa) | stdev (uHa) |
|---|---:|---:|
| default | 154.15 | 46.98 |
| MOPAC `PRECISE` only | 152.76 | 46.85 |
| pyberny 10x only | 154.15 | 46.98 |
| both | 154.29 | 47.02 |

With `PRECISE` each seed's absolute energy shifts by ~4 uHa (the MOPAC
SCF noise floor) but the cross-seed *pattern* is preserved bit-for-bit.
So MOPAC's SCF noise is ~4 uHa per call, well below the 154 uHa
cross-seed spread - confirming the spread is a pyberny BFGS-trajectory
effect, not MOPAC numerical noise.

## Connecting the estradiol minima by linear interpolation

The default-tolerance run found 5 energy-distinct minima for estradiol
(clustered at 0.5 kcal/mol tolerance): basin 0 at -0.158928 Ha (deepest,
6 seeds) up to basin 4 at -0.151174 Ha (the published Birkholz-Schlegel
start, 28 seeds in this cluster). `experiments/microvariations/estradiol_minima_path.py`
Kabsch-aligns one representative seed per basin and runs MOPAC PM7
single points along straight-line Cartesian interpolations between
consecutive basins. Plot: `minima_interpolation.png`.

The result is striking: **every segment is monotonically uphill, no
peaks**. The energy slides smoothly from each deeper basin up to the
shallower one along the linear path.

Endpoint slopes (numerical derivative of E along the path) clarify what's
real and what isn't:

| Segment | slope at t=0 (deeper basin) | slope at t=1 (shallower basin) |
|---|---:|---:|
| 0 -> 1 | +89 uHa/unit | +1620 uHa/unit |
| 1 -> 2 | +5 uHa/unit | +5741 uHa/unit |
| 2 -> 3 | +94 uHa/unit | +3528 uHa/unit |
| 3 -> 4 | **+1647 uHa/unit** | +3512 uHa/unit |

Basins 0, 1, 2 look like genuine minima (slope ~0 at t=0). Basin 3 has
a substantial slope leaving toward basin 4 (+1647 uHa/unit), and at t=1
every segment shows a large positive slope - meaning the higher basins
sit on shoulders the linear path is still climbing to reach. **Basins 3
and 4 are very likely not true minima** but points where pyberny's
gradient criteria were satisfied even though there's a downhill
direction in Cartesian space toward a deeper basin. The published
estradiol.xyz starting structure is one of these (basin 4, ~4.86
kcal/mol above the true PM7 minimum).

## Why pyberny declares basin 3 and basin 4 as converged

`experiments/microvariations/estradiol_verbose_diag.py` re-runs the offending seeds with
full pyberny logging plus `debug=True` so we can grab the optimizer
state at termination. The verbose logs land in
`experiments/microvariations/verbose_diag/log_*.txt`.

The diagnostics:

| metric                            | basin 4 (high) | basin 3        | basin 0 (deepest, sanity) |
|-----------------------------------|---------------:|---------------:|--------------------------:|
| Internal-coord gradient (max)     | 2.0e-6         | 5.0e-7         | 1.9e-5                    |
| Convergence threshold (max)       | 4.5e-4         | 4.5e-4         | 4.5e-4                    |
| **Internal grad below threshold** | **220x**       | **910x**       | **23x**                   |
| **Cartesian gradient norm**       | **3.6e-2 Ha/bohr** | **2.3e-2 Ha/bohr** | **8.9e-5 Ha/bohr**    |
| Proj. of g_cart onto basin-0 dir  | -6.8e-3        | -4.8e-3        | -6.3e-6                   |
| Pseudoinverse gap (last step)     | **1.9e+03**    | **1.6e+05**    | (no warning)              |
| Smallest BFGS Hessian eigenvalue  | 0.002          | **0.0002**     | 0.0008                    |

At a *true* minimum (basin 0), the Cartesian gradient norm is 8.9e-5
Ha/bohr - genuinely small. At basins 3 and 4 it's ~25-400x larger than
that and ~50-80x above pyberny's gradient threshold itself - the
Cartesian gradient is far from zero. **Yet pyberny's internal-coord
gradient reads 5 orders of magnitude below threshold.**

The culprit is `src/berny/Math.py:15`. The redundant-internal-coord
gradient transformation goes via `pinv(B B^T)`, where `B` is the
Wilson B-matrix mapping internals to Cartesians. Pyberny's `pinv`:

```python
gaps = D[:-1] / D[1:]            # ratios between consecutive singular values
n = np.flatnonzero(gaps > 1e3)[0]  # first gap above 1000
D[n + 1 :] = 0                   # zero everything past that gap
```

When the redundant internals become linearly dependent in some
Cartesian direction - which is what happens at basins 3 and 4, where
the verbose log records `Pseudoinverse gap of only: 1.9e+03` (basin 4)
and `1.6e+05` (basin 3) - that direction is **silently discarded**.
The 3.6e-2 Ha/bohr Cartesian gradient component along it gets projected
to zero, leaving an internal gradient of 2e-6. The convergence test
sees a satisfied criterion and the optimizer halts.

Pyberny *does* log the warning, but does not propagate it into the
convergence test - so a user reading only `converged=True` would never
notice the optimization terminated on a rank-deficient projection.

Potential fixes (out of scope here):
- Add a backup Cartesian-gradient convergence criterion.
- Refuse to declare convergence whenever `Math.pinv` issues the gap
  warning at the final step.
- Investigate the `Transformation did not converge in 20 iterations`
  step that immediately precedes both pathological convergences -
  that's the Cartesian<->internal back-transform also struggling on
  the same singular geometry, and it could be an earlier red flag.

## Which internal coordinate becomes problematic, exactly

`experiments/microvariations/estradiol_internals_diag.py` tracks the B-matrix's singular
value spectrum at every step and identifies which internal coordinates
contribute to the discarded null direction.

For both pathological cases the trace is identical and traces to a
**single near-linear angle**: `Angle(37-36-40)` = **H37-C36-O40, the
H-C-O angle at estradiol's 17-beta-hydroxyl carbon**.

| Step | basin 4: H-C-O angle | sv[0] of B*B^T | pinv gap index | gap |
|---|---:|---:|---:|---:|
| 1-6 | ~110-160 deg | ~1e4 | 125 (natural)  | ~1e13 |
| 7 (warning) | 176.23 deg | ~3e4 | 125 (natural)  | 6e11 |
| 8 (final) | **179.59 deg** | **30898** | **0 (spurious)** | 1927 |

| Step | basin 3: H-C-O angle | sv[0] of B*B^T | pinv gap index | gap |
|---|---:|---:|---:|---:|
| 1-7 | normal | ~1e4 | 125 (natural)  | ~1e13 |
| 8 | ~178 deg | ~1.6e6 | 125 (natural)  | 3e10 |
| 9 (final) | **179.96 deg** | **2.5e6** | **0 (spurious)** | 155387 |

At a true minimum (basin 0), the H-C-O angle stays around 110 deg and
`sv[0]` stays ~1e4 throughout all 25 steps; the pinv gap is the
"natural" one at index 125 (after the 3N - 6 = 126 chemical DOFs).

### Mechanism

The BFGS step at the previous iteration pushes H37-C36-O40 past 175
deg. Two dihedrals (`Dihedral(37-36-40-34)` and `Dihedral(37-36-40-41)`)
use this near-180 angle as one of their constituent angles. In
`Dihedral.eval(grad=True)` (`src/berny/coords.py:148-204`) the gradient
formula contains `1 / norm(a1)`, where `a1 = v1 - dot(v1, ew) * ew` is
the projection of v1 onto the plane perpendicular to the central bond.
**When the H-C-O angle goes to 180 deg, `norm(a1) -> 0` and the
dihedral's B-row gradient magnitude diverges.**

The next iteration computes B*B^T and its top singular value spikes
into the thousands (or millions). `Math.pinv` then finds a spurious
gap at index 0 between `sv[0]` (the rogue dihedral) and `sv[1]` (a
normal-magnitude coordinate), truncates everything past index 0, and
returns a pseudoinverse that retains only the one uninformative
direction. The projection of the Cartesian gradient through this
crippled pseudoinverse yields an internal gradient ~1e-7 Ha/bohr
- far below threshold - and the convergence test passes.

### What this means for the published Birkholz-Schlegel start

The published estradiol.xyz starts with a normal H-C-O angle, but
**pyberny's BFGS step magnitude is large enough that small starting
perturbations can push this angle past 175 deg within a few steps**.
Once that happens, the dihedral gradient blows up and the pinv
truncation fires - producing a "converged" structure ~5 kcal/mol above
the true PM7 minimum.

This is not a soft-mode story. It's a **coordinate-singularity
catastrophe** triggered by an sp3 carbon being pushed close to
inversion (planar) by the BFGS path. Robust fixes would include:
- Special-casing dihedral gradients when adjacent angles approach 180 deg
  (analogous to the existing `abs(phi) > pi - 1e-6` branch for the
  dihedral itself).
- Trust-radius clipping per-coordinate to prevent any individual angle
  step from crossing 175 deg.
- Rebuilding the internal-coord set when an angle goes near-linear,
  removing the now-ill-defined dihedrals.

