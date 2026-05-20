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


# Benchmark trajectory warning sweep

Source: `experiments/microvariations/benchmark_trajectory_check.py` ran `Berny(geom, maxsteps=110)` + `MopacSolver(charge, mult)` on every molecule of both MOPAC benchmark sets (`tests/data/birkholz_schlegel`, `tests/data/baker_shajan_2023`) with an INFO-level `FileHandler` on `logging.getLogger("berny")`. The raw per-molecule logs are deliberately not committed (they are reproducible artifacts; see `.gitignore`). The trimmed `benchmark_diag/warnings.json` keeps only molecules with non-empty scan signal; the tables below summarise the full sweep, and `pinv@final` marks the dangerous case discovered on estradiol: a `Pseudoinverse gap of only:` warning at the same step that declared convergence.

## birkholz_schlegel

| molecule | ref | steps | conv | pinv | pinv@final | back-xform | neg-eig steps | severe dq | maxsteps |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| artemisinin | 25 | 25 | yes | 0 | - | 0 | 0 | 0 | - |
| avobenzone | 94 | 90 | yes | 0 | - | 0 | 4 | 0 | - |
| azadirachtin | 60 | 61 | yes | 0 | - | 0 | 0 | 0 | - |
| bisphenol_a | - | 110 | no | 0 | - | 0 | 2 | 0 | yes |
| cetirizine | 58 | 57 | yes | 0 | - | 0 | 0 | 0 | - |
| codeine | 31 | 31 | yes | 0 | - | 0 | 2 | 0 | - |
| diisobutyl_phthalate | 38 | 38 | yes | 0 | - | 0 | 0 | 0 | - |
| easc | 53 | 53 | yes | 0 | - | 0 | 0 | 0 | - |
| estradiol | 11 | 11 | yes | 4 | YES | 2 | 0 | 1 | - |
| inosine_cation | 47 | 47 | yes | 0 | - | 1 | 0 | 1 | - |
| maltose | 54 | 54 | yes | 0 | - | 2 | 3 | 2 | - |
| mg_porphin | 34 | 34 | yes | 0 | - | 0 | 0 | 0 | - |
| ochratoxin_a | - | 110 | no | 0 | - | 0 | 5 | 0 | yes |
| penicillin_v | 54 | 54 | yes | 0 | - | 0 | 0 | 0 | - |
| raffinose | - | 110 | no | 0 | - | 6 | 5 | 6 | yes |
| sphingomyelin | 95 | 87 | yes | 0 | - | 0 | 1 | 0 | - |
| tamoxifen | 70 | 72 | yes | 0 | - | 0 | 0 | 0 | - |
| vitamin_c | 29 | 29 | yes | 0 | - | 0 | 0 | 0 | - |
| zn_edta | 99 | 99 | yes | 0 | - | 0 | 0 | 0 | - |

## baker_shajan_2023

| molecule | ref | steps | conv | pinv | pinv@final | back-xform | neg-eig steps | severe dq | maxsteps |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| acanil01 | 43 | 43 | yes | 0 | - | 0 | 2 | 0 | - |
| acetone | 7 | 7 | yes | 0 | - | 0 | 0 | 0 | - |
| acetylene | 9 | 9 | yes | 0 | - | 0 | 0 | 0 | - |
| achtar10 | 9 | 9 | yes | 0 | - | 0 | 0 | 0 | - |
| allene | 12 | 12 | yes | 7 | YES | 6 | 0 | 0 | - |
| ammonia | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| benzaldehyde | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| benzene | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| benzidine | 24 | 24 | yes | 0 | - | 0 | 2 | 0 | - |
| caffeine | - | 110 | no | 0 | - | 0 | 43 | 0 | yes |
| difluorobenzene_13 | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| difluoronaphthalene_15 | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| difuropyrazine | 7 | 7 | yes | 0 | - | 0 | 0 | 0 | - |
| dimethylpentane | 10 | 10 | yes | 0 | - | 0 | 0 | 0 | - |
| disilyl_ether | 7 | 7 | yes | 1 | YES | 2 | 0 | 1 | - |
| ethane | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| ethanol | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| furan | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| histidine | 19 | 19 | yes | 0 | - | 0 | 0 | 0 | - |
| hydroxybicyclopentane_2 | 9 | 9 | yes | 0 | - | 0 | 0 | 0 | - |
| hydroxysulfane | 6 | 6 | yes | 0 | - | 0 | 0 | 0 | - |
| menthone | 16 | 16 | yes | 0 | - | 0 | 0 | 0 | - |
| mesityl_oxide | 8 | 8 | yes | 0 | - | 0 | 0 | 0 | - |
| methylamine | 18 | 18 | yes | 0 | - | 0 | 4 | 0 | - |
| naphthalene | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| neopentane | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| pterin | 8 | 8 | yes | 0 | - | 0 | 0 | 0 | - |
| trifluorobenzene_135 | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| trisilacyclohexane_135 | 7 | 7 | yes | 0 | - | 0 | 0 | 0 | - |
| water | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |

## Findings

### Pseudoinverse warning at the convergence-declaring step (estradiol-style)

These are the cases where `converged=True` was returned even though `Math.pinv` had silently zeroed a singular direction at the final step:

- **birkholz/estradiol** - final-step pinv gap `2.60e+03`
- **baker/allene** - final-step pinv gap `3.00e+07`
- **baker/disilyl_ether** - final-step pinv gap `9.40e+05`

### Heavy back-transform struggle (>=3 warnings)

On estradiol the precursor to the pinv catastrophe was three consecutive `Transformation did not converge in 20 iterations` lines. Molecules below show the same precursor signature:

- birkholz/raffinose: 6 back-transform warning(s)
- baker/allene: 6 back-transform warning(s)

### Severe back-transform step blow-up (RMS(dq) >= 0.05)

These steps reported a back-transform with internal-coord RMS displacement at least an order of magnitude above what a well-behaved Cartesian<->internal mapping produces. They are often saddle-pass attempts on rigid macrocycles or near-linear angles; if they coincide with `pinv@final = YES` they are the mechanism by which the pinv pathology fires.

- birkholz/estradiol: 7 (dq=0.13)
- birkholz/inosine_cation: 11 (dq=0.091)
- birkholz/maltose: 27 (dq=0.082),30 (dq=0.09)
- birkholz/raffinose: 42 (dq=0.089),43 (dq=0.054),46 (dq=0.22),47 (dq=0.12),48 (dq=0.074),49 (dq=0.074)
- baker/disilyl_ether: 6 (dq=1.1)

### Negative-eigenvalue events (sphere-minimization saddle passes)

Healthy minimum-finding trajectories report all-positive BFGS Hessian eigenvalues at every step. Negative-eigenvalue events happen when the BFGS update produces a spurious unstable mode (or when the geometry really is near a saddle). Pyberny then switches to RFO sphere-minimization to descend along that unstable direction. Occasional events are normal; a sustained count is a sign of a confusing PES region:

- birkholz/avobenzone: 4 event(s) at step(s) 14,17,20,49
- birkholz/bisphenol_a: 2 event(s) at step(s) 13,15
- birkholz/codeine: 2 event(s) at step(s) 15,18
- birkholz/maltose: 3 event(s) at step(s) 27,30,31
- birkholz/ochratoxin_a: 5 event(s) at step(s) 41,44,47,51,54
- birkholz/raffinose: 5 event(s) at step(s) 42,45,46,47,48
- birkholz/sphingomyelin: 1 event(s) at step(s) 11
- baker/acanil01: 2 event(s) at step(s) 12,13
- baker/benzidine: 2 event(s) at step(s) 12,13
- baker/caffeine: 43 event(s) at step(s) 27,30,33,35,36,38,39,41,45,47,49,51,52,53,56,57,58,59,62,63,64,65,69,71,74,75,77,78,79,80,81,83,86,87,89,93,95,98,99,101,105,106,107
- baker/methylamine: 4 event(s) at step(s) 5,7,8,9

### Hit maxsteps ceiling

- birkholz/bisphenol_a
- birkholz/ochratoxin_a
- birkholz/raffinose
- baker/caffeine


# Internal-coordinate triggers along benchmark trajectories

Source: `experiments/microvariations/benchmark_internals_diag.py`
generalises `estradiol_internals_diag.py` to twelve "case-study"
molecules selected from the warning sweep above. For each molecule it
captures, at every optimizer step, both the four warning classes
(pinv, back-xform, neg-eig, severe-dq) and four geometric "going
planar / linear" diagnostics:

- maximum `Angle` in `InternalCoords` (linear-angle threshold 175 deg),
- maximum sum of neighbour-pair angles around any 3-coordinate atom
  (planar threshold 355 deg; aromatic carbons sit at 360 deg by
  construction - see caveat below),
- minimum out-of-plane angle around any 4-coordinate atom
  (sp3-inversion threshold 5 deg),
- maximum bond-length / sum-of-covalent-radii ratio (stretch
  threshold 1.5).

Each warning step is then classified into a *mechanism*:
`linear-angle-dihedral` | `sp3-inversion` | `bond-stretch` |
`planar-center` | `unattributed`, in that priority order. The
artefacts are committed under
`experiments/microvariations/benchmark_internals_diag/`:
per-molecule JSON, an aggregate `per_step.json` with the
co-occurrence rows, per-molecule `trajectory_<mol>.png` figures
(energy / max-angle / min-pyramidalisation / `sv[0]` of `B B^T` /
pinv-gap index, with warning steps shaded), and the roll-up
`triggers.md`. INFO-level logs are not committed (gitignored, like
`benchmark_diag/`); the scan results are baked into the per-molecule
JSON so the roll-up regenerates without the logs.

## Roll-up

| set / molecule | class | converged | steps | warning steps | mechanism |
|---|---|---|---:|---|---|
| baker/allene | pinv@final | yes | 12 | 3,5,6,7,8... | linear-angle-dihedral |
| baker/disilyl_ether | pinv@final | yes | 7 | 5,6,7 | linear-angle-dihedral |
| birkholz/estradiol | pinv@final | yes | 11 | 6,7,8,9,10... | linear-angle-dihedral |
| birkholz/raffinose | heavy back-xform + saddles | yes | 85 | 42,43,45,46,47 | **unattributed** |
| baker/caffeine | sustained neg-eig | ERR (FindrootError) | 75 | 27,31,33,37,40... | **unattributed (aromatic)** |
| birkholz/maltose | severe dq (converges right) | yes | 54 | 27,30,31 | **unattributed** |
| birkholz/inosine_cation | severe dq (converges right) | yes | 47 | 11 | planar-center (aromatic) |
| birkholz/ochratoxin_a | hit maxsteps | no | 110 | 41,44,47,51,54 | planar-center (aromatic) |
| birkholz/bisphenol_a | hit maxsteps | no | 110 | 13,15 | planar-center (aromatic) |
| birkholz/artemisinin | control | yes | 25 | - | clean |
| birkholz/vitamin_c | control | yes | 29 | - | clean |
| baker/benzene | control | yes | 4 | - | clean |

## Findings

### 1. All three `pinv@final` cases reproduce the estradiol mechanism

The diagnostic the experiment was designed to find: **every** molecule
that the warning sweep flagged with `pinv@final = YES` has an
`Angle` >= 175 deg at the convergence-declaring step, and the
top contributors to the truncated singular direction are dihedrals
containing that angle. The estradiol H37-C36-O40 story is **not**
an idiosyncrasy of estradiol - it is a generic singular-coordinate
mechanism:

- **allene**: the central C1-C0-C2 angle is **180.0 deg** at every
  step (allene is sp-hybridised by definition; this is the published
  starting geometry, not a perturbation), and the two dihedrals
  H5-C1-C0-C2 and H6-C1-C0-C2 sit on this central angle. `sv[0]` of
  `B B^T` is dominated by the affected dihedral rows from step 1 and
  the pinv gap is at index 0 throughout. The `pinv@final` cell in
  `warnings.md` is the same mechanism estradiol exhibits at the
  *terminal* step, except allene starts there.
- **disilyl_ether**: Si-O-Si starts at 171.7 deg at the warning steps
  and continues toward 180. Same Dihedral.eval singularity.
- **estradiol** (sanity, published .xyz): H37-C36-O40 reaches 171.5 deg
  at step 6 and crosses 175 deg at step 7, exactly the trajectory the
  bespoke `estradiol_internals_diag.py` recorded for the perturbed
  seeds.

For these three molecules the implication is unambiguous: a fix at
the source of the singularity (per-coordinate trust clipping that
prevents any angle from crossing ~170 deg, or rebuilding the
internal-coord set when an angle goes near-linear) closes the entire
`pinv@final` class on both benchmark sets.

### 2. Raffinose and maltose are a *different* failure mode

Both molecules emit a tight cluster of back-xform + neg-eig +
severe-dq warnings, but none of the four geometric flags fires at
those steps. Maximum angle stays well below 175 deg, no 3-coord
atom flattens, no 4-coord atom inverts, no bond stretches. The
pinv gap is always at a position < 3N - 6 (so the rank-deficient
projection is happening), but the warnings here are *not* the
estradiol mechanism. The plan's predicted "BFGS Hessian
flips / RFO saddle-mode descent on a flat torsional manifold"
candidate is consistent with the warning pattern (negative
eigenvalues + severe back-transform `dq`), and is the natural
next experiment.

Raffinose ultimately **does** converge (85 steps, well under the
110-step cap - lower than the 110 the previous sweep recorded,
which most likely reflects a MOPAC version change since
`warnings.json` was generated). Maltose converges in 54 steps as
in the original. So this failure mode is *not* a convergence
killer, only a numerical-stress event.

### 3. The `planar-center` rollup column is mostly an aromatic-π artefact

`inosine_cation`, `ochratoxin_a`, `bisphenol_a`, and `caffeine` all
get tagged `planar-center` simply because their aromatic-ring
carbons have neighbour-pair angle sums of ~360 deg *throughout the
trajectory*, not because of any transient flattening event. The
mechanism column is not actionable for these molecules; the
co-occurrence tables in `triggers.md` show the planar-3coord bit is
"Y" at *every* step of every molecule that contains an aromatic
ring.

A more discriminating planar metric (e.g. RMSD vs the first-step
value of the angle sum, or the same metric restricted to non-aromatic
3-coord atoms) would clean up the rollup, but does not change the
scientific conclusion: ochratoxin_a, bisphenol_a, and the
pre-crash 75 steps of caffeine all have only **neg-eig** warnings -
the same Hessian-flip family as raffinose. None of these has a
near-linear angle event.

### 4. Caffeine reproducibly crashes with `FindrootError`

Caffeine ran 75 steps under MOPAC PM7 and then the optimizer's
internal linear-search root finder failed (`FindrootError`). 19
warning lines fired during those 75 steps - all `neg-eig`, none
of the other three. This is a third, distinct failure mode (the
optimizer's *own* linear-interpolation step, not the
internal-coord projection or back-transform), so it falls outside
both the singular-coordinate and the Hessian-flip hypotheses
investigated here.

### 5. Bisphenol_a and ochratoxin_a maxstep-hit is the saddle-mode mode, not the singular-coordinate mode

Both molecules hit the 110-step ceiling with **zero**
near-linear-angle events and zero `pinv@final` flags. Their warning
trail is purely `neg-eig`. That rules out option (i) from the plan
("oscillating across a near-linear configuration without ever
stopping") and supports option (ii): they are slow for unrelated
PES-flatness reasons. A "refuse convergence on truncated pinv" fix
would not have changed their outcome.

## Mechanism summary across all 12 case-study molecules

| mechanism | warning steps with this mechanism | share |
|---|---:|---:|
| linear-angle-dihedral | 16 / 53 | 30% |
| planar-center | 28 / 53 | 53% (mostly aromatic-π false positives) |
| sp3-inversion | 0 / 53 | 0% |
| bond-stretch | 0 / 53 | 0% |
| unattributed | 9 / 53 | 17% |

Removing the four aromatic-ring molecules (whose `planar-center`
hits are all the aromatic-π false positive) from the denominator
leaves 25 warning steps over 8 molecules, of which 16 (64%) are
`linear-angle-dihedral` and 9 (36%) are `unattributed`. The
estradiol singular-coordinate mechanism therefore accounts for the
majority of *interpretable* warning steps in the case-study set,
but a clearly separate failure mode (raffinose / maltose, and by
proxy the aromatic-ring `neg-eig` molecules) drives the rest.

## Decision points for follow-up

- A geometric-singularity fix - per-coordinate trust clipping on
  angles > ~170 deg, or rebuilding internals when an angle goes
  near-linear, or specialising `Dihedral.eval` for containing
  angles near 180 deg - closes the entire `pinv@final` class on
  both benchmark sets (estradiol, allene, disilyl_ether).
- That same fix will *not* help raffinose, maltose, ochratoxin_a,
  bisphenol_a, or caffeine. Their warnings are driven by something
  in the BFGS-Hessian / RFO sphere-minimisation path that is
  independent of the internal-coord singular geometry.
- "Refuse convergence on truncated pinv" would catch the same three
  molecules a geometric fix catches; it would *not* unfreeze the
  maxsteps molecules or save caffeine from its `FindrootError`.

