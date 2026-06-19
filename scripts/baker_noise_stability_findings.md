# Baker benchmark: stability of convergence and located minimum under start-geometry noise

Generated with `scripts/noise_stability.py` (GFN2-xTB via `tblite`). Each of
the 30 Baker molecules was optimized from its reference start geometry and
from noisy copies produced by adding isotropic Gaussian noise to every
Cartesian coordinate at five RMS amplitudes (sigma = 0.02, 0.05, 0.1, 0.2,
0.3 A) with 6 independent seeds each — **900 noisy trials + 30 clean
references**. For every trial we record whether it converged, hit the
100-step ceiling, or errored before optimizing, plus its final energy
relative to the clean run. Raw data: `artifacts/baker_noise_stability.json`
(regenerate with the command at the bottom).

## Headline

- **886 / 900 (98.4 %) noisy trials converged.** 6 (0.7 %) hit the 100-step
  ceiling, 8 (0.9 %) raised an exception before/while optimizing. Every
  failure occurs at sigma >= 0.2 A; below that, convergence is 100 %.
- **The located minimum is very stable:** 727 / 886 converged trials (82 %)
  land within 0.1 kcal/mol of the clean minimum, and a further 108 (12 %)
  within 2 kcal/mol (nearby conformers). Only 51 trials (5.8 %) differ by
  more than 2 kcal/mol, and **20 of those 51 occur only at the most extreme
  sigma = 0.3 A**.
- The behaviour splits cleanly into **two regimes**: a benign low-amplitude
  regime (<= 0.2 A) where a small, fixed set of molecules deterministically
  relax to a *lower* minimum than their reference, and a destructive
  high-amplitude regime (0.3 A) where rigid molecules occasionally break.

## Convergence stability

| sigma (A) | trials | converged | ceiling | error | mean steps | median | max | mean steps / clean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean (0) |  30 |  30 | 0 | 0 |  7.4 |  - |  - | 1.00 |
| 0.02 | 180 | 180 | 0 | 0 | 13.5 | 11 | 50 | 2.13 |
| 0.05 | 180 | 180 | 0 | 0 | 14.6 | 12 | 45 | 2.33 |
| 0.1  | 180 | 180 | 0 | 0 | 16.9 | 14 | 83 | 2.73 |
| 0.2  | 180 | 178 | 0 | 2 | 25.5 | 22 | 76 | 4.24 |
| 0.3  | 180 | 168 | 6 | 6 | 33.1 | 31 | 89 | 5.66 |

- **Convergence itself is robust.** Up to 0.1 A every trial converged; even
  at 0.2 A only 2/180 failed and at 0.3 A 12/180 (93 % still converge).
- **Cost grows smoothly with noise.** Mean step count rises from 7.4 (clean)
  to ~2x at 0.02-0.1 A and ~5.7x at 0.3 A. The optimizer recovers from
  displaced starts but pays for it in cycles; worst cases approach the
  100-step ceiling.
- **The 14 hard failures are all at sigma >= 0.2 A** and come in three
  flavours (see `## Representative errors` in the generated report):
  - `CoordinateError` — large distortion makes a near-linear fragment the
    redundant-internal builder cannot form dihedrals through;
  - `LinAlgError: SVD did not converge` / xTB `SCF not converged` — the
    distorted geometry is numerically pathological;
  - step-ceiling non-convergence (caffeine, histidine, ethanol,
    difluoronaphthalene, hydroxybicyclopentane).

## Minimum stability

Distribution of |E_noisy - E_clean| over all 886 converged noisy trials:

| band (kcal/mol) | trials | interpretation |
|---|---:|---|
| <= 0.1            | 727 | same minimum |
| 0.1 - 2           | 108 | nearby conformer |
| 2 - 20            |  31 | different isomer / partial relaxation |
| > 20              |  20 | spurious high-energy structure (all at sigma = 0.3) |

Of the 159 trials that left the clean basin, **137 went *down* in energy and
only 22 went up** — i.e. noise much more often finds a *better* minimum than
a worse one. The two regimes:

### Low-amplitude regime (<= 0.2 A): a fixed set of "frustrated" references

The same handful of molecules leave the clean basin from the smallest noise
onward, by an essentially **constant, negative, seed-independent** amount —
the signature of a clean reference geometry that optimized to a symmetric
saddle / non-minimum, from which *any* symmetry-breaking displacement drops
deterministically into the true minimum:

| molecule | dE to clean (kcal/mol) | seeds affected (per sigma) | from sigma |
|---|---:|---:|---:|
| methylamine   | -6.24 | 6 / 6 | 0.02 |
| mesityl_oxide | -1.30 | 6 / 6 | 0.02 |
| benzidine     | -1.27 | 6 / 6 | 0.02 |
| acanil01      | -0.75 | 5-6 / 6 | 0.02 |
| caffeine      | -0.36 to -0.42 | 1-6 / 6 | 0.02 |

The standout is **methylamine: every one of its 30 noisy trials, even at
0.02 A, relaxes to a minimum 6.24 kcal/mol below the unperturbed reference.**
For these molecules the noisy runs are *more* correct than the clean run —
the noise sensitivity is exposing that the unperturbed Baker reference
optimization halts above the true minimum, not that the optimizer is
unstable. (This same fixed set accounts for essentially all of the 0.1-2
kcal/mol conformer band as well: it is the same molecules reappearing across
seeds and amplitudes, not a broadening spread. The aromatics — `benzene`,
`furan`, `naphthalene`, the fluorobenzenes — stay in the clean basin until
0.3 A, where they jump straight to the >20 kcal/mol broken structures below;
they have no intermediate conformer band.)

### High-amplitude regime (0.3 A): stochastic breakage of rigid molecules

A qualitatively new failure mode appears **only at sigma = 0.3 A** (RMS
per-atom displacement ~0.52 A): large *positive* dE of +25 to +115 kcal/mol,
on otherwise rigid molecules (benzene +93, benzidine +115, caffeine +90,
the halobenzenes +88, naphthalene +57, furan +48, histidine +97, ...). Here
the displacement is large enough to scramble the aromatic/ring connectivity,
and the optimizer converges to a broken, high-energy stationary point. These
are genuine robustness failures, but they are stochastic (only some seeds)
and confined to the most aggressive amplitude tested.

## Conclusions

1. **PyBerny's convergence is robust to physically reasonable start-geometry
   noise.** Displacements up to ~0.1 A RMS never broke convergence across 30
   molecules x 6 seeds; the only cost is a ~2-3x increase in step count.
2. **The located minimum is stable to the same noise**: ~82 % of trials
   return to the identical minimum and ~94 % to within 2 kcal/mol.
3. **Where noise *does* change the result at low amplitude, it almost always
   finds a *lower* minimum** — flagging a small set of Baker references
   (methylamine most clearly) whose unperturbed optimization stops above the
   true minimum. This is a benchmark-data observation, not an optimizer
   instability, and would be worth confirming against the reference method.
4. **Only at an extreme 0.3 A do coordinate-build errors, step-ceiling
   non-convergence, and convergence to spurious structures appear** — the
   expected breakdown when the start is displaced far enough to alter
   connectivity.

## Reproduce

```sh
scripts/noise_stability.py --benchmark baker --seeds 6 \
    --sigmas 0.02 0.05 0.1 0.2 0.3 \
    --out artifacts/baker_noise_stability.md \
    --out-json artifacts/baker_noise_stability.json
```

(~44 min wall on one core; the high-sigma trials on the larger flexible
molecules dominate. Reduce `--seeds` / drop `0.3` for a quick pass.)
