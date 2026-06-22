# Step-count stability of pyberny trajectories under small start-geometry noise

*2026-06-22 — follow-up to the
[`2026-06-20-baker-noise-stability`](../2026-06-20-baker-noise-stability) report.*

The parent study established that pyberny's *located minimum* is stable to small
Cartesian noise in the start geometry (≈82 % of trials return to the same
minimum, ≈94 % within 2 kcal/mol). It left a separate question open: even when a
slightly perturbed start converges to the **same** minimum, is the **number of
optimization steps** stable, or does the trajectory length scatter?

This matters because the benchmark grades pyberny on step counts (`pyberny_steps`
/ `xtb_gfn2_steps`), so step-count reproducibility under tiny geometric
perturbations is exactly the kind of robustness the regression gate assumes.

## Method

For every Baker molecule **except the seven "frustrated reference" molecules** —
`methylamine`, `mesityl_oxide`, `benzidine`, `acanil01`, `caffeine`, `ethanol`,
`histidine` — whose unperturbed start sits on/near a symmetric saddle and which
therefore relax to a *different* minimum under any noise (see the
[`baker-symmetry-saddle`](../2026-06-20-baker-symmetry-saddle) and
[`baker-ethanol-histidine-conformer`](../2026-06-20-baker-ethanol-histidine-conformer)
reports and [pyberny#148](https://github.com/jhrmnn/pyberny/issues/148)):

1. optimize the clean start (GFN2-xTB) → reference step count `N0`;
2. optimize **30 noisy copies** at each amplitude σ ∈ {0.02, 0.05, 0.1} Å (RMS
   per Cartesian coordinate);
3. keep only trials that converge to the **same minimum** (final energy within
   0.1 kcal/mol of the clean run) and record their step counts.

23 molecules × 3 amplitudes × 30 seeds. Scripts in `scripts/`, raw data in
`data/step_stability.json`, figure `step_count_stability.png`.

![Step-count stability](step_count_stability.png)

## Result: stable per molecule, but noisier than the minimum

**Conditioned on the molecule and on reaching the same minimum, the step count
is tight and bounded:**

| σ (Å) | median per-molecule CV | trials within ±25 % of molecule median |
|---:|---:|---:|
| 0.02 | 10 % | 93 % |
| 0.05 | 11 % | 92 % |
| 0.1  | 15 % | 91 % |

Nothing approaches the 100-step ceiling and there is no heavy tail — even the
worst per-molecule outliers stay ≤1.9× that molecule's median (right panel:
steep ECDFs centred on 1.0). So the *trajectory length* is reproducible to
roughly ±10–15 %, materially noisier than the *minimum* (reproducible to
<0.1 kcal/mol) but still well-behaved.

Two things qualify "stable":

### 1. A systematic inflation off the pre-relaxed starts (not instability)

The Baker start geometries are already very close to their minimum, so the clean
runs are unusually short (3–7 steps for most molecules). Any displacement throws
away that head start, so perturbed runs cost a **near-constant ~2–3× more steps**
— with *tight* spread, i.e. a multiplicative offset, not scatter. Examples at
σ = 0.05 (clean → median): `acetone` 5 → 20 (4.0×), `benzene` 3 → 11 (3.7×),
`naphthalene` 4 → 12 (3.0×), `neopentane` 3 → 8 (2.7×). The lone inversion is
**`achtar10`** (clean = 30, the slowest converger): noise *shortens* it to a
median of 24 (0.8×) — a small nudge knocks it off its long, grinding descent.
Molecules whose clean start is not pre-relaxed to the same degree
(`hydroxybicyclopentane_2`, `hydroxysulfane` ≈1.0×; `menthone` 1.1×) barely
inflate at all.

### 2. A soft-PES minority that genuinely scatters more

Four molecules show CV ≈ 24–30 % at σ = 0.05, with ranges spanning roughly a
factor of two:

| molecule | clean | σ=0.05 median | range | CV |
|---|---:|---:|---:|---:|
| disilyl_ether | 16 | 27 | 16–44 | 30 % |
| acetone | 5 | 20 | 11–30 | 27 % |
| pterin | 7 | 20 | 13–37 | 24 % |
| achtar10 | 30 | 24 | 10–37 | 24 % |

These have flat / soft regions where the path length is genuinely sensitive to
where the perturbed start lands, but even their worst trajectory is only
~1.5–1.9× the median — no blow-up, no non-convergence.

**Amplitude dependence** is mild: the median per-molecule CV grows only from
10 % to 15 % as σ goes 0.02 → 0.1 Å, so the picture is robust across the
small-noise regime.

## Conclusion

Once the saddle-point (frustrated-reference) molecules are excluded and trials
are conditioned on reaching the same minimum, pyberny's trajectories **are
step-count stable** — per-molecule CV ≈ 10–15 %, ~92 % of runs within ±25 % of
the molecule's median, bounded with no ceiling tail. The two caveats are an
*expected* ~2–3× inflation off the benchmark's unusually short pre-relaxed
starts (a tight multiplicative offset, not instability), and a handful of
soft-surface molecules (`disilyl_ether`, `acetone`, `pterin`, `achtar10`) whose
step count is ~25–30 % variable. This is consistent with the regression gate's
7 %-with-a-2-step-floor tolerance being applied to a *fixed* start geometry: the
step counter is reproducible for a given start, and only loosely so across
geometric perturbations of it.

## Reproduce

```sh
# ~30 min, 1 core (small-sigma trajectories are short and fast)
python scripts/step_stability.py          # writes step_stability.json
python scripts/plot_step_stability.py step_stability.json step_count_stability.png
```

Requires pyberny installed from a checkout (`pip install -e ".[benchmark]"`).
