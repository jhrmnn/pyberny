# The sphere-restricted convergence gate: false negatives at flat minima

## Motivation

Issue #129 (from the #128 oligomer-benchmark dissection) identifies the
`Minimization on sphere` convergence gate as the proximate cause of about
half the unconverged MOPAC PM7 runs in the oligomer benchmark. When the
quadratic step is truncated to the trust-region sphere, `is_converged`
(`src/berny/berny.py`) used to replace the two displacement criteria with a
single hard-coded `('Minimization on sphere', False)`, so a run **could not**
be declared converged while pinned to the sphere — no matter how small the
gradient was.

The original #129 framing blamed *PM7 energy noise*: near a flat minimum the
trust radius sawtooths (collapse → regrow → collapse) and keeps the step on the
sphere indefinitely. After #132, the noise was traced and **#136 largely
eliminated it** — `MopacSolver` now reads the high-precision `AUX` file
(energy to 15 sig figs, ~5e-12 Ha vs. the `.out` file's ~1.6e-8 Ha print
grid) and the benchmark drops its PM7-specific `energy_noise=2e-7` override.

This experiment was **re-run on top of #136** to answer: did removing the
noise resolve the false negatives, or do they persist? It (1) reproduces the
false negative under the new AUX solver, (2) compares the two fixes proposed
in #129 against the optimizer's *actual* behaviour, and (3) checks for
regressions on the bundled `birkholz`/`baker` benchmarks (whose references
were reseeded by #136).

## Headline: the noise fix did **not** resolve it

**The sphere-lock persists, essentially unchanged, with the noise gone** —
and the lower noise floor makes it *more* prevalent, because the trust radius
now collapses even harder near a flat minimum:

- `polyserine_n5` still spends **98 %** of its 130 steps on the sphere and
  never converges under the old gate; the trust radius still sawtooths (now
  down to ~3e-5). Identical to the pre-#136 behaviour.
- **It surfaced on a bundled benchmark.** #136's own changelog notes that
  `azadirachtin` "no longer converges within the step ceiling (its flat
  minimum locks onto the trust-region sphere once the noise floor drops)" and
  reseeds its reference to `null`. This re-run reproduces that: `azadirachtin`
  runs all 130 steps at 78 % on-sphere with a **final gradient of
  0.01× / 0.02×** the threshold — a textbook false negative, now on a molecule
  that ships in the wheel.

So the root cause is **not energy noise** but the trust-region controller
collapsing on a genuinely flat PES where the quadratic model poorly predicts
sub-µHa energy changes. The convergence-gate fix addresses it regardless of
the noise level; #136 and this fix are orthogonal and complementary.

## The bug, and a three-way contradiction

The old gate:

```python
criteria = [('Gradient RMS', ...), ('Gradient maximum', ...)]
if on_sphere:
    criteria.append(('Minimization on sphere', False))   # hard block
else:
    criteria.extend([('Step RMS', ...), ('Step maximum', ...)])
```

Three sources of truth disagreed about what *should* happen on a sphere step:

| Source | Behaviour on a sphere-restricted step |
|---|---|
| **The code** (`is_converged`) | hard-blocked — never converges |
| **The docs** (`doc/standard_method.rst`) | "skips the step-based criteria and demands the gradient-based criteria only" |
| **The SM itself** (Birkholz–Schlegel / Gaussian) | all four criteria, the displacement ones tested against the *actual* step taken |

The fix makes the code follow the SM: the four standard criteria are always
tested, with the displacement criteria evaluated against the actual
(trust-limited) step. On a *wide-trust* sphere step the displacement is large
and fails by construction — exactly the SM intent that a converged minimum sit
at an interior step. On a *collapsed-trust* sphere step (the flat-minimum case)
the displacement is tiny and the four criteria are genuinely met, so the run
converges.

## Method

- MOPAC PM7 via the post-#136 `MopacSolver` (AUX file), `Berny(maxsteps=130)`
  with the default `energy_noise=2e-8` — i.e. exactly what `scripts/benchmark.py`
  now uses.
- Geometries: the [ghutchis/oligomer-benchmarks](https://github.com/ghutchis/oligomer-benchmarks)
  set from #127 (the 14 ceiling cases from #128), plus the bundled `birkholz`
  (19) and `baker` (30) sets for the regression check.
- Per-step structured traces via `Berny(trace=...)`.

**Why counterfactual analysis on baseline traces is exact here.** Both
candidate fixes change *only* the return value of `is_converged`; they touch
no step computation. The trajectory — geometries, Hessian, trust radius — is
therefore identical to baseline up to the step a fix first declares
convergence, so reading the baseline (old-gate) traces predicts exactly where
each candidate stops. Confirmed end-to-end: the patched four-criteria
optimizer stops `polyserine_n5` at step 50, the step its baseline trace
predicts.

Two candidates from #129 were compared:

- **gradient-only** — on a sphere step require only the two gradient criteria
  (what the docs claimed PyBerny already did).
- **four-criteria** — always test all four, displacement against the actual
  step (the fix shipped here).

## Results

### 1. The false negative persists under the AUX solver

![polyserine_n5 still locked on the trust sphere](polyserine_n5_trace.png)

With the noise essentially gone, `polyserine_n5` behaves just as it did before
#136. After ~step 46 the energy is pinned at its minimum and **both** gradient
norms sit below threshold for the rest of the run, but the trust radius
sawtooths and keeps **98 %** of steps on the sphere, so the old gate never
converges and burns all 130 steps. The four-criteria fix converges at **step
50** (gradient-only at 46) — a ~2.6× reduction for an identical result. That
the picture is unchanged with precise energies is the point: this is a
trust-controller/PES-flatness effect, not a noise artefact.

### 2. Four-criteria converges true minima and rejects oscillators — gradient-only does not

![Where each rule declares convergence on the ceiling cases](oligomer_outcomes.png)

The AUX solver **also fixes the old `KCAL/ANGSTROM` gradient-parser crash**
(#128), so `nylon6_n7`/`nylon6_n8` now run to completion (and turn out to be
genuine oscillators). One case, `polyalanine_n3`, hits a *new* edge case — a
`ZeroDivisionError` once the trust collapses to ~6e-6, a consequence of the
precise energies driving the trust far smaller than the old print grid allowed;
it is a genuinely non-converging (gradient never met) floppy case, separate
from #129. On the remaining 13 ceiling cases:

- **four-criteria converges the genuine minima** — `polyserine_n5` (50),
  `polyglycine_n7` (35), `polyserine_n8` (114), `polyglycine_n4` (120) — each
  at a step where all four criteria hold.
- **It correctly leaves the oscillators at the ceiling** — `nylon6_n6/n7/n8`,
  `polyglycine_n8`, `polyserine_n3/n7`, `polyalanine_n7` end 2–31× over the
  max-gradient threshold (genuinely hopping between conformers; out of scope).
- **gradient-only would prematurely "converge" oscillators** — e.g.
  `polyalanine_n7` at step 17 while it ends 5× over threshold — because their
  gradient momentarily dips on a *large* on-sphere step. The displacement
  criteria block exactly these.

The displacement criteria are what separate "settled at a minimum" from
"passing through with a small instantaneous gradient on a big step." Dropping
them (the gradient-only rule the docs described) is unsafe; keeping them,
tested against the actual step, is the literal SM and what the fix does.

### 3. Regression check on the bundled benchmarks

Counterfactual over all 19 `birkholz` + 30 `baker` molecules under the AUX
solver, compared against master's reseeded references. Every change is toward
**convergence at a genuine minimum** (final gradient well below threshold); no
molecule converges away from its minimum and none that converged now fails:

| molecule | master ref | four-criteria | final ∇ (×max thr) | note |
|---|---:|---:|---:|---|
| `azadirachtin` (birkholz) | `null` | **66** | 0.02× | recovery; = its pre-#136 reference of 66 |
| `caffeine` (baker) | `null` | **12** | 0.10× | recovery |
| `raffinose` (birkholz) | `null` | **86** | 0.17× | recovery (borderline; PM7-run-dependent) |
| `penicillin_v` (birkholz) | 54 | 50 | 0.03× | −4 steps, same minimum |
| `bisphenol_a` (birkholz) | 45 | 43 | 0.06× | −2 steps, same minimum |

So the four-criteria fix **recovers three bundled non-convergers**
(`azadirachtin`, `caffeine`, `raffinose` are all `null` in master) and shortens
two more by a few steps. All other benchmark molecules are unchanged.

By contrast the gradient-only rule would shift **16** benchmark molecules
(10 birkholz + 6 baker), many *much* earlier on wide on-sphere steps —
`azadirachtin` 38 (vs the true 66), `bisphenol_a` 11, `benzidine` 8,
`acanil01` 10 — i.e. away from the settled minimum.

**Implication for merging.** Because #136 lowered the noise floor, the
four-criteria fix now changes a handful of bundled step counts (above), unlike
the pre-#136 run where it changed none. These are improvements (three
recoveries, two slightly shorter), but adopting the fix means regenerating
`mopac_pm7_steps` for the affected molecules. The exact counts are
PM7-runner-dependent (see `birkholz_schlegel/SOURCE.md`) — `caffeine`/
`raffinose` even converged under the *old* gate on this runner but are `null`
on master's — so they should be reseeded from a CI benchmark run, not from
these local numbers. (PR-time CI does not gate on step counts; the per-row
benchmark check runs only on manual `workflow_dispatch`.)

## Conclusions

1. **#136 did not resolve #129.** Removing the energy noise left the
   sphere-lock false negatives intact (`polyserine_n5` unchanged) and exposed a
   fresh one on a bundled molecule (`azadirachtin`). The cause is the
   trust-region controller on a flat PES, not noise — so the convergence-gate
   fix is needed independently of #136 and is now demonstrable without the
   external oligomer submodule.
2. **The hard-block is a deviation from the SM.** The fix replaces it with the
   literal four-criterion test (displacement against the actual step), which
   recovers `azadirachtin` (at its pre-#136 step count of 66) and the
   `polyserine_n5`-style cases.
3. **Four-criteria beats gradient-only.** Gradient-only (what the docs
   described) prematurely converges oscillators on wide on-sphere steps; the
   displacement test prevents it.
4. **The fix only ever converges *earlier*, never away from a minimum.** On a
   wide-trust sphere step it behaves exactly like the old hard-block. With the
   AUX solver it shifts a few bundled step counts (all improvements), which
   should be reseeded on merge.
5. **Out of scope (real, separate problems).** Genuine conformer oscillation
   (recurrent negative Hessian eigenvalue); the trust-radius sawtooth itself
   (#129 direction 2 — still present, just no longer noise-driven); and the new
   `ZeroDivisionError` when the trust collapses to ~1e-6 under precise energies
   (`polyalanine_n3`), a solver/optimizer-robustness issue distinct from the
   convergence gate.

## Reproduction

Clone [ghutchis/oligomer-benchmarks](https://github.com/ghutchis/oligomer-benchmarks),
then for each molecule run `Berny(geom, maxsteps=130)` (default `energy_noise`)
against the post-#136 `berny.solvers.MopacSolver` (PM7, AUX) with
`trace=<path>`. From each old-gate trace, a candidate rule's convergence step
is the first step whose `convergence.criteria` satisfy that rule (gradient-only:
the two gradient entries; four-criteria: all four, the step criteria read from
`total_step`). The `birkholz`/`baker` regression is the same counterfactual via
`berny.benchmarks.iter_molecules`. Needs `mopac` on `$PATH`; ~30 min
single-threaded. The two figures here are the record of the run; the driver and
raw traces are not committed.
