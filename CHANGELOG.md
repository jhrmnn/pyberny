# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for NumPy 2.
- Covalent radii for Ce–Yb, Po, At, and Fr–U from Cordero et al., *Dalton Trans.*, 2008, 2832, so that geometries containing these elements no longer crash `InternalCoords`.
- New `berny.BernyParams` dataclass listing every tunable optimizer parameter; useful for discovery and type-checked construction.
- `berny.solvers.MopacSolver` now accepts `charge` and `mult` keyword arguments so charged or open-shell systems no longer have to be patched in by hand.
- Opt-in benchmark suite (`scripts/benchmark.py`) reproducing 19 of the 20 molecules from Birkholz & Schlegel, *Theor. Chem. Acc.* **135**, 84 (2016); PySCF runs are driven through PySCF's own `pyscf.geomopt.berny_solver` bridge.
- Linear-bend internal coordinates via dummy ("ghost") atoms (issue #30). Near-linear triples `i-j-k` (angle > 175°) now place two mutually orthogonal dummy atoms perpendicular to the `i-k` axis and replace the singular `Angle(i,j,k)` with four well-behaved bends through ≈90°. This fixes optimization failures for molecules containing triple bonds (acetylenes, nitriles, CO₂) reported in issue #23. Dummy positions live in `InternalCoords.dummy_atoms` and are refreshed from the real-atom coordinates on every step; the `Geometry` yielded by `Berny` should be treated as immutable by callers. The optimizer additionally rebuilds the internal-coordinate set on the fly when an sp-like triple crosses the linear threshold mid-run (175° to enter, 170° to exit), so molecules that *become* linear during optimization (e.g. a bent CO₂ relaxing toward 180°) get the same dummy-atom treatment as molecules that start linear.
- `Ghost`, `X`, and `Bq` species (and any name with a leading `-`) are now recognised as basis-function-only centres with zero covalent radius (issue #9). Geometries containing such atoms — common in PySCF/ASE workflows — no longer crash `InternalCoords`.
- New `berny.tests` subpackage shipping reusable, optimizer-agnostic end-to-end tests built on analytic model potentials whose minima are known in closed form, so any optimizer (not just `Berny`) can install pyberny and check itself via `run_and_check(potential, minimize)`. The bundled `LinearBendCrossover` and `DihedralFromLinear` potentials exercise the linear-bend / dihedral coordinate handoff in both directions.
- New `berny.benchmarks` subpackage shipping the Birkholz–Schlegel and Baker (Shajan-2023) benchmark sets — starting geometries plus `reference.json` metadata — directly inside the wheel, with a small discovery API (`BENCHMARKS`, `data_dir`, `load_reference`, `iter_molecules`) so downstream optimizers can drive themselves through the same standard sets that `scripts/benchmark.py` uses.
- Interactive 3D viewer of the Birkholz & Schlegel benchmark molecules, published with the documentation, showing each starting geometry with atoms labelled by their 0-based index.
- PEP 561 `py.typed` marker — pyberny now ships type information, and the codebase is fully annotated and checked under `mypy --strict`.

### Changed

- Minimum supported Python version raised to 3.10.
- `berny.Math.FindrootException` renamed to `berny.Math.FindrootError`.
- Dropped the runtime dependency on `setuptools` (`pkg_resources`).
- Unknown keyword arguments to `Berny()` now raise `TypeError` instead of being silently absorbed.
- Mid-run internal-coordinate rebuilds now preserve accumulated Hessian curvature for surviving coordinates instead of restarting entirely from a diagonal guess.
- Linear-bend mid-run rebuild now also fires on near-linear angles at higher-coordination centres (not only sp-like ones); the singular angle is dropped rather than replaced by dummies and the dependent dihedrals are reconstructed against the straightened geometry. Fixes a class of estradiol / zn_edta optimization failures (pinv warnings, trust-radius crash) at a small step-count cost on cases where the offending angle was already stable. `mopac_pm7_steps` references updated for estradiol (11→27), azadirachtin (60→66), zn_edta (100→119), acanil01 (44→40), and mesityl_oxide (8→12) accordingly; benchmark MOPAC ceiling raised from 110 to 130 steps to accommodate zn_edta's longer trajectory.
- `berny.solvers.MopacSolver` now reads MOPAC's `AUX` file instead of the human-readable `.out` file. The `.out` heat of formation is printed only to `1e-5 kcal/mol` (a ~1.6e-8 Ha grid); the `AUX` file carries the same energy and the gradients to 15 significant figures, lowering the solver's effective noise floor by ~1000× (to ~5e-12 Ha). This changes some PM7 optimization trajectories slightly.

### Removed

- The module-level `berny.berny.defaults` dict. Use `berny.BernyParams` (or pass overrides as keyword arguments to `Berny()`) instead.

### Fixed

- `get_property` now raises a clear `KeyError` identifying the species and property when the requested datum is missing, instead of letting the call fail later with an opaque numpy error.
- Lookup of species data by atomic number (rather than symbol) now works correctly.

## [0.6.3] - 2021-02-22

### Fixed

- CLI

[unreleased]: https://github.com/jhrmnn/pyberny/compare/0.6.3...HEAD
[0.6.3]: https://github.com/jhrmnn/pyberny/releases/tag/0.6.3
