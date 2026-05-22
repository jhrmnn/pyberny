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
- Linear-bend internal coordinates via dummy ("ghost") atoms (issue #30). Near-linear triples `i-j-k` (angle > 175°) now place two mutually orthogonal dummy atoms perpendicular to the `i-k` axis and replace the singular `Angle(i,j,k)` with four well-behaved bends through ≈90°. This fixes optimization failures for molecules containing triple bonds (acetylenes, nitriles, CO₂) reported in issue #23. Dummy positions live in `InternalCoords.dummy_atoms` and are refreshed from the real-atom coordinates on every step; the `Geometry` yielded by `Berny` should be treated as immutable by callers.
- `Ghost`, `X`, and `Bq` species (and any name with a leading `-`) are now recognised as basis-function-only centres with zero covalent radius (issue #9). Geometries containing such atoms — common in PySCF/ASE workflows — no longer crash `InternalCoords`.

### Changed

- Minimum supported Python version raised to 3.9.
- `berny.Math.FindrootException` renamed to `berny.Math.FindrootError`.
- Dropped the runtime dependency on `setuptools` (`pkg_resources`).
- Unknown keyword arguments to `Berny()` now raise `TypeError` instead of being silently absorbed.

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
