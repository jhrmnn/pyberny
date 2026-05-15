# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for NumPy 2.
- Covalent radii for Ce–Yb, Po, At, and Fr–U from Cordero et al., *Dalton Trans.*, 2008, 2832, so that geometries containing these elements no longer crash `InternalCoords`.

### Changed

- Minimum supported Python version raised to 3.9.
- `berny.Math.FindrootException` renamed to `berny.Math.FindrootError`.
- Dropped the runtime dependency on `setuptools` (`pkg_resources`).

### Fixed

- `get_property` now raises a clear `KeyError` identifying the species and property when the requested datum is missing, instead of letting the call fail later with an opaque numpy error.
- Lookup of species data by atomic number (rather than symbol) now works correctly.
- `InternalCoord.__eq__` no longer silently returns `None`; equality and hashing now behave consistently, fixing membership checks used in dihedral-swap detection.
- The `berny` CLI no longer crashes on every step: it now passes energy and gradients to `Berny.send` as a tuple and terminates cleanly on convergence.
- Dumping a crystal `Geometry` in the `aims` format now includes `lattice_vector` lines, so the format round-trips correctly.
- `Geometry` now coerces `coords` and `lattice` to `float`, so integer inputs no longer trip the dump formatter.
- The `mopac` pytest fixture is now decorated correctly (the previous form used a function-default that pytest ignored).

## [0.6.3] - 2021-02-22

### Fixed

- CLI

[unreleased]: https://github.com/jhrmnn/pyberny/compare/0.6.3...HEAD
[0.6.3]: https://github.com/jhrmnn/pyberny/releases/tag/0.6.3
