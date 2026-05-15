# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for NumPy 2.
- Test matrix runs on Python 3.9 through 3.13.

### Changed

- Minimum supported Python version raised to 3.9.
- `berny.species_data` now uses `importlib.resources` instead of the
  deprecated `pkg_resources`.
- Renamed `berny.Math.FindrootException` to `berny.Math.FindrootError`.
- Build backend switched to `poetry_dynamic_versioning.backend` so the
  dynamic version is populated for PEP 517 builds (`pip install`,
  `python -m build`), not only for `poetry build`.

### Fixed

- CI workflows updated for current GitHub Actions, MOPAC switched to
  the open-source release at `openmopac/mopac`.

## [0.6.3] - 2021-02-22

### Fixed

- CLI

[unreleased]: https://github.com/jhrmnn/pyberny/compare/0.6.3...HEAD
[0.6.3]: https://github.com/jhrmnn/pyberny/releases/tag/0.6.3
