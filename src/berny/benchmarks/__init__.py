# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Standard geometry-optimization benchmark sets bundled with pyberny.

Two molecule sets ship with the package, each as a directory of starting
``.xyz`` geometries plus a ``reference.json`` recording per-molecule
metadata (charge, multiplicity, atom count) and reference step counts
from the original paper and from pyberny itself:

* ``'birkholz'`` -- the 19-molecule set from Birkholz & Schlegel,
  *Theor. Chem. Acc.* **135**, 84 (2016); see
  ``birkholz_schlegel/SOURCE.md``.
* ``'baker'`` -- the 30-molecule Baker test set as redistributed by
  Shajan, Manathunga, Goetz & Merz, *chemrxiv* 2023:7r7qn; see
  ``baker_shajan_2023/SOURCE.md``.

Discovery API::

    from berny.benchmarks import BENCHMARKS, load_reference, iter_molecules

    for name, geom, ref in iter_molecules('birkholz'):
        ...  # drive your optimizer on `geom` and compare against `ref`

The ``BENCHMARKS`` mapping is also re-exported by ``scripts/benchmark.py``,
which adds PySCF / MOPAC adapters and a markdown reporting harness on top.

Resource access goes through :func:`importlib.resources.files`, mirroring
how ``species-data.csv`` is loaded elsewhere in the package, so
:func:`load_reference` and :func:`iter_molecules` work in zipimport /
packed environments as well as ordinary on-disk installs.
"""

import json
from importlib.resources import files

_PKG = files(__package__)

#: Mapping of short benchmark names to the on-disk subdirectory name
#: holding their geometries and ``reference.json``. The subdirectory
#: names encode provenance (paper / SI source); the short keys are the
#: stable public identifiers used by the CLI and downstream tooling.
_SUBDIRS = {
    'birkholz': 'birkholz_schlegel',
    'baker': 'baker_shajan_2023',
}

#: Mapping of short benchmark names to their data directories, resolved
#: via :func:`importlib.resources.files`. For ordinary file-system
#: installs the values are :class:`pathlib.Path` objects supporting
#: ``/`` / ``read_text()`` / ``exists()`` as usual.
BENCHMARKS = {name: _PKG.joinpath(sub) for name, sub in _SUBDIRS.items()}


def _resolve(benchmark):
    try:
        return _SUBDIRS[benchmark]
    except KeyError:
        raise ValueError(
            f'unknown benchmark {benchmark!r}; valid: {sorted(_SUBDIRS)}'
        ) from None


def data_dir(benchmark):
    """Return the on-disk path to ``benchmark``'s data directory.

    Returned object is whatever :func:`importlib.resources.files` resolves
    to for the package; for ordinary file-system installs this is a
    :class:`pathlib.Path`. Library code that should work in zipimport /
    packed environments should prefer :func:`load_reference` /
    :func:`iter_molecules` (which use resource ``read_text``) instead.
    """
    return _PKG.joinpath(_resolve(benchmark))


def load_reference(benchmark):
    """Return the parsed ``reference.json`` for ``benchmark``."""
    text = _PKG.joinpath(_resolve(benchmark), 'reference.json').read_text()
    return json.loads(text)


def iter_molecules(benchmark, names=None):
    """Yield ``(name, Geometry, ref)`` triples for ``benchmark``.

    ``names`` optionally restricts and orders the iteration; the default
    is every molecule in ``reference.json``, sorted by name. Raises
    :class:`ValueError` for an unknown benchmark or unknown molecule name.
    """
    from berny import geomlib

    sub = _resolve(benchmark)
    reference = load_reference(benchmark)
    selected = sorted(reference) if names is None else list(names)
    missing = [n for n in selected if n not in reference]
    if missing:
        raise ValueError(f'unknown molecules in {benchmark!r}: {missing}')
    base = _PKG.joinpath(sub)
    for name in selected:
        xyz_text = base.joinpath(f'{name}.xyz').read_text()
        yield name, geomlib.loads(xyz_text, 'xyz'), reference[name]


__all__ = ['BENCHMARKS', 'data_dir', 'iter_molecules', 'load_reference']
