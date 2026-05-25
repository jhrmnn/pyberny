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
"""

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: Mapping of short benchmark names to the on-disk subdirectory holding
#: their geometries and ``reference.json``. The subdirectory names encode
#: provenance (paper / SI source); the short keys are the stable public
#: identifiers used by the CLI and downstream tooling.
BENCHMARKS = {
    'birkholz': _HERE / 'birkholz_schlegel',
    'baker': _HERE / 'baker_shajan_2023',
}


def data_dir(benchmark):
    """Return the filesystem path to ``benchmark``'s data directory."""
    try:
        return BENCHMARKS[benchmark]
    except KeyError:
        raise ValueError(
            f'unknown benchmark {benchmark!r}; valid: {sorted(BENCHMARKS)}'
        ) from None


def load_reference(benchmark):
    """Return the parsed ``reference.json`` for ``benchmark``."""
    return json.loads((data_dir(benchmark) / 'reference.json').read_text())


def iter_molecules(benchmark, names=None):
    """Yield ``(name, Geometry, ref)`` triples for ``benchmark``.

    ``names`` optionally restricts and orders the iteration; the default
    is every molecule in ``reference.json``, sorted by name.
    """
    from berny import geomlib

    reference = load_reference(benchmark)
    selected = sorted(reference) if names is None else list(names)
    missing = [n for n in selected if n not in reference]
    if missing:
        raise ValueError(f'unknown molecules in {benchmark!r}: {missing}')
    d = data_dir(benchmark)
    for name in selected:
        yield name, geomlib.readfile(str(d / f'{name}.xyz')), reference[name]


__all__ = ['BENCHMARKS', 'data_dir', 'load_reference', 'iter_molecules']
