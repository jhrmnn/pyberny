# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Standard geometry-optimization benchmark sets bundled with pyberny.

Each benchmark is a ``reference.json`` recording per-molecule metadata
(charge, multiplicity, atom count) and reference step counts from the
original paper and from pyberny itself, paired with the starting ``.xyz``
geometries it scores:

* ``'birkholz'`` -- the 19-molecule set from Birkholz & Schlegel,
  *Theor. Chem. Acc.* **135**, 84 (2016); see
  ``birkholz_schlegel/SOURCE.md``.
* ``'baker'`` -- the 30-molecule Baker test set as redistributed by
  Shajan, Manathunga, Goetz & Merz, *chemrxiv* 2023:7r7qn; see
  ``baker_shajan_2023/SOURCE.md``.
* ``'oligomers'`` -- length series of common oligomers (acenes, poly-ynes,
  PPE, peptides, …); see ``oligomers/SOURCE.md``.

The ``birkholz`` and ``baker`` geometries sit next to their
``reference.json`` in package data. The ``oligomers`` geometries instead
come from the ``external/oligomer-benchmarks`` git submodule, so only its
``reference.json`` is package data and each entry carries a ``file`` key
locating its ``.xyz`` under the submodule's geometry root. The discovery
API hides this split: every benchmark is read the same way.

Discovery API::

    from berny.benchmarks import BENCHMARKS, load_reference, iter_molecules

    for name, geom, ref in iter_molecules('birkholz'):
        ...  # drive your optimizer on `geom` and compare against `ref`

The ``BENCHMARKS`` mapping is also re-exported by ``scripts/benchmark.py``,
which adds PySCF / xTB adapters and a markdown reporting harness on top.

Resource access goes through :func:`importlib.resources.files`, mirroring
how ``species-data.csv`` is loaded elsewhere in the package, so
:func:`load_reference` and :func:`iter_molecules` work in zipimport /
packed environments as well as ordinary on-disk installs.
"""

import json
from collections.abc import Iterable, Iterator
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # ``importlib.resources.abc.Traversable`` is the canonical home as of
    # Python 3.11; the typeshed stubs only expose it under
    # ``importlib.abc``, which is deprecated for removal in 3.14 but still
    # works statically.
    from importlib.abc import Traversable

    from berny.geomlib import Geometry

_PKG = files(__package__)

#: Mapping of short benchmark names to the package subdirectory holding
#: their ``reference.json`` (and ``SOURCE.md``). The subdirectory names
#: encode provenance (paper / SI source); the short keys are the stable
#: public identifiers used by the CLI and downstream tooling. For
#: ``birkholz`` / ``baker`` this directory also holds the ``.xyz``
#: geometries; for ``oligomers`` it holds only ``reference.json`` (the
#: geometries come from the submodule -- see ``_GEOM_ROOTS``).
_SUBDIRS = {
    'birkholz': 'birkholz_schlegel',
    'baker': 'baker_shajan_2023',
    'oligomers': 'oligomers',
}

#: Repository root, used to reach the ``external/oligomer-benchmarks`` git
#: submodule. Resolved from this source file rather than from package
#: resources: the submodule is a development/CI asset that only exists in a
#: source checkout, never in a packaged (wheel / zipimport) install.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Override of the geometry root for benchmarks whose ``.xyz`` files do not
#: live next to their ``reference.json``. Absent benchmarks default to
#: their package subdirectory (``data_dir``).
_GEOM_ROOTS: 'dict[str, Traversable]' = {
    'oligomers': _REPO_ROOT / 'external' / 'oligomer-benchmarks' / 'xyz',
}


def _resolve(benchmark: str) -> str:
    try:
        return _SUBDIRS[benchmark]
    except KeyError:
        raise ValueError(
            f'unknown benchmark {benchmark!r}; valid: {sorted(_SUBDIRS)}'
        ) from None


def data_dir(benchmark: str) -> 'Traversable':
    """Return the geometry root for ``benchmark`` (where ``.xyz`` live).

    For ``birkholz`` / ``baker`` this is the package data subdirectory
    (also holding ``reference.json``); for ``oligomers`` it is the
    ``external/oligomer-benchmarks`` submodule's geometry root. Note that
    ``reference.json`` is *not* necessarily under this directory -- use
    :func:`load_reference` to read it. Library code that should work in
    zipimport / packed environments should prefer :func:`load_reference` /
    :func:`iter_molecules` over poking at the returned path directly.
    """
    _resolve(benchmark)  # validate name
    return _GEOM_ROOTS.get(benchmark) or _PKG.joinpath(_SUBDIRS[benchmark])


#: Mapping of short benchmark names to their geometry roots; back-compat
#: alias kept because ``scripts/benchmark.py`` and downstream tooling
#: import ``BENCHMARKS`` directly.
BENCHMARKS: 'dict[str, Traversable]' = {name: data_dir(name) for name in _SUBDIRS}


def load_reference(benchmark: str) -> dict[str, Any]:
    """Return the parsed ``reference.json`` for ``benchmark``."""
    text = _PKG.joinpath(_resolve(benchmark)).joinpath('reference.json').read_text()
    data: dict[str, Any] = json.loads(text)
    return data


def iter_molecules(
    benchmark: str, names: Iterable[str] | None = None
) -> Iterator[tuple[str, 'Geometry', Any]]:
    """Yield ``(name, Geometry, ref)`` triples for ``benchmark``.

    ``names`` optionally restricts and orders the iteration; the default
    is every molecule in ``reference.json``, sorted by name. Each ``.xyz``
    is located at ``ref['file']`` relative to the benchmark's geometry root
    when that key is present (the ``oligomers`` submodule layout), else at
    ``<name>.xyz`` next to ``reference.json``. Raises :class:`ValueError`
    for an unknown benchmark or unknown molecule name.
    """
    from berny import geomlib

    reference = load_reference(benchmark)
    selected = sorted(reference) if names is None else list(names)
    missing = [n for n in selected if n not in reference]
    if missing:
        raise ValueError(f'unknown molecules in {benchmark!r}: {missing}')
    base = data_dir(benchmark)
    for name in selected:
        rec = reference[name]
        xyz_text = base.joinpath(rec.get('file', f'{name}.xyz')).read_text()
        yield name, geomlib.loads(xyz_text, 'xyz'), rec


__all__ = ['BENCHMARKS', 'data_dir', 'iter_molecules', 'load_reference']
