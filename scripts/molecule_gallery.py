#!/usr/bin/env python3
"""Generate a standalone 3D viewer page for the bundled benchmark sets.

Reads the benchmark ``.xyz`` geometries and ``reference.json`` metadata from
:mod:`berny.benchmarks` and emits a single self-contained HTML file with the
geometries of every available benchmark embedded inline and 3Dmol.js loaded
from a pinned CDN. The page offers a drop-down to switch between benchmarks
(``birkholz``, ``baker``, ``oligomers``). It is built independently of Sphinx
and merged into the documentation site at deploy time (see
``.github/workflows/doc.yaml``).

The ``oligomers`` geometries come from the ``external/oligomer-benchmarks`` git
submodule and are not shipped in the wheel; that benchmark is silently omitted
when its geometries are not checked out (see :mod:`berny.benchmarks`).
"""

import argparse
import json
import sys
from pathlib import Path

from berny.benchmarks import (
    geometries_available,
    load_reference,
    require_geometries,
)

TEMPLATE = Path(__file__).resolve().parent / 'molecule_gallery.html'
PLACEHOLDER = '__GALLERY_DATA__'

#: Benchmarks shown in the gallery, in drop-down order, each with the
#: human-readable label and one-line provenance blurb rendered in the header.
BENCHMARKS = {
    'birkholz': {
        'label': 'Birkholz & Schlegel',
        'description': (
            'Starting geometries of the benchmark set from Birkholz &amp; '
            'Schlegel, <em>Theor. Chem. Acc.</em> <strong>135</strong>, 84 '
            '(2016).'
        ),
    },
    'baker': {
        'label': 'Baker (Shajan et al.)',
        'description': (
            "Baker's classic 30-molecule test set as redistributed by Shajan, "
            'Manathunga, Goetz &amp; Merz, <em>chemRxiv</em> 2023:7r7qn.'
        ),
    },
    'oligomers': {
        'label': 'Oligomers',
        'description': (
            'Chain-length series of common oligomers (acenes, poly-ynes, PPE, '
            'polythiophene, peptides, and commodity polymers).'
        ),
    },
}


def _dash(value):
    """Render ``None``/missing numeric metadata as an em dash."""
    return '—' if value is None else str(value)


def _molecule_fields(meta):
    """Return the ``[label, value]`` metadata rows to show for one molecule.

    The benchmarks share a core schema (atoms / charge / multiplicity / xTB
    step count) but differ in the rest: ``birkholz`` and ``baker`` carry the
    paper's reference step count plus PyBerny's own, while ``oligomers`` carry
    a chemical ``family`` instead. Each row is included only when the backing
    field is present so the viewer stays a dumb renderer.
    """
    fields = [
        ['Atoms', _dash(meta.get('atoms'))],
        ['Charge', _dash(meta.get('charge'))],
        ['Multiplicity', _dash(meta.get('mult'))],
    ]
    if 'family' in meta:
        fields.append(['Family', meta['family']])
    if 'paper_steps' in meta:
        steps = _dash(meta.get('paper_steps'))
        method = meta.get('paper_steps_method')
        basis = meta.get('paper_steps_basis')
        if method and basis:
            steps += f' ({method}/{basis})'
        fields.append(['Paper steps', steps])
    if 'pyberny_steps' in meta:
        fields.append(['PyBerny steps', _dash(meta.get('pyberny_steps'))])
    fields.append(['xTB GFN2 steps', _dash(meta.get('xtb_gfn2_steps'))])
    return fields


def _sorted_items(benchmark, reference):
    """Order a benchmark's ``reference.json`` entries for display.

    ``birkholz`` / ``baker`` molecules are named, so they sort by name.
    ``oligomers`` are unnamed chain-length sweeps; grouping them by ``family``
    and then atom count keeps each series contiguous and increasing in size.
    """
    items = list(reference.items())
    if benchmark == 'oligomers':
        items.sort(key=lambda kv: (kv[1].get('family', ''), kv[1]['atoms'], kv[0]))
    else:
        items.sort(key=lambda kv: kv[1].get('name', kv[0]).lower())
    return items


def collect_molecules(benchmark):
    """Return one benchmark's molecules as display dicts in viewer order.

    Each dict carries the display ``name``, the raw ``.xyz`` text (3Dmol parses
    it client-side, and the atom order is preserved so a label equals its
    0-based array index), and the ``fields`` metadata rows from
    :func:`_molecule_fields`.
    """
    reference = load_reference(benchmark)
    base = require_geometries(benchmark)
    molecules = []
    for mol_id, meta in _sorted_items(benchmark, reference):
        rel = meta.get('file', f'{mol_id}.xyz')
        molecules.append(
            {
                'id': mol_id,
                'name': meta.get('name', mol_id),
                'fields': _molecule_fields(meta),
                'xyz': base.joinpath(rel).read_text(),
            }
        )
    return molecules


def build_payload(benchmarks=BENCHMARKS):
    """Return the ``{order, benchmarks}`` data embedded into the gallery.

    Benchmarks whose geometries are not on disk (``oligomers`` without its
    submodule) are skipped so the gallery still builds from a packaged
    install.
    """
    order = []
    data = {}
    for name, info in benchmarks.items():
        if not geometries_available(name):
            print(f'Skipping {name!r}: geometries not available', file=sys.stderr)
            continue
        order.append(name)
        data[name] = {
            'label': info['label'],
            'description': info['description'],
            'molecules': collect_molecules(name),
        }
    return {'order': order, 'benchmarks': data}


def build_html(template=TEMPLATE):
    """Render the gallery HTML with every available benchmark embedded inline."""
    payload = build_payload()
    data_js = json.dumps(payload, separators=(',', ':'))
    return Path(template).read_text(encoding='utf-8').replace(PLACEHOLDER, data_js)


def main(argv=None):
    """Write the gallery HTML to the path given by ``-o/--output``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '-o', '--output', type=Path, required=True, help='output HTML path'
    )
    args = parser.parse_args(argv)
    html = build_html()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f'Wrote {args.output} ({len(html)} bytes)', file=sys.stderr)


if __name__ == '__main__':
    main()
