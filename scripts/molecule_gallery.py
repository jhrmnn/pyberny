#!/usr/bin/env python3
"""Generate a standalone 3D viewer page for the Birkholz-Schlegel benchmark.

Reads the benchmark ``.xyz`` geometries and ``reference.json`` metadata from
:mod:`berny.benchmarks` (``birkholz_schlegel`` subset) and emits a single
self-contained HTML file with the geometries embedded inline and 3Dmol.js
loaded from a pinned CDN. The page is built independently of Sphinx and
merged into the documentation site at deploy time (see
``.github/workflows/doc.yaml``).
"""

import argparse
import json
import sys
from pathlib import Path

from berny.benchmarks import data_dir as _bench_data_dir

DATA_DIR = _bench_data_dir('birkholz')
TEMPLATE = Path(__file__).resolve().parent / 'molecule_gallery.html'
PLACEHOLDER = '__BIRKHOLZ_DATA__'


def collect_molecules(data_dir):
    """Return the benchmark molecules as dicts sorted by display name.

    Each dict carries the metadata from ``reference.json`` plus the raw
    ``.xyz`` text under the ``xyz`` key (3Dmol parses it client-side, and the
    atom order is preserved so a label equals its 0-based array index).
    """
    data_dir = Path(data_dir)
    with open(data_dir / 'reference.json') as f:
        reference = json.load(f)
    molecules = []
    for mol_id, meta in reference.items():
        molecules.append(
            {
                'id': mol_id,
                'name': meta['name'],
                'atoms': meta['atoms'],
                'charge': meta['charge'],
                'mult': meta['mult'],
                'paper_steps': meta['paper_steps'],
                'paper_steps_method': meta['paper_steps_method'],
                'paper_steps_basis': meta['paper_steps_basis'],
                'pyberny_steps': meta['pyberny_steps'],
                'mopac_pm7_steps': meta['mopac_pm7_steps'],
                'xyz': (data_dir / f'{mol_id}.xyz').read_text(),
            }
        )
    molecules.sort(key=lambda mol: mol['name'].lower())
    return molecules


def build_html(data_dir=DATA_DIR, template=TEMPLATE):
    """Render the gallery HTML with the benchmark geometries embedded inline."""
    molecules = collect_molecules(data_dir)
    data_js = json.dumps(molecules, separators=(',', ':'))
    return Path(template).read_text().replace(PLACEHOLDER, data_js)


def main(argv=None):
    """Write the gallery HTML to the path given by ``-o/--output``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '-o', '--output', type=Path, required=True, help='output HTML path'
    )
    parser.add_argument(
        '--data-dir', type=Path, default=DATA_DIR, help='benchmark data directory'
    )
    args = parser.parse_args(argv)
    html = build_html(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f'Wrote {args.output} ({len(html)} bytes)', file=sys.stderr)


if __name__ == '__main__':
    main()
