#!/usr/bin/env python3
"""Split the Birkholz-Schlegel 2016 SI text file into per-molecule .xyz files.

Run once by a maintainer; the resulting files in
``src/berny/benchmarks/birkholz_schlegel/`` are committed and the SI .txt
itself is not. The SI is the electronic supplementary material of Birkholz &
Schlegel, *Theor. Chem. Acc.* **135**, 84 (2016), doi:10.1007/s00214-016-1847-3.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


def slugify(name):
    """Lowercase ASCII slug with underscores: ``Mg Porphin`` -> ``mg_porphin``."""
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_').lower()


def parse_si(text):
    """Yield ``(name, charge, mult, atoms)`` for each molecule block."""
    blocks = re.split(r'\n-+\n', text)
    for block in blocks[1:]:  # skip header block
        lines = [ln for ln in block.splitlines() if ln.strip()]
        name = lines[0].strip()
        charge, mult = map(int, lines[1].split())
        atoms = []
        for ln in lines[2:]:
            parts = ln.split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        yield name, charge, mult, atoms


def write_xyz(path, name, atoms):
    """Write a standard single-geometry XYZ file."""
    with path.open('w') as f:
        f.write(f'{len(atoms)}\n')
        f.write(f'{name} (Birkholz & Schlegel, TCA 135:84, 2016)\n')
        for el, x, y, z in atoms:
            f.write(f'{el:<2s} {x:18.10f} {y:18.10f} {z:18.10f}\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('si_txt', type=Path, help='path to 214_2016_1847_MOESM1_ESM.txt')
    ap.add_argument(
        '--out',
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / 'src'
        / 'berny'
        / 'benchmarks'
        / 'birkholz_schlegel',
        help='output directory (default: src/berny/benchmarks/birkholz_schlegel)',
    )
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    text = args.si_txt.read_text()

    # Disambiguate slugs that would collide (e.g. neutral vs cation).
    slug_overrides = {'Inosine': 'inosine_cation'}

    reference = {}
    for name, charge, mult, atoms in parse_si(text):
        slug = slug_overrides.get(name, slugify(name))
        xyz_path = args.out / f'{slug}.xyz'
        write_xyz(xyz_path, name, atoms)
        reference[slug] = {
            'name': name,
            'atoms': len(atoms),
            'charge': charge,
            'mult': mult,
            'paper_steps': None,
            'paper_steps_method': 'HF',
            'paper_steps_basis': '3-21G',
            'mopac_pm7_steps': None,
            'pyberny_steps': None,
        }
        print(
            f'wrote {xyz_path.name}  ({len(atoms)} atoms, q={charge:+d}, 2S+1={mult})'
        )

    ref_path = args.out / 'reference.json'
    preserved_keys = (
        'paper_steps',
        'paper_steps_method',
        'paper_steps_basis',
        'mopac_pm7_steps',
        'pyberny_steps',
    )
    if ref_path.exists():
        existing = json.loads(ref_path.read_text())
        for slug, ref in reference.items():
            for key in preserved_keys:
                if key in existing.get(slug, {}):
                    ref[key] = existing[slug][key]
    ref_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + '\n')
    print(f'wrote {ref_path.name}  ({len(reference)} molecules)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
