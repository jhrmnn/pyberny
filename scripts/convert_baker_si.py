#!/usr/bin/env python3
"""Convert the Shajan-2023 SI ZIP into per-molecule .xyz files + reference.json.

Run once by a maintainer; the resulting files in
``src/berny/benchmarks/baker_shajan_2023/`` are committed and the SI ZIP
itself is not. The SI is the electronic supplementary material of Shajan,
Manathunga, Goetz & Merz, *Geometry Optimization: A Comparison of Different
Open-Source Geometry Optimizers*, chemrxiv 2023, doi:10.26434/chemrxiv-2023-7r7qn-v4.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

# Mapping from the SI's filename (under ``quick-geopt-si/ASE_inputs/`` in the
# zip) to the project's lowercase-with-underscores slug. The order here matches
# Table 1 of the paper, which is how the ``paper_steps`` list below is indexed.
ORDERED_ENTRIES = [
    ('Water.xyz', 'water', 'Water', 4),
    ('Ammonia.xyz', 'ammonia', 'Ammonia', 4),
    ('Ethane.xyz', 'ethane', 'Ethane', 4),
    ('Acetylene.xyz', 'acetylene', 'Acetylene', 5),
    ('Allene.xyz', 'allene', 'Allene', 5),
    ('Hydroxysulfane.xyz', 'hydroxysulfane', 'Hydroxysulfane', 6),
    ('Benzene.xyz', 'benzene', 'Benzene', 3),
    ('Methylamine.xyz', 'methylamine', 'Methylamine', 4),
    ('Ethanol.xyz', 'ethanol', 'Ethanol', 5),
    ('Acetone.xyz', 'acetone', 'Acetone', 5),
    ('Disilyl-ether.xyz', 'disilyl_ether', 'Disilyl ether', 9),
    (
        '1_3_5-Trisilacyclohexane.xyz',
        'trisilacyclohexane_135',
        '1,3,5-Trisilacyclohexane',
        4,
    ),
    ('Benzaldehyde.xyz', 'benzaldehyde', 'Benzaldehyde', 7),
    ('1_3-Difluorobenzene.xyz', 'difluorobenzene_13', '1,3-Difluorobenzene', 4),
    (
        '1_3_5-Trifluorobenzene.xyz',
        'trifluorobenzene_135',
        '1,3,5-Trifluorobenzene',
        4,
    ),
    ('Neopentane.xyz', 'neopentane', 'Neopentane', 4),
    ('Furan.xyz', 'furan', 'Furan', 5),
    ('Naphthalene.xyz', 'naphthalene', 'Naphthalene', 5),
    (
        '1_5-Difluoronaphthalene.xyz',
        'difluoronaphthalene_15',
        '1,5-Difluoronaphthalene',
        5,
    ),
    (
        '2_Hydroxybicyclopentane.xyz',
        'hydroxybicyclopentane_2',
        '2-Hydroxybicyclopentane',
        9,
    ),
    ('ACHTAR10.xyz', 'achtar10', 'ACHTAR10', 10),
    ('ACANIL01.xyz', 'acanil01', 'ACANIL01', 7),
    ('Benzidine.xyz', 'benzidine', 'Benzidine', 7),
    ('Pterin.xyz', 'pterin', 'Pterin', 11),
    ('Difuropyrazine.xyz', 'difuropyrazine', 'Difuropyrazine', 7),
    ('Mesityl-oxide.xyz', 'mesityl_oxide', 'Mesityl oxide', 7),
    ('Histidine.xyz', 'histidine', 'Histidine', 14),
    ('Dimethylpentane.xyz', 'dimethylpentane', 'Dimethylpentane', 6),
    ('Caffeine.xyz', 'caffeine', 'Caffeine', 10),
    ('Menthone.xyz', 'menthone', 'Menthone', 10),
]

ZIP_PREFIX = 'quick-geopt-si/ASE_inputs/'


def parse_xyz_blob(blob, filename):
    """Parse an SI XYZ blob into a list of ``(element, x, y, z)`` tuples.

    The SI files declare an atom count on line 0 and then leave line 1
    blank; some of those headers disagree with the actual number of atom
    records that follow (e.g. ``Acetone.xyz`` declares 12 but lists 10
    atoms). We trust the parsed records and warn on a mismatch rather
    than failing — every committed ``.xyz`` is regenerated with the
    correct header by :func:`write_xyz`.
    """
    lines = blob.splitlines()
    declared = int(lines[0])
    atoms = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        # Upstream SI uses fully-uppercase element symbols (e.g. "SI" for
        # silicon); pyberny.geomlib looks them up case-sensitively against
        # "Si", so canonicalize to title-case here.
        element = parts[0].capitalize()
        atoms.append((element, float(parts[1]), float(parts[2]), float(parts[3])))
    if len(atoms) != declared:
        print(
            f'warning: {filename} declares {declared} atoms but has '
            f'{len(atoms)} records; trusting records',
            file=sys.stderr,
        )
    return atoms


def write_xyz(path, name, atoms):
    """Write a single-geometry XYZ file with a citation comment line."""
    with path.open('w') as f:
        f.write(f'{len(atoms)}\n')
        f.write(
            f'{name} (Shajan, Manathunga, Goetz, Merz, '
            f'chemrxiv 2023:7r7qn v4 -- Baker test set)\n'
        )
        for el, x, y, z in atoms:
            f.write(f'{el:<2s} {x:18.10f} {y:18.10f} {z:18.10f}\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        'si_zip',
        type=Path,
        help='path to input_files_to_different_geometry_optimizers.zip',
    )
    ap.add_argument(
        '--out',
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / 'src'
        / 'berny'
        / 'benchmarks'
        / 'baker_shajan_2023',
        help='output directory (default: src/berny/benchmarks/baker_shajan_2023)',
    )
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    reference = {}
    with zipfile.ZipFile(args.si_zip) as zf:
        for filename, slug, display_name, paper_steps in ORDERED_ENTRIES:
            blob = zf.read(ZIP_PREFIX + filename).decode()
            atoms = parse_xyz_blob(blob, filename)
            xyz_path = args.out / f'{slug}.xyz'
            write_xyz(xyz_path, display_name, atoms)
            reference[slug] = {
                'name': display_name,
                'atoms': len(atoms),
                'charge': 0,
                'mult': 1,
                'paper_steps': paper_steps,
                'paper_steps_method': 'HF',
                'paper_steps_basis': '6-31G**',
                'mopac_pm7_steps': None,
                'pyberny_steps': None,
            }
            print(f'wrote {xyz_path.name}  ({len(atoms)} atoms, paper={paper_steps})')

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
