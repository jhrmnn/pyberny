import importlib.util
from pathlib import Path

from berny.benchmarks import geometries_available

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    'molecule_gallery', ROOT / 'scripts' / 'molecule_gallery.py'
)
molecule_gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(molecule_gallery)

# Expected molecule counts per benchmark; ``oligomers`` is only present when
# its (submodule-backed) geometries are checked out.
EXPECTED_COUNTS = {'birkholz': 19, 'baker': 30, 'oligomers': 91}


def _available_benchmarks():
    return [name for name in molecule_gallery.BENCHMARKS if geometries_available(name)]


def test_payload_includes_available_benchmarks():
    payload = molecule_gallery.build_payload()
    available = _available_benchmarks()
    assert payload['order'] == available
    # birkholz and baker are package data, so they are always present.
    assert {'birkholz', 'baker'} <= set(payload['order'])
    for name in payload['order']:
        molecules = payload['benchmarks'][name]['molecules']
        assert len(molecules) == EXPECTED_COUNTS[name]


def test_collect_molecules_carries_geometry_and_metadata():
    for name in _available_benchmarks():
        molecules = molecule_gallery.collect_molecules(name)
        for mol in molecules:
            # Atom count in the first xyz line matches the embedded geometry.
            atoms = next(value for label, value in mol['fields'] if label == 'Atoms')
            assert mol['xyz'].splitlines()[0].strip() == atoms
            assert mol['name']


def test_build_html_embeds_every_benchmark():
    html = molecule_gallery.build_html()
    assert molecule_gallery.PLACEHOLDER not in html
    assert '3Dmol' in html
    payload = molecule_gallery.build_payload()
    for name in payload['order']:
        assert payload['benchmarks'][name]['label'] in html
        for mol in payload['benchmarks'][name]['molecules']:
            assert mol['id'] in html
            assert mol['name'] in html
