import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    'molecule_gallery', ROOT / 'scripts' / 'molecule_gallery.py'
)
molecule_gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(molecule_gallery)


def test_gallery_embeds_all_benchmark_molecules():
    molecules = molecule_gallery.collect_molecules(molecule_gallery.DATA_DIR)
    assert len(molecules) == 19
    html = molecule_gallery.build_html()
    assert molecule_gallery.PLACEHOLDER not in html
    assert '3Dmol' in html
    for mol in molecules:
        assert mol['id'] in html
        assert mol['name'] in html
        assert mol['xyz'].splitlines()[0].strip() == str(mol['atoms'])
