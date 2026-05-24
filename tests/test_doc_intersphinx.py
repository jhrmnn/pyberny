from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_intersphinx_mapping_has_local_fallback_inventory():
    conf = (ROOT / 'doc' / 'conf.py').read_text()
    assert 'https://docs.python.org/3/objects.inv' in conf
    assert 'python-objects.inv' in conf


def test_local_python_inventory_stub_has_valid_header():
    header = (ROOT / 'doc' / 'python-objects.inv').read_text().splitlines()
    assert header[:3] == [
        '# Sphinx inventory version 1',
        '# Project: Python',
        '# Version: 3',
    ]
