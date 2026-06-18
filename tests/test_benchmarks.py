"""Tests for the :mod:`berny.benchmarks` discovery API."""

import pytest

from berny.benchmarks import (
    BENCHMARKS,
    data_dir,
    iter_molecules,
    load_reference,
)
from berny.geomlib import Geometry


def _oligomers_geometries_available():
    """True when the external/oligomer-benchmarks submodule is checked out."""
    try:
        return data_dir('oligomers').joinpath('acenes', 'naphthalene.xyz').exists()
    except OSError:
        return False


def test_benchmarks_mapping_has_known_keys():
    assert set(BENCHMARKS) == {'birkholz', 'baker', 'oligomers'}


@pytest.mark.parametrize(
    ('name', 'n_molecules'),
    [('birkholz', 19), ('baker', 30), ('oligomers', 91)],
)
def test_load_reference_returns_expected_set(name, n_molecules):
    ref = load_reference(name)
    assert len(ref) == n_molecules
    # Sanity-check the schema on an arbitrary entry: every molecule must at
    # least record its atom count, charge, and multiplicity.
    sample = next(iter(ref.values()))
    for key in ('atoms', 'charge', 'mult'):
        assert key in sample, f'missing {key!r} in {name} reference entry'


def test_oligomers_reference_entries_carry_file_key():
    # The oligomers geometries live in a submodule, so every entry must
    # locate its .xyz via a ``file`` key relative to the geometry root.
    ref = load_reference('oligomers')
    assert all('file' in rec for rec in ref.values())


@pytest.mark.parametrize('name', ['birkholz', 'baker'])
def test_iter_molecules_yields_geometry_matching_reference(name):
    triples = list(iter_molecules(name))
    ref = load_reference(name)
    assert len(triples) == len(ref)
    # iter_molecules defaults to sorted(reference) ordering -- the API
    # contract downstream regression rows rely on.
    assert [n for n, _, _ in triples] == sorted(ref)
    for mol_name, geom, meta in triples:
        assert isinstance(geom, Geometry)
        assert len(geom) == meta['atoms'], (
            f'{name}/{mol_name}: geometry has {len(geom)} atoms but '
            f"reference says {meta['atoms']}"
        )


@pytest.mark.skipif(
    not _oligomers_geometries_available(),
    reason='external/oligomer-benchmarks submodule not checked out',
)
def test_iter_molecules_oligomers_reads_from_submodule():
    # The oligomers set resolves each .xyz via ref['file'] under a geometry
    # root outside the package (the submodule); exercise that path end to end.
    triples = list(iter_molecules('oligomers'))
    ref = load_reference('oligomers')
    assert len(triples) == len(ref) == 91
    assert [n for n, _, _ in triples] == sorted(ref)
    for mol_name, geom, meta in triples:
        assert isinstance(geom, Geometry)
        assert len(geom) == meta['atoms'], (
            f'oligomers/{mol_name}: geometry has {len(geom)} atoms but '
            f"reference says {meta['atoms']}"
        )


def test_iter_molecules_honors_subset_and_order():
    chosen = ['estradiol', 'codeine']
    out = [n for n, _, _ in iter_molecules('birkholz', names=chosen)]
    assert out == chosen


def test_load_reference_rejects_unknown_benchmark():
    with pytest.raises(ValueError, match='unknown benchmark'):
        load_reference('does-not-exist')


def test_data_dir_rejects_unknown_benchmark():
    with pytest.raises(ValueError, match='unknown benchmark'):
        data_dir('does-not-exist')


def test_iter_molecules_rejects_unknown_molecule():
    with pytest.raises(ValueError, match='unknown molecules'):
        list(iter_molecules('birkholz', names=['not_a_real_molecule']))


def test_data_dir_resolves_to_packaged_directory():
    # data_dir is a thin convenience for scripts that want a filesystem
    # path; for ordinary file-system installs the result must contain
    # the reference.json that load_reference reads via importlib.resources.
    assert (data_dir('birkholz') / 'reference.json').exists()
