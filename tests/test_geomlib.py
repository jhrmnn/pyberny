from pathlib import Path

import numpy as np
import pytest

from berny.geomlib import Geometry, load, loads, readfile


def test_aims_round_trip_molecule():
    g = Geometry(['H', 'O', 'H'], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    g2 = loads(format(g, 'aims'), 'aims')
    assert g2.species == g.species
    assert np.allclose(g2.coords, g.coords)
    assert g2.lattice is None


def test_aims_round_trip_crystal():
    # Until 2026, dumping a crystal as `aims` silently dropped the lattice
    # vectors, so round-tripping returned a molecule.
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    g = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]], lattice=lattice)
    g2 = loads(format(g, 'aims'), 'aims')
    assert g2.lattice is not None
    assert np.allclose(g2.lattice, g.lattice)
    assert np.allclose(g2.coords, g.coords)


def test_xyz_round_trip():
    g = Geometry(['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]])
    g2 = loads(format(g, 'xyz'), 'xyz')
    assert g2.species == g.species
    assert np.allclose(g2.coords, g.coords)
    assert g2.lattice is None


def test_dump_empty_format_uses_repr():
    g = Geometry(['H'], [[0.0, 0.0, 0.0]])
    assert format(g, '') == repr(g)


def test_dump_unknown_format_raises():
    g = Geometry(['H'], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match='Unknown format'):
        format(g, 'pdb')


def test_load_unknown_format_raises():
    with pytest.raises(ValueError, match='Unknown format'):
        loads('', 'pdb')


def test_repr_distinguishes_molecule_and_crystal():
    mol = Geometry(['H'], [[0.0, 0.0, 0.0]])
    crys = Geometry(['H'], [[0.0, 0.0, 0.0]], lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    assert 'in a lattice' not in repr(mol)
    assert 'in a lattice' in repr(crys)


def test_formula_groups_and_orders_species():
    g = Geometry(
        ['H', 'C', 'H', 'O', 'H', 'H'],
        [[i, 0.0, 0.0] for i in range(6)],
    )
    # Sorted alphabetically; counts collapsed; singletons omit the number.
    assert g.formula == 'CH4O'


def test_from_atoms_applies_unit():
    g = Geometry.from_atoms([('H', [1.0, 0.0, 0.0])], unit=2.0)
    assert np.allclose(g.coords, [[2.0, 0.0, 0.0]])


def test_copy_is_independent():
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    g = Geometry(['H', 'H'], [[0, 0, 0], [0, 0, 0.5]], lattice=lattice)
    g2 = g.copy()
    g2.coords[0, 0] = 99.0
    g2.lattice[0, 0] = 99.0
    g2.species[0] = 'He'
    assert g.coords[0, 0] == 0.0
    assert g.lattice[0, 0] == 5.0
    assert g.species[0] == 'H'


def test_write_dispatches_by_extension(tmp_path):
    g = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    xyz = tmp_path / 'mol.xyz'
    aims = tmp_path / 'mol.aims'
    geom_in = tmp_path / 'geometry.in'
    g.write(str(xyz))
    g.write(str(aims))
    g.write(str(geom_in))
    # Round-trip the formats that can be loaded.
    assert loads(xyz.read_text(), 'xyz').species == g.species
    assert loads(aims.read_text(), 'aims').species == g.species
    assert loads(geom_in.read_text(), 'aims').species == g.species


def test_write_unknown_extension_raises(tmp_path):
    g = Geometry(['H'], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match='Unknown file extension'):
        g.write(str(tmp_path / 'mol.pdb'))


def test_readfile_infers_format_from_extension(tmp_path):
    g = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    xyz = tmp_path / 'mol.xyz'
    aims = tmp_path / 'mol.aims'
    geom_in = tmp_path / 'geometry.in'
    g.write(str(xyz))
    g.write(str(aims))
    g.write(str(geom_in))
    assert readfile(str(xyz)).species == g.species
    assert readfile(str(aims)).species == g.species
    assert readfile(str(geom_in)).species == g.species


def test_readfile_explicit_format_overrides_extension(tmp_path):
    g = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    path = tmp_path / 'mol.weird'
    path.write_text(format(g, 'xyz'))
    assert readfile(str(path), fmt='xyz').species == g.species


def test_readfile_unknown_extension_raises(tmp_path):
    path = tmp_path / 'mol.pdb'
    path.write_text('')
    with pytest.raises(ValueError, match='Cannot infer format'):
        readfile(str(path))


def test_load_from_file_object(tmp_path):
    g = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    path = tmp_path / 'mol.xyz'
    path.write_text(format(g, 'xyz'))
    with open(path, encoding='utf-8') as f:
        loaded = load(f, 'xyz')
    assert loaded.species == g.species
    assert np.allclose(loaded.coords, g.coords)


def test_bondmatrix_water():
    # O-H bonds present; H-H not bonded.
    g = Geometry(['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]])
    B = g.bondmatrix()
    assert B[0, 1] and B[1, 0]
    assert B[0, 2] and B[2, 0]
    assert not B[1, 2] and not B[2, 1]


def test_dist_and_dist_diff():
    g = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    d, diff = g.dist_diff()
    assert d[0, 1] == pytest.approx(5.0)
    assert np.allclose(diff[0, 1], [-3.0, -4.0, 0.0])
    # diag is set to inf to avoid self-counting in bond detection.
    assert np.isinf(d[0, 0])
    assert g.dist()[0, 1] == pytest.approx(5.0)


def test_dist_diff_between_two_geoms():
    g1 = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    g2 = Geometry(['H', 'H', 'H'], [[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])
    d, diff = g1.dist_diff(g2)
    assert d[0, 1] == pytest.approx(5.0)
    assert d[1, 0] == pytest.approx(1.0)
    assert np.allclose(diff[0, 1], [0.0, -5.0, 0.0])


def test_masses_cms_and_inertia_for_diatomic():
    # Two equal H atoms along z: CMS at the midpoint, moment of inertia is
    # m*r^2 about the two perpendicular axes and 0 about the bond axis.
    d = 1.0
    g = Geometry(['H', 'H'], [[0.0, 0.0, -d / 2], [0.0, 0.0, d / 2]])
    m = g.masses[0]
    assert np.allclose(g.masses, [m, m])
    assert np.allclose(g.cms, [0.0, 0.0, 0.0])
    I = g.inertia
    # Symmetric, so off-diagonals vanish.
    assert np.allclose(I - np.diag(np.diag(I)), 0)
    assert I[0, 0] == pytest.approx(2 * m * (d / 2) ** 2)
    assert I[1, 1] == pytest.approx(2 * m * (d / 2) ** 2)
    assert I[2, 2] == pytest.approx(0.0, abs=1e-12)


def test_supercell_on_molecule_returns_copy():
    g = Geometry(['H', 'H'], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    sc = g.supercell()
    assert sc.lattice is None
    assert sc.species == g.species
    assert np.allclose(sc.coords, g.coords)
    # Independent copy.
    sc.coords[0, 0] = 99.0
    assert g.coords[0, 0] == 0.0


def test_supercell_default_ranges_expands_unit_cell():
    a = 3.0
    lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    g = Geometry(['H'], [[0.0, 0.0, 0.0]], lattice=lattice)
    sc = g.supercell()  # ranges = ((-1, 1)…) → shifts in [-1, 0, 1] per axis.
    assert len(sc) == 27
    # Lattice scales by (b − a) = 2, not by the number of shifts.
    assert np.allclose(sc.lattice, np.array(lattice) * 2)


def test_supercell_cutoff_uses_super_circum():
    a = 2.0
    lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    g = Geometry(['H'], [[0.0, 0.0, 0.0]], lattice=lattice)
    # super_circum picks the smallest cell whose layer-sep ≥ cutoff. With
    # a=2 and cutoff=3 the heuristic adds at least one layer on each side.
    sc = g.supercell(cutoff=3.0)
    assert sc.lattice is not None
    assert len(sc) > 1


def test_super_circum_on_molecule_returns_none():
    g = Geometry(['H'], [[0.0, 0.0, 0.0]])
    assert g.super_circum(5.0) is None


def test_super_circum_cubic_lattice():
    a = 2.0
    g = Geometry(['H'], [[0.0, 0.0, 0.0]], lattice=[[a, 0, 0], [0, a, 0], [0, 0, a]])
    # layer separation = a, so ceil(r/a + 0.5) for each axis.
    assert np.array_equal(g.super_circum(3.0), [2, 2, 2])


XYZ_DIR = Path(__file__).parent


def test_readfile_loads_bundled_xyz():
    # Sanity-check against an existing fixture so we'd notice if the loader
    # ever broke on real-world input.
    g = readfile(str(XYZ_DIR / 'water.xyz'))
    assert g.species == ['O', 'H', 'H']
    assert len(g) == 3
