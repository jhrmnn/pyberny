import numpy as np

from berny.geomlib import Geometry, loads


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
