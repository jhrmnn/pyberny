import pytest

from berny.coords import Angle, Bond, Dihedral, InternalCoords, angstrom
from berny.geomlib import Geometry
from berny.species_data import get_property, species_data


def test_internal_coord_equality_and_hashing():
    # Until 2026, InternalCoord.__eq__ silently returned None, breaking every
    # `x in set(...)`/`dict` lookup that the dihedral-swap logic in
    # InternalCoords.eval_geom relies on.
    assert Bond(1, 2) == Bond(2, 1)
    assert Bond(1, 2) != Bond(1, 3)
    assert hash(Bond(1, 2)) == hash(Bond(2, 1))
    assert Angle(1, 2, 3) == Angle(3, 2, 1)
    assert Angle(1, 2, 3) != Bond(1, 2)
    s = {Bond(1, 2), Bond(2, 1), Angle(1, 2, 3)}
    assert len(s) == 2
    assert Dihedral(1, 2, 3, 4) == Dihedral(4, 3, 2, 1)


def test_cycle_dihedrals():
    geom = Geometry.from_atoms(
        [(ws[1], ws[2:5]) for ws in (l.split() for l in """\
    1 H -0.000000000000 0.000000000000 -1.142569988888
    2 O 1.784105551801 1.364934064507 -1.021376180623
    3 H 2.248320553963 2.318104360291 -2.500037742933
    4 H 3.285761299420 0.674554743661 -0.259576564237
    5 O -1.784105551799 -1.364934064536 -1.021376180591
    6 H -2.248320553963 -2.318104360291 -2.500037742933
    7 H -3.285761299424 -0.674554743614 -0.259576564287
    8 O 5.839754502206 -0.500682935209 1.037064691223
    9 H 7.440059622286 -1.597667062287 0.565115038647
    10 H 6.475526400773 0.638572472561 2.500357106648
    11 O -5.839754502205 0.500682935191 1.037064691242
    12 H -7.440059622286 1.597667062287 0.565115038647
    13 H -6.475526400773 -0.638572472561 2.500357106648
    """.strip().split('\n'))],
        unit=1 / angstrom,
    )
    coords = InternalCoords(geom)
    assert not [dih for dih in coords.dihedrals if len(set(dih.idx)) < 4]


def test_internal_coords_with_previously_missing_radius():
    # Astatine had no covalent radius, which used to crash InternalCoords
    # with an opaque numpy error (see issue #...).
    geom = Geometry.from_atoms(
        [('At', [0.0, 0.0, 0.0]), ('At', [0.0, 0.0, 2.5])],
        unit=1 / angstrom,
    )
    coords = InternalCoords(geom)
    assert len(coords.bonds) == 1


def test_get_property_missing_data_raises_keyerror():
    # If a species exists but the requested property is empty, the user
    # should get a clear KeyError naming the species and the property,
    # not a cryptic numpy TypeError further down the line.
    species_data['Zz'] = {
        'number': 999.0,
        'name': 'fake',
        'symbol': 'Zz',
        'covalent_radius': '',
        'mass': 0.0,
        'vdw_radius': 0.0,
    }
    try:
        with pytest.raises(KeyError, match='covalent_radius'):
            get_property('Zz', 'covalent_radius')
        with pytest.raises(KeyError, match='covalent_radius'):
            get_property(999, 'covalent_radius')
    finally:
        del species_data['Zz']
