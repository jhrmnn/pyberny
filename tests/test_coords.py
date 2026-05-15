import numpy as np
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
    # Comparison against an unrelated type returns NotImplemented from
    # __eq__, which Python turns into False.
    assert Bond(1, 2) != 'not-a-coord'
    assert Bond(1, 2) != 42


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


def _no_singular_angle(coords, geom):
    """Assert that no Angle coordinate is near the singular branch (~180 deg)."""
    all_coords = coords._all_coords(geom.supercell())
    for c in coords.angles:
        phi = c.eval(all_coords)
        assert phi < np.pi - 5 * np.pi / 180, (c, phi)


def test_linear_bends_replace_singular_angle_co2():
    # CO2 is linear; the O-C-O angle should not appear as a regular angle
    # coordinate. Instead two dummies should be placed and four angles
    # routed through them.
    geom = Geometry(
        ['O', 'C', 'O'],
        [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]],
    )
    coords = InternalCoords(geom)
    assert coords.dummy_atoms.shape == (2, 3)
    assert len(coords.angles) == 4
    _no_singular_angle(coords, geom)


def test_linear_bends_acetylene():
    # H-C#C-H: two linear triples (H-C-C and C-C-H), four dummies, eight angles.
    geom = Geometry(
        ['H', 'C', 'C', 'H'],
        [[0, 0, -1.7], [0, 0, -0.6], [0, 0, 0.6], [0, 0, 1.7]],
    )
    coords = InternalCoords(geom)
    assert coords.dummy_atoms.shape == (4, 3)
    assert len(coords.angles) == 8
    _no_singular_angle(coords, geom)


def test_linear_bends_dummies_perpendicular_to_axis():
    # Each dummy should sit perpendicular to the host i-k axis, displaced
    # from the host j atom.
    geom = Geometry(
        ['O', 'C', 'O'],
        [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]],
    )
    coords = InternalCoords(geom)
    for spec, d in zip(coords._dummy_specs, coords.dummy_atoms):
        offset = d - geom.coords[spec.j]
        axis = geom.coords[spec.k] - geom.coords[spec.i]
        axis = axis / np.linalg.norm(axis)
        assert abs(np.dot(offset, axis)) < 1e-10
        assert abs(np.linalg.norm(offset) - 1.0) < 1e-10  # _DUMMY_OFFSET


def test_b_matrix_shape_excludes_dummies():
    # The B-matrix should be sized over real atoms only; dummies are
    # handled implicitly through the frozen-dummy approximation.
    geom = Geometry(
        ['O', 'C', 'O'],
        [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]],
    )
    coords = InternalCoords(geom)
    B = coords.B_matrix(geom)
    assert B.shape == (len(coords), 3 * len(geom))


def test_ghost_atom_in_geometry_does_not_crash():
    # Issue #9: previously, a "Ghost" species in the input geometry raised
    # KeyError deep inside InternalCoords. Now it should build without
    # raising and the ghost should not enter any covalent bond.
    geom = Geometry(
        ['H', 'H', 'Ghost'],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]],
    )
    coords = InternalCoords(geom)
    # No bond should connect a ghost atom (index 2) covalently.
    assert all(b.weak >= 1 for b in coords.bonds if 2 in b.idx)


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


def test_get_property_unknown_symbol_raises():
    with pytest.raises(KeyError, match="'Xx'"):
        get_property('Xx', 'mass')


def test_get_property_unknown_number_raises():
    with pytest.raises(KeyError, match='9999'):
        get_property(9999, 'mass')


def test_get_property_known_species_lookup_by_number():
    # Hydrogen is element 1 — exercises the int-lookup branch.
    assert get_property(1, 'symbol') == 'H'


def test_internal_coord_repr_with_connectivity():
    # __repr__ requires the `weak` attribute, which is only populated when
    # the coord is constructed with a connectivity matrix `C`.
    C = np.ones((4, 4), dtype=bool)
    assert repr(Bond(0, 1, C=C)).startswith('Bond(') and 'weak=0' in repr(
        Bond(0, 1, C=C)
    )
    assert repr(Angle(0, 1, 2, C=C)).startswith('Angle(')
    assert repr(Dihedral(0, 1, 2, 3, C=C)).startswith('Dihedral(')


def test_angle_eval_clamps_collinear_atoms():
    # Three colinear atoms — the dot product would round to slightly above
    # 1, triggering the upper clip; arccos must still return 0 or π.
    a = Angle(0, 1, 2)
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    # Both extend the same way → 0 rad after clamp.
    assert a.eval(coords) == pytest.approx(0.0)
    # Opposite direction → π rad.
    coords[2] = [-2.0, 0.0, 0.0]
    assert a.eval(coords) == pytest.approx(np.pi)


def test_angle_grad_near_pi():
    # A nearly-π angle hits the "phi > π − 1e-6" gradient branch.
    a = Angle(0, 1, 2)
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 1e-8, 0.0]])
    phi, grad = a.eval(coords, grad=True)
    assert phi == pytest.approx(np.pi, abs=1e-3)
    assert len(grad) == 3
    assert all(np.all(np.isfinite(g)) for g in grad)


def _fd_grad(coord, geom, h, unwrap=False):
    # 4th-order central-difference gradient of coord.eval w.r.t. atomic
    # positions in Å. The analytic gradient returned by coord.eval is
    # d(value)/d(position in Bohr); to compare against an FD computed on
    # positions in Å, multiply the analytic gradient by `angstrom` (Bohr/Å).
    out = np.zeros_like(geom)
    for a in range(geom.shape[0]):
        for c in range(3):

            def ev(d, _a=a, _c=c):
                g = geom.copy()
                g[_a, _c] += d
                return coord.eval(g)

            f1, fm1, f2, fm2 = ev(h), ev(-h), ev(2 * h), ev(-2 * h)
            if unwrap:
                # Dihedrals jump by 2π across the ±π branch cut. Unwrap so
                # the central difference sees a smooth function.
                vals = [fm2, fm1, f1, f2]
                ref = vals[0]
                for i, v in enumerate(vals):
                    while v - ref > np.pi:
                        v -= 2 * np.pi
                    while ref - v > np.pi:
                        v += 2 * np.pi
                    vals[i] = v
                fm2, fm1, f1, f2 = vals
            out[a, c] = (-f2 + 8 * f1 - 8 * fm1 + fm2) / (12 * h)
    return out


@pytest.mark.parametrize(
    'p',
    [
        np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[1.7, -0.3, 0.6], [0.1, 0.2, -0.4]]),
        np.array([[0.0, 0.0, 2.5], [0.0, 0.0, 0.0]]),
    ],
)
def test_bond_grad_fd(p):
    b = Bond(0, 1)
    _, grad = b.eval(p, grad=True)
    fd = _fd_grad(b, p, 1e-4)
    np.testing.assert_allclose(np.array(grad) * angstrom, fd, atol=1e-9)


@pytest.mark.parametrize(
    'theta', [0.3, 0.9, np.pi / 2, 1.92, 110 * np.pi / 180, np.pi - 0.1]
)
def test_angle_grad_fd(theta):
    # Generic branch only. Angles are unsigned (phi ∈ [0, π]), so phi has a
    # kink at π — the perpendicular direction of the gradient is ambiguous
    # in that limit and FD does not converge to a unique value. The near-π
    # special branch (test_angle_grad_near_pi) is therefore not FD-checked.
    a = Angle(0, 1, 2)
    p = np.array(
        [[np.cos(theta), np.sin(theta), 0.0], [0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]
    )
    _, grad = a.eval(p, grad=True)
    fd = _fd_grad(a, p, 1e-4)
    np.testing.assert_allclose(np.array(grad) * angstrom, fd, atol=1e-8)


def _make_dihedral_geom(
    phi,
    r1=1.0,
    r2=1.1,
    rjk=1.3,
    th1=110 * np.pi / 180,
    th2=105 * np.pi / 180,
):
    # Build a 4-atom geometry with a controlled dihedral. Bond lengths and
    # bend angles are deliberately unequal so that the A=v1·ew/|w| and
    # B=v2·ew/|w| projection factors in the special-branch formulas are
    # non-zero (a symmetric geometry would mask bugs in the A/B terms).
    j = np.zeros(3)
    k = np.array([rjk, 0.0, 0.0])
    i = j + r1 * np.array([np.cos(th1), np.sin(th1), 0.0])
    kl = r2 * np.array(
        [-np.cos(th2), np.sin(th2) * np.cos(phi), np.sin(th2) * np.sin(phi)]
    )
    return np.array([i, j, k, k + kl])


@pytest.mark.parametrize(
    'phi', [-2.5, -1.0, -0.3, 0.1, 0.5, 1.2, np.pi / 2, np.pi - 0.2]
)
def test_dihedral_grad_fd(phi):
    # Generic-branch FD check across a representative range of dihedrals.
    d = Dihedral(0, 1, 2, 3)
    p = _make_dihedral_geom(phi)
    _, grad = d.eval(p, grad=True)
    fd = _fd_grad(d, p, 1e-4, unwrap=True)
    np.testing.assert_allclose(np.array(grad) * angstrom, fd, atol=1e-8)


# Special-branch tests for dihedrals near the planarity detection threshold.
# The branches are selected at |phi| < 1e-6 (near 0) and |phi| > π - 1e-6
# (near π) in Dihedral.eval. We sample phi values straddling the threshold
# on both sides — values just outside still use the generic 1/sin(phi)
# formula, values just inside use the special formula — and verify that the
# analytic gradient matches a numerical one in both regimes. The FP precision
# of arccos near ±1 limits how accurate the computed phi (and hence FD) can
# be for very small |phi| or |π−phi|, so a relatively loose atol and large
# FD step are used; this is still tight enough to catch the kind of bug
# these special branches exist to avoid.
_NEAR_PLANAR_PHIS = [
    # Just outside the near-0 threshold (generic branch)
    -2e-6,
    1.1e-6,
    # Just inside (special branch)
    9e-7,
    5e-7,
    1e-7,
    -1e-7,
    0.0,
    # Just outside the near-π threshold (generic branch)
    np.pi - 2e-6,
    -(np.pi - 1.1e-6),
    # Just inside (special branch)
    np.pi - 9e-7,
    np.pi - 5e-7,
    np.pi - 1e-7,
    -(np.pi - 1e-7),
    np.pi,
]


@pytest.mark.parametrize('phi', _NEAR_PLANAR_PHIS)
def test_dihedral_grad_fd_near_planar(phi):
    d = Dihedral(0, 1, 2, 3)
    p = _make_dihedral_geom(phi)
    _, grad = d.eval(p, grad=True)
    # Larger FD step than test_dihedral_grad_fd because the computed phi
    # itself has only ~8 significant digits at these scales (arccos near ±1
    # loses precision); a smaller h would put FD into the noise floor.
    fd = _fd_grad(d, p, 1e-3, unwrap=True)
    # The generic 1/sin(phi) formula loses precision just outside the
    # threshold and the FD itself is noisy when computed phi values are
    # near the FP precision floor of arccos, so a looser tolerance is
    # used than in the bulk-phi test above.
    np.testing.assert_allclose(np.array(grad) * angstrom, fd, atol=5e-4)


def test_dihedral_grad_continuous_across_threshold():
    # The special-branch and generic-branch formulas should agree in the
    # limit. Compare the analytic gradient at phi values straddling the
    # detection thresholds on essentially the same geometry.
    d = Dihedral(0, 1, 2, 3)
    for phi_gen, phi_spec in [(1.1e-6, 9e-7), (np.pi - 1.1e-6, np.pi - 9e-7)]:
        _, g_gen = d.eval(_make_dihedral_geom(phi_gen), grad=True)
        _, g_spec = d.eval(_make_dihedral_geom(phi_spec), grad=True)
        np.testing.assert_allclose(np.array(g_gen), np.array(g_spec), atol=1e-3)


def test_internal_coords_repr_and_str_describe_counts():
    geom = Geometry(
        ['O', 'H', 'H'],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
    )
    coords = InternalCoords(geom)
    r = repr(coords)
    assert r.startswith('<InternalCoords')
    assert 'bonds' in r and 'angles' in r and 'dihedrals' in r
    s = str(coords)
    assert 'Internal coordinates' in s
    assert 'Number of fragments' in s


def test_internal_coords_without_dihedrals():
    # `dihedral=False` skips the get_dihedrals loop — covers the missing
    # branch around InternalCoords.__init__.
    geom = Geometry(
        ['O', 'H', 'H', 'H'],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    coords = InternalCoords(geom, dihedral=False)
    assert coords.dihedrals == []
    assert len(coords.bonds) > 0


def test_internal_coords_on_crystal_prunes_via_reduce():
    # A small primitive H₂ crystal exercises the `_reduce` path that
    # InternalCoords runs only when geom.lattice is not None.
    geom = Geometry(
        ['H', 'H'],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    )
    coords = InternalCoords(geom)
    # Reduce keeps at least one bond between the two atoms.
    assert len(coords.bonds) >= 1
    assert len(coords) > 0
