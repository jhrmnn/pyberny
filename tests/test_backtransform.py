"""Round-trip tests for ``InternalCoords.update_geom``.

For each case we pick a base geometry ``geom0``, generate a known cartesian
displacement ``dx`` to define a target ``geom1 = geom0 + dx``, compute the
forward delta ``dq = coords.eval_geom(geom1, template=q0) - q0`` and then
verify that ``coords.update_geom(geom0, q0, dq, B_inv)`` recovers a geometry
whose internal coordinates equal ``q0 + dq``.

Note: internal coordinates are invariant under rigid translation/rotation, so
the recovered cartesian geometry is *not* expected to equal ``geom1`` on the
nose — only its internal-coordinate image is. That image is the actual claim
of the back-transformation; cartesian alignment is incidental.

This pins down the back-transformation independently of the optimizer: the
"right answer" (in internal-coord space) is known by construction.
"""

import numpy as np
import pytest

from berny import Math
from berny.coords import InternalCoords
from berny.geomlib import Geometry

# ---------------------------------------------------------------------------
# Fixture geometries (no file I/O so tests are self-contained)
# ---------------------------------------------------------------------------


def _water():
    return Geometry(
        ['O', 'H', 'H'],
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.9266272, 0.0],
        ],
    )


def _hydrogen_peroxide(dihedral_deg=111.5):
    # Standard H2O2-like geometry with a tunable H-O-O-H dihedral.
    r_oo = 1.475
    r_oh = 0.95
    a = np.deg2rad(94.8)
    phi = np.deg2rad(dihedral_deg)
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([r_oo, 0.0, 0.0])
    # H1 is in the xy-plane attached to O1.
    h1 = o1 + r_oh * np.array([-np.cos(a), np.sin(a), 0.0])
    # H2 attached to O2, rotated by `phi` around the O-O axis from the H1 side.
    h2_local = r_oh * np.array(
        [-np.cos(a), np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi)]
    )
    h2 = o2 + h2_local
    return Geometry(['H', 'O', 'O', 'H'], [h1, o1, o2, h2])


def _ethane():
    # Staggered ethane: C-C along x, three H per carbon at tetrahedral angles.
    r_cc = 1.54
    r_ch = 1.09
    # tetrahedral half-opening: H-C-C angle = 110.6° -> sin/cos
    a = np.deg2rad(110.6)
    cs, sn = np.cos(a), np.sin(a)
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([r_cc, 0.0, 0.0])
    atoms = [('C', c1), ('C', c2)]
    for k in range(3):
        theta = 2 * np.pi * k / 3
        h = c1 + r_ch * np.array([cs, sn * np.cos(theta), sn * np.sin(theta)])
        atoms.append(('H', h))
    for k in range(3):
        theta = 2 * np.pi * k / 3 + np.pi  # staggered
        h = c2 + r_ch * np.array([-cs, sn * np.cos(theta), sn * np.sin(theta)])
        atoms.append(('H', h))
    return Geometry([a[0] for a in atoms], [a[1] for a in atoms])


def _h2_crystal():
    return Geometry(
        ['H', 'H'],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b_inv(coords, geom):
    B = coords.B_matrix(geom)
    return B.T.dot(np.linalg.pinv(B.dot(B.T)))


def _round_trip(coords, geom0, dx):
    """Run the forward + backward transformation for cartesian step ``dx``.

    Returns ``(q0, dq, q_out, geom_out, log_lines)``.
    """
    geom1 = geom0.copy()
    geom1.coords = geom0.coords + dx
    q0 = coords.eval_geom(geom0)
    q1 = coords.eval_geom(geom1, template=q0)
    dq = q1 - q0
    B_inv = _b_inv(coords, geom0)
    log_lines = []
    q_out, geom_out = coords.update_geom(geom0, q0, dq, B_inv, log=log_lines.append)
    return q0, dq, q_out, geom_out, log_lines


def _converged(log_lines):
    return any('Perfect transformation' in line for line in log_lines)


def _iter_count(log_lines):
    # First log line is "... in N iterations"; pull out N.
    head = log_lines[0]
    return int(head.split(' in ')[1].split()[0])


def _rng_displacement(geom, magnitude, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(geom.coords.shape) * magnitude


# ---------------------------------------------------------------------------
# Core parametrized round-trip tests
# ---------------------------------------------------------------------------

# Cases: (id, factory, dx_factory, tol_q, max_iters)
_CASES = [
    ('zero', _water, lambda g: np.zeros_like(g.coords), 1e-12, 1),
    ('water-small', _water, lambda g: _rng_displacement(g, 1e-3, 1), 1e-8, 5),
    (
        'h2o2-small',
        _hydrogen_peroxide,
        lambda g: _rng_displacement(g, 1e-3, 2),
        1e-8,
        5,
    ),
    (
        'h2o2-moderate',
        _hydrogen_peroxide,
        lambda g: _rng_displacement(g, 1e-2, 3),
        1e-6,
        10,
    ),
    ('ethane-moderate', _ethane, lambda g: _rng_displacement(g, 1e-2, 4), 1e-6, 10),
    ('ethane-large', _ethane, lambda g: _rng_displacement(g, 0.05, 5), 1e-4, 20),
]


@pytest.mark.parametrize(
    'name,factory,dx_factory,tol_q,max_iters',
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_backtransform_recovers_forward_step(
    name, factory, dx_factory, tol_q, max_iters
):
    geom0 = factory()
    coords = InternalCoords(geom0)
    dx = dx_factory(geom0)
    q0, dq, q_out, geom_out, log = _round_trip(coords, geom0, dx)

    assert _converged(log), f'{name}: update_geom did not converge: {log}'
    assert _iter_count(log) <= max_iters, f'{name}: too many iterations ({log[0]})'

    # The actual back-transformation claim: the recovered internal coords
    # equal q0 + dq (the forward delta we asked for).
    q_rms = Math.rms(q_out - (q0 + dq))
    assert q_rms < tol_q, f'{name}: q_out RMS {q_rms} > {tol_q}'

    # And the recovered cartesian geometry, when re-evaluated, lands on the
    # same internal-coordinate point as geom1 = geom0 + dx. This is a
    # stronger check than the previous one because q_out is taken from the
    # last inner iterate; this one re-evaluates from scratch.
    q_check = coords.eval_geom(geom_out, template=q0)
    q_check_rms = Math.rms(q_check - (q0 + dq))
    assert q_check_rms < tol_q, f'{name}: eval(geom_out) RMS {q_check_rms} > {tol_q}'


# ---------------------------------------------------------------------------
# Specialised cases
# ---------------------------------------------------------------------------


def test_backtransform_rigid_rotation_about_bond():
    """Rotating one H of H2O2 around the O-O axis is the cleanest dihedral move.

    The forward delta is essentially a single dihedral change; the back-
    transformation must recover the new dihedral value (the rotated cartesian
    geometry is one of many that produce the same internal coords).
    """
    geom0 = _hydrogen_peroxide(dihedral_deg=111.5)
    coords = InternalCoords(geom0)
    # Rotate atom index 3 (the second H) by 20° around the O-O axis (x-axis,
    # passing through atom 1 = O1 at origin).
    angle = np.deg2rad(20.0)
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    dx = np.zeros_like(geom0.coords)
    dx[3] = rot.dot(geom0.coords[3]) - geom0.coords[3]
    q0, dq, q_out, geom_out, log = _round_trip(coords, geom0, dx)

    assert _converged(log), log
    assert Math.rms(q_out - (q0 + dq)) < 1e-5
    q_check = coords.eval_geom(geom_out, template=q0)
    assert Math.rms(q_check - (q0 + dq)) < 1e-5


def test_backtransform_dihedral_wrap_across_pi():
    """A dihedral crossing ±π must produce a "short way around" forward delta.

    eval_geom(template=q) wraps the raw arccos result by ±2π/±π so that the
    delta is small. update_geom relies on this and must still recover the
    rotated geometry.
    """
    # Base dihedral close to +π so a small physical rotation tips it past ±π.
    geom0 = _hydrogen_peroxide(dihedral_deg=170.0)
    coords = InternalCoords(geom0)
    # Rotate the trailing H by +30° about the O-O axis, taking the dihedral
    # from +170° to +200° (i.e. -160° after wrapping).
    angle = np.deg2rad(30.0)
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    dx = np.zeros_like(geom0.coords)
    dx[3] = rot.dot(geom0.coords[3]) - geom0.coords[3]
    q0, dq, q_out, geom_out, log = _round_trip(coords, geom0, dx)

    # The forward dq for the wrapped dihedral must be the *short* way around
    # (|dq| close to the 30° rotation, not 330°).
    dih_idx = [i for i, c in enumerate(coords) if c.__class__.__name__ == 'Dihedral']
    assert dih_idx, 'expected at least one dihedral in H2O2'
    for i in dih_idx:
        assert abs(dq[i]) < np.pi, f'forward dq[{i}]={dq[i]} not wrapped to short path'

    assert _converged(log), log
    q_check = coords.eval_geom(geom_out, template=q0)
    assert Math.rms(q_check - (q0 + dq)) < 1e-6
    assert Math.rms(q_out - (q0 + dq)) < 1e-6


def test_backtransform_near_linear_angle():
    """Stresses the ``Angle.eval`` clamp branch (φ → π).

    A nearly-linear triatomic with a tiny perturbation that keeps it near
    linear must still round-trip; the clamp shouldn't poison the
    back-transformation.
    """
    geom0 = Geometry(
        ['H', 'O', 'H'],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 1e-3, 0.0]],
    )
    coords = InternalCoords(geom0)
    dx = _rng_displacement(geom0, 1e-4, 7)
    q0, dq, q_out, geom_out, log = _round_trip(coords, geom0, dx)

    assert _converged(log), log
    # Tolerance is loose because near-π angle gradients are large.
    assert Math.rms(q_out - (q0 + dq)) < 1e-5
    assert np.all(np.isfinite(q_out))
    assert np.all(np.isfinite(geom_out.coords))


def test_backtransform_rejects_periodic_geometry():
    """``update_geom`` doesn't support crystals: ``B_matrix`` returns a
    supercell-sized inverse but the cartesian update only fits the unit cell.

    This pins the current limitation so that any future change (either a fix
    that adds support, or an explicit error) is a visible, deliberate edit.
    """
    geom0 = _h2_crystal()
    coords = InternalCoords(geom0)
    q0 = coords.eval_geom(geom0)
    B_inv = _b_inv(coords, geom0)
    # An arbitrary in-range dq just to exercise the call.
    dq = np.zeros(len(q0))
    with pytest.raises(ValueError, match='could not be broadcast'):
        coords.update_geom(geom0, q0, dq, B_inv)


def test_backtransform_fallback_on_non_convergence():
    """A grossly oversized dq must trigger the ``keep_first`` fallback.

    The iteration cannot converge for an unphysically large internal-coordinate
    step. ``update_geom`` should log "did not converge", return the first-
    iteration geometry, and produce only finite numbers.
    """
    geom0 = _ethane()
    coords = InternalCoords(geom0)
    # Build a realistic forward step then blow it up by 100×.
    dx = _rng_displacement(geom0, 1e-2, 11)
    q0 = coords.eval_geom(geom0)
    q1 = coords.eval_geom(Geometry(list(geom0.species), geom0.coords + dx), template=q0)
    huge_dq = 100.0 * (q1 - q0)
    B_inv = _b_inv(coords, geom0)
    log_lines = []
    q_out, geom_out = coords.update_geom(
        geom0, q0, huge_dq, B_inv, log=log_lines.append
    )

    assert any('did not converge' in line for line in log_lines), log_lines
    # 20 iterations is the hard cap in update_geom.
    assert _iter_count(log_lines) == 20
    # Fallback returns the *first* iterate, not the diverged final state.
    expected_first = geom0.coords + B_inv.dot(huge_dq).reshape(-1, 3) * 0.52917721092
    assert Math.rms(geom_out.coords - expected_first) < 1e-10
    assert np.all(np.isfinite(q_out))
    assert np.all(np.isfinite(geom_out.coords))


def test_backtransform_log_message_format():
    """Downstream log parsers depend on the exact wording emitted by update_geom.

    A regression in the format strings at ``coords.py:393-401`` should fail
    here before it reaches optimizer logs.
    """
    geom0 = _water()
    coords = InternalCoords(geom0)
    dx = _rng_displacement(geom0, 1e-3, 13)
    *_, log_lines = _round_trip(coords, geom0, dx)

    assert len(log_lines) == 2
    assert log_lines[0].startswith('Perfect transformation to cartesians in ')
    assert log_lines[0].endswith(' iterations')
    assert log_lines[1].startswith('* RMS(dcart): ')
    assert 'RMS(dq):' in log_lines[1]
