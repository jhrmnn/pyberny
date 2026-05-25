import numpy as np
import pytest

from berny import Berny, BernyParams, Geometry
from berny.berny import (
    is_converged,
    linear_search,
    quadratic_step,
    update_hessian,
    update_trust,
)


def water():
    return Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
    )


def test_berny_params_defaults():
    p = BernyParams()
    assert p.gradientmax == 0.45e-3
    assert p.trust == 0.3
    assert p.energy_noise == 2e-8
    assert p.dihedral is True


def test_berny_param_override():
    b = Berny(water(), trust=0.5, gradientrms=1e-4)
    assert b.trust == 0.5
    assert b._state.params.gradientrms == 1e-4
    # Untouched defaults still come from BernyParams.
    assert b._state.params.gradientmax == 0.45e-3


def test_berny_unknown_param_rejected():
    # BernyParams has fixed fields; typos no longer silently end up in a
    # params dict that nobody reads.
    with pytest.raises(TypeError, match='gradeintrms'):
        Berny(water(), trust=0.5, gradeintrms=1e-4)


def test_berny_debug_restart_roundtrip():
    geom = water()
    b = Berny(geom, debug=True)
    next(b)
    state = b.send((0.0, np.zeros((3, 3))))
    assert isinstance(state, dict)
    assert 'geom' in state and 'params' in state
    b2 = Berny(geom, restart=state)
    assert b2.trust == b._state.trust


def _co2_geom(angle_deg, bond=1.16):
    half = np.deg2rad(angle_deg) / 2
    return Geometry(
        ['O', 'C', 'O'],
        [
            [-bond * np.sin(half), -bond * np.cos(half), 0.0],
            [0.0, 0.0, 0.0],
            [bond * np.sin(half), -bond * np.cos(half), 0.0],
        ],
    )


def _co2_harmonic(geom, *, k_bond=1.0, k_ang=0.5, r0=1.16):
    """Quadratic potential for CO2 with equilibrium at linear (180°) and
    C-O bonds at ``r0``. Returns ``(energy, gradient_NX3)``.

    Energy is in atomic-style units; the absolute scale is irrelevant for
    convergence-shape testing. The angle term is a quadratic restraint on
    ``cos(angle) + 1`` rather than ``angle - pi`` so the gradient stays
    finite at the linear configuration (the singularity that linear bends
    are meant to neutralise lives in the *internal-coordinate* description,
    not in this Cartesian potential).
    """
    coords = geom.coords
    rO1 = coords[0] - coords[1]
    rO2 = coords[2] - coords[1]
    d1 = np.linalg.norm(rO1)
    d2 = np.linalg.norm(rO2)
    e_bond = 0.5 * k_bond * ((d1 - r0) ** 2 + (d2 - r0) ** 2)
    u1 = rO1 / d1
    u2 = rO2 / d2
    cos_th = float(np.dot(u1, u2))
    cos_th = max(-1.0, min(1.0, cos_th))
    # Linear at cos_th = -1: penalise (cos_th + 1).
    e_ang = 0.5 * k_ang * (cos_th + 1.0) ** 2
    g = np.zeros_like(coords)
    # Bond gradients
    g[0] += k_bond * (d1 - r0) * u1
    g[1] -= k_bond * (d1 - r0) * u1
    g[2] += k_bond * (d2 - r0) * u2
    g[1] -= k_bond * (d2 - r0) * u2
    # Angle gradient via dE/dcos_th * d(cos_th)/d(coords).
    dE_dc = k_ang * (cos_th + 1.0)
    # d(cos_th)/d(rO1) = (u2 - cos_th * u1) / d1, similarly for rO2.
    dC_drO1 = (u2 - cos_th * u1) / d1
    dC_drO2 = (u1 - cos_th * u2) / d2
    g[0] += dE_dc * dC_drO1
    g[1] -= dE_dc * dC_drO1
    g[2] += dE_dc * dC_drO2
    g[1] -= dE_dc * dC_drO2
    return e_bond + e_ang, g


def test_berny_rebuilds_internal_coords_when_co2_straightens():
    # C2 regression test (also exercises #53's linear-bend builder mid-run).
    # Start CO2 bent at 160° — below the 175° linear threshold, so the
    # initial InternalCoords contains a regular Angle and zero dummies. The
    # harmonic potential drives the molecule toward 180°; once the bend
    # crosses 175° during the optimization, Berny.send must detect it and
    # rebuild InternalCoords with two dummies, then continue to converge.
    # Without C2, the singular-angle gradient takes over once the bend is
    # past 175° and the trust radius collapses (see #23). With C2, dummies
    # are introduced on the fly and convergence proceeds normally.
    geom = _co2_geom(160)
    berny = Berny(geom, maxsteps=60)
    assert berny._state.coords.dummy_atoms.shape == (0, 3)
    assert berny._state.coords._linear_set == set()

    rebuilt = False
    for step_geom in berny:
        e, g = _co2_harmonic(step_geom)
        berny.send((e, g))
        if not rebuilt and berny._state.coords.dummy_atoms.shape[0] > 0:
            rebuilt = True
    assert berny.converged, f'did not converge in {berny._n} steps'
    assert rebuilt, 'expected an adaptive rebuild as CO2 straightened past 175°'
    # Final configuration: linear and at the equilibrium bond length.
    final = berny._state.geom.coords
    d1 = np.linalg.norm(final[0] - final[1])
    d2 = np.linalg.norm(final[2] - final[1])
    assert d1 == pytest.approx(1.16, abs=5e-3)
    assert d2 == pytest.approx(1.16, abs=5e-3)
    u1 = (final[0] - final[1]) / d1
    u2 = (final[2] - final[1]) / d2
    # cos(angle) ≈ -1 ⇒ linear.
    assert float(np.dot(u1, u2)) < -0.999


class TestUpdateHessian:
    def test_bfgs_simple(self):
        # H = I, dq = [1, 0, 0], dg = [2, 0, 0]:
        #   dq.dg = 2; outer(dg,dg)/2 = diag(2,0,0)
        #   H@outer(dq,dq)@H = diag(1,0,0); dq.H.dq = 1
        # BFGS update: I + diag(2,0,0) - diag(1,0,0) = diag(2,1,1).
        H = np.eye(3)
        dq = np.array([1.0, 0.0, 0.0])
        dg = np.array([2.0, 0.0, 0.0])
        new_H = update_hessian(H, dq, dg)
        np.testing.assert_allclose(new_H, np.diag([2.0, 1.0, 1.0]))

    def test_remains_symmetric(self):
        rng = np.random.default_rng(0)
        H = rng.standard_normal((4, 4))
        H = H @ H.T + np.eye(4)  # symmetric PD
        dq = rng.standard_normal(4)
        dg = rng.standard_normal(4) + dq  # ensure dq.dg != 0
        new_H = update_hessian(H, dq, dg)
        np.testing.assert_allclose(new_H, new_H.T, atol=1e-9)


class TestUpdateTrust:
    def test_fifth_positional_argument_remains_log(self):
        messages = []

        def log(msg, **kwargs):
            messages.append(msg)

        new_trust = update_trust(0.3, 0.5, 1.0, np.array([0.1, 0.0]), log)
        assert new_trust == 0.3
        assert messages

    def test_predicted_energy_below_noise_holds(self):
        new_trust = update_trust(
            0.3, 0.2, 1e-9, np.array([0.1, 0.0]), energy_noise=2e-8
        )
        assert new_trust == 0.3

    def test_predicted_energy_below_noise_expands_on_boundary(self):
        new_trust = update_trust(
            0.3, -0.2, 1e-9, np.array([0.3, 0.0]), energy_noise=2e-8
        )
        assert new_trust == pytest.approx(0.6)

    def test_zero_de_treated_as_perfect(self):
        # dE == 0 → r = 1.0. r > 0.75 but |norm(dq) - trust| != 0,
        # so the trust radius shouldn't change.
        new_trust = update_trust(0.3, 0.0, 1.0, np.array([0.1, 0.0]))
        assert new_trust == 0.3

    def test_poor_fit_shrinks(self):
        # r = 0.05 < 0.25 → return norm(dq) / 4.
        new_trust = update_trust(0.3, 0.05, 1.0, np.array([0.1, 0.0]))
        assert new_trust == pytest.approx(0.025)

    def test_good_fit_at_boundary_expands(self):
        # r = 0.9 > 0.75 AND norm(dq) is (within 1e-10) equal to trust:
        # trust doubles.
        new_trust = update_trust(0.3, 0.9, 1.0, np.array([0.3, 0.0]))
        assert new_trust == pytest.approx(0.6)

    def test_moderate_fit_unchanged(self):
        # 0.25 <= r <= 0.75 → trust unchanged.
        new_trust = update_trust(0.3, 0.5, 1.0, np.array([0.1, 0.0]))
        assert new_trust == 0.3


class TestLinearSearch:
    def test_quartic_picks_minimum(self):
        # y(x) = (x - 0.5)^2: minimum at 0.5, value 0.
        t, E = linear_search(0.25, 0.25, -1.0, 1.0)
        assert t == pytest.approx(0.5)
        assert pytest.approx(0.0, abs=1e-10) == E

    def test_cubic_fallback(self):
        # Quartic fit fails (discriminant just barely negative) for these
        # inputs but a cubic fit picks up — linear_search returns the cubic's
        # minimum.
        t, E = linear_search(0.0, 1.01, 1.0, 1.0)
        # The cubic has minimum way outside (-1, 2), so the function takes the
        # cubic result and returns it; ensure we got a value (not a tuple of
        # Nones).
        assert t is not None
        assert E is not None


class TestQuadraticStep:
    def test_pure_rfo_small_gradient(self):
        # Gradient pulling toward minimum, well within trust radius:
        # the RFO branch should be taken (on_sphere=False).
        g = np.array([0.01, 0.0])
        H = np.eye(2)
        w = np.array([1.0, 1.0])
        _dq, dE, on_sphere = quadratic_step(g, H, w, trust=1.0)
        assert on_sphere is False
        # Predicted dE for a quadratic with positive-def H and step pulling
        # downhill is negative.
        assert dE < 0

    def test_sphere_minimization_large_gradient(self):
        # Gradient much larger than trust radius forces minimization on the
        # trust sphere.
        g = np.array([10.0, 0.0])
        H = np.eye(2)
        w = np.array([1.0, 1.0])
        dq, _dE, on_sphere = quadratic_step(g, H, w, trust=0.1)
        assert on_sphere is True
        assert np.linalg.norm(dq) == pytest.approx(0.1, rel=1e-3)


class TestIsConverged:
    def test_zero_forces_zero_step_converges(self):
        forces = np.zeros(6)
        step = np.zeros(6)
        assert is_converged(forces, step, on_sphere=False, params=BernyParams())

    def test_large_gradient_not_converged(self):
        forces = np.ones(6)
        step = np.zeros(6)
        assert not is_converged(forces, step, on_sphere=False, params=BernyParams())

    def test_on_sphere_blocks_convergence(self):
        # Even tiny gradients/steps shouldn't be reported as converged when
        # the previous quadratic step ran into the trust-sphere boundary.
        forces = np.zeros(6)
        step = np.zeros(6)
        assert not is_converged(forces, step, on_sphere=True, params=BernyParams())


class TestTrace:
    """Structured per-step JSON trace recording.

    When ``trace=<path>`` is passed to ``Berny``, every ``send()`` appends
    a dict-like record (mirroring the textual log output) to a list and
    the full list is rewritten to the JSON file. Tests below verify the
    file exists, parses as JSON, has one entry per send, contains the
    expected keys, and is JSON-serialisable (no numpy scalars).
    """

    def _run_co2_with_trace(self, tmp_path):
        path = tmp_path / 'trace.json'
        geom = _co2_geom(160)
        b = Berny(geom, trace=str(path), maxsteps=60)
        n_sends = 0
        for step_geom in b:
            e, g = _co2_harmonic(step_geom)
            b.send((e, g))
            n_sends += 1
        return path, b, n_sends

    def test_trace_file_created_with_one_record_per_step(self, tmp_path):
        import json

        path, b, n_sends = self._run_co2_with_trace(tmp_path)
        assert b.converged
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == n_sends
        assert [r['step'] for r in data] == list(range(1, n_sends + 1))

    def test_trace_records_have_expected_step1_keys(self, tmp_path):
        import json

        path, _, _ = self._run_co2_with_trace(tmp_path)
        data = json.loads(path.read_text())
        # Step 1 (the first send()) has no Hessian/trust/linear-search update yet.
        first = data[0]
        assert first['step'] == 1
        for key in (
            'step',
            'energy',
            'coord_rebuild',
            'quadratic_step',
            'total_step',
            'convergence',
            'converged',
            'max_steps_reached',
        ):
            assert key in first, key
        assert 'hessian_update' not in first
        assert 'linear_search' not in first
        assert 'trust_update' not in first
        # Step 2+ have the BFGS / trust / interpolation updates.
        second = data[1]
        assert second['step'] == 2
        for key in ('hessian_update', 'trust_update', 'linear_search'):
            assert key in second, key
        # Final record reports convergence.
        assert data[-1]['converged'] is True
        # The CO2-straightening path triggers an adaptive rebuild.
        assert any(r['coord_rebuild'] for r in data)

    def test_trace_record_values_are_json_serialisable(self, tmp_path):
        # No numpy scalars / arrays should leak in: json.loads on the file
        # must round-trip and every leaf must be a plain Python type.
        import json

        path, _, _ = self._run_co2_with_trace(tmp_path)
        data = json.loads(path.read_text())

        def assert_plain(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    assert_plain(v)
            elif isinstance(obj, list):
                for v in obj:
                    assert_plain(v)
            else:
                assert isinstance(obj, (str, int, float, bool, type(None))), type(obj)

        assert_plain(data)

    def test_trace_file_is_rewritten_after_each_step(self, tmp_path):
        # The trace file must be readable mid-run, not just at the end,
        # so a CI shard killed by a timeout still ships partial data.
        import json

        path = tmp_path / 'trace.json'
        geom = _co2_geom(160)
        b = Berny(geom, trace=str(path), maxsteps=60)
        sizes_seen = []
        for step_geom in b:
            e, g = _co2_harmonic(step_geom)
            b.send((e, g))
            assert path.exists()
            data = json.loads(path.read_text())
            sizes_seen.append(len(data))
            if len(sizes_seen) >= 3:
                break
        assert sizes_seen == [1, 2, 3]

    def test_no_trace_means_no_file(self, tmp_path):
        # The default path is unchanged: no trace param → no file written.
        path = tmp_path / 'trace.json'
        geom = _co2_geom(160)
        b = Berny(geom, maxsteps=5)
        next(b)
        b.send((0.0, np.zeros((3, 3))))
        assert not path.exists()
