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
    try:
        Berny(water(), trust=0.5, gradeintrms=1e-4)
    except TypeError as e:
        assert 'gradeintrms' in str(e)
    else:
        raise AssertionError('expected TypeError for unknown param')


def test_berny_debug_restart_roundtrip():
    geom = water()
    b = Berny(geom, debug=True)
    next(b)
    state = b.send((0.0, np.zeros((3, 3))))
    assert isinstance(state, dict)
    assert 'geom' in state and 'params' in state
    b2 = Berny(geom, restart=state)
    assert b2.trust == b._state.trust


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
        assert E == pytest.approx(0.0, abs=1e-10)

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
        dq, dE, on_sphere = quadratic_step(g, H, w, trust=1.0)
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
        dq, dE, on_sphere = quadratic_step(g, H, w, trust=0.1)
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
