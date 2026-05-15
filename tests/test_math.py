import numpy as np
import pytest

from berny import Math


class TestRms:
    def test_basic(self):
        assert Math.rms(np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))

    def test_zero(self):
        assert Math.rms(np.zeros(5)) == 0

    def test_empty_returns_none(self):
        assert Math.rms(np.array([])) is None

    def test_matrix(self):
        # rms is over all elements, not row-wise.
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert Math.rms(A) == pytest.approx(np.sqrt(30 / 4))


class TestPinv:
    def test_identity(self):
        A = np.eye(3)
        np.testing.assert_allclose(Math.pinv(A), A, atol=1e-12)

    def test_well_conditioned_symmetric(self):
        # Math.pinv is only used inside pyberny on symmetric matrices
        # (B @ B.T). The SVD reconstruction it performs assumes U == V, so
        # the symmetric case is the contract we want to lock in.
        rng = np.random.default_rng(0)
        M = rng.standard_normal((4, 4))
        A = M @ M.T + np.eye(4)  # symmetric positive-definite
        np.testing.assert_allclose(Math.pinv(A), np.linalg.pinv(A), atol=1e-9)

    def test_truncates_small_singular_values(self):
        # Construct a symmetric matrix with a deliberate gap in singular
        # values.
        Q = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        D = np.diag([1.0, 1.0, 1e-10])
        A = Q @ D @ Q.T
        # The 1e-10 singular value should be cut by the gap heuristic, so
        # pinv(A) @ A is a rank-2 projector rather than the identity.
        P = Math.pinv(A) @ A
        assert np.linalg.matrix_rank(P, tol=1e-6) == 2


class TestCross:
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        np.testing.assert_allclose(Math.cross(a, b), np.cross(a, b))

    def test_orthogonal_to_inputs(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        c = Math.cross(a, b)
        assert abs(np.dot(c, a)) < 1e-12
        assert abs(np.dot(c, b)) < 1e-12


class TestFitCubic:
    def test_real_cubic(self):
        # y(x) = x^3 + x^2 - 3x + 1, derivative roots at approx -1.39 and 0.72.
        # y0 = 1, y1 = 0, g0 = -3, g1 = 2.
        x, y = Math.fit_cubic(1.0, 0.0, -3.0, 2.0)
        assert x == pytest.approx(0.7208, abs=1e-3)
        assert y == pytest.approx(np.polyval([1, 1, -3, 1], x), abs=1e-9)

    def test_maximum_inside_interval_returns_none(self):
        # y(x) = -x^3 + 0.45 x^2 + 0.54 x has derivative roots at -0.3 and 0.6
        # (a < 0, so min at -0.3 and max at 0.6). The max sits inside (0,1)
        # and closer to 0.5 than the min, which is the third bail-out
        # condition documented in fit_cubic.
        x, y = Math.fit_cubic(0.0, -0.01, 0.54, -1.56)
        assert x is None and y is None


class TestFitQuartic:
    def test_exact_parabola(self):
        # Same parabola as above; quartic should also nail it.
        y0, y1, g0, g1 = 0.25, 0.25, -1.0, 1.0
        x, y = Math.fit_quartic(y0, y1, g0, g1)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_negative_discriminant_returns_none(self):
        # Linear function: y(x) = x. Discriminant ends up negative, so the
        # constrained-quartic fit refuses to commit and returns None.
        x, y = Math.fit_quartic(0.0, 1.0, 1.0, 1.0)
        assert x is None and y is None


class TestFindroot:
    def test_linear(self):
        # f(x) = x - 3, root at 3. lim = 10 satisfies f(lim) > 0.
        x = Math.findroot(lambda x: x - 3, 10.0)
        assert x == pytest.approx(3.0, rel=1e-6)

    def test_finds_negative_root(self):
        # f(x) = x + 2, root at -2. lim must satisfy f(lim) > 0, so lim = 5.
        x = Math.findroot(lambda x: x + 2, 5.0)
        assert x == pytest.approx(-2.0, rel=1e-6)

    def test_unreachable_positive_raises(self):
        # f is always negative — algorithm gives up after halving 1000 times.
        with pytest.raises(RuntimeError, match='Cannot find'):
            Math.findroot(lambda x: -1.0, 0.0)
