# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ['findroot', 'fit_cubic', 'fit_quartic']

FloatArray = NDArray[np.floating[Any]]


def rms(A: FloatArray) -> float | None:
    if A.size == 0:
        return None
    return float(np.sqrt(np.sum(A**2) / A.size))


def pinv(A: FloatArray, log: Callable[[str], None] = lambda _: None) -> FloatArray:
    U, D, V = np.linalg.svd(A)
    thre = 1e3
    thre_log = 1e8
    gaps = D[:-1] / D[1:]
    try:
        n = int(np.flatnonzero(gaps > thre)[0])
    except IndexError:
        n = len(gaps)
    else:
        gap = gaps[n]
        if gap < thre_log:
            log(f'Pseudoinverse gap of only: {gap:.1e}')
    D[n + 1 :] = 0
    D[: n + 1] = 1 / D[: n + 1]
    return U.dot(np.diag(D)).dot(V)  # type: ignore[no-any-return]


def cross(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def fit_cubic(
    y0: float, y1: float, g0: float, g1: float
) -> tuple[float | None, float | None]:
    """Fit cubic polynomial to function values and derivatives at x = 0, 1.

    Returns position and function value of minimum if fit succeeds. Fit does
    not succeeds if

    1. polynomial doesn't have extrema or
    2. maximum is from (0,1) or
    3. maximum is closer to 0.5 than minimum
    """
    a = 2 * (y0 - y1) + g0 + g1
    b = -3 * (y0 - y1) - 2 * g0 - g1
    p = np.array([a, b, g0, y0])
    r = np.roots(np.polyder(p))
    if not np.isreal(r).all():
        return None, None
    r_sorted = sorted(x.real for x in r)
    if p[0] > 0:
        maxim, minim = r_sorted
    else:
        minim, maxim = r_sorted
    if 0 < maxim < 1 and abs(minim - 0.5) > abs(maxim - 0.5):
        return None, None
    return minim, float(np.polyval(p, minim))


def fit_quartic(
    y0: float, y1: float, g0: float, g1: float
) -> tuple[float | None, float | None]:
    """Fit constrained quartic polynomial to function values and erivatives at x = 0,1.

    Returns position and function value of minimum or None if fit fails or has
    a maximum. Quartic polynomial is constrained such that it's 2nd derivative
    is zero at just one point. This ensures that it has just one local
    extremum.  No such or two such quartic polynomials always exist. From the
    two, the one with lower minimum is chosen.
    """

    def g(y0: float, y1: float, g0: float, g1: float, c: float) -> FloatArray:
        a = c + 3 * (y0 - y1) + 2 * g0 + g1
        b = -2 * c - 4 * (y0 - y1) - 3 * g0 - g1
        return np.array([a, b, c, g0, y0])

    def quart_min(p: FloatArray) -> tuple[float, float]:
        r = np.roots(np.polyder(p))
        is_real = np.isreal(r)
        if is_real.sum() == 1:
            minim = r[is_real][0].real
        else:
            minim = r[(r == max(-abs(r))) | (r == -max(-abs(r)))][0].real
        return float(minim), float(np.polyval(p, minim))

    # discriminant of d^2y/dx^2=0
    D = -((g0 + g1) ** 2) - 2 * g0 * g1 + 6 * (y1 - y0) * (g0 + g1) - 6 * (y1 - y0) ** 2
    if D < 1e-11:
        return None, None
    m = -5 * g0 - g1 - 6 * y0 + 6 * y1
    p1 = g(y0, y1, g0, g1, 0.5 * (m + np.sqrt(2 * D)))
    p2 = g(y0, y1, g0, g1, 0.5 * (m - np.sqrt(2 * D)))
    if p1[0] < 0 and p2[0] < 0:
        return None, None
    [minim1, minval1] = quart_min(p1)
    [minim2, minval2] = quart_min(p2)
    if minval1 < minval2:
        return minim1, minval1
    return minim2, minval2


class FindrootError(Exception):
    pass


def findroot(f: Callable[[float], float], lim: float) -> float:
    """Find root of increasing function on (-inf,lim).

    Assumes f(-inf) < 0, f(lim) > 0.
    """
    d = 1.0
    for _ in range(1000):
        val = f(lim - d)
        if val > 0:
            break
        d = d / 2  # find d so that f(lim-d) > 0
    else:
        raise RuntimeError('Cannot find f(x) > 0')
    x = lim - d  # initial guess
    dx = 1e-10  # step for numerical derivative
    fx = f(x)
    err = abs(fx)
    for _ in range(1000):
        fxpdx = f(x + dx)
        dxf = (fxpdx - fx) / dx
        x = x - fx / dxf
        fx = f(x)
        err_new = abs(fx)
        if err_new >= err:
            return x
        err = err_new
    raise FindrootError
