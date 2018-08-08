# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np


def rms(A):
    if A.size == 0:
        return None
    return np.sqrt(np.sum(A**2)/A.size)


def pinv(A, log=lambda _: None):
    U, D, V = np.linalg.svd(A)
    thre = 1e3
    thre_log = 1e8
    gaps = D[:-1]/D[1:]
    try:
        n = np.flatnonzero(gaps > thre)[0]
    except IndexError:
        n = len(gaps)
    else:
        gap = gaps[n]
        if gap < thre_log:
            log('Pseudoinverse gap of only: {:.1e}'.format(gap))
    D[n+1:] = 0
    D[:n+1] = 1/D[:n+1]
    return U.dot(np.diag(D)).dot(V)


def cross(a, b):
    return np.array([
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0]
    ])


def fit_cubic(y0, y1, g0, g1):
    """Fit cubic polynomial to function values and derivatives at x = 0, 1.

    Returns position and function value of minimum if fit succeeds. Fit does
    not succeeds if

    1. polynomial doesn't have extrema or
    2. maximum is from (0,1) or
    3. maximum is closer to 0.5 than minimum
    """
    a = 2*(y0-y1)+g0+g1
    b = -3*(y0-y1)-2*g0-g1
    p = np.array([a, b, g0, y0])
    r = np.roots(np.polyder(p))
    if not np.isreal(r).all():
        return None, None
    r = sorted(x.real for x in r)
    if p[0] > 0:
        maxim, minim = r
    else:
        minim, maxim = r
    if 0 < maxim < 1 and abs(minim-0.5) > abs(maxim-0.5):
        return None, None
    return minim, np.polyval(p, minim)


def fit_quartic(y0, y1, g0, g1):
    """Fit constrained quartic polynomial to function values and erivatives at x = 0,1.

    Returns position and function value of minimum or None if fit fails or has
    a maximum. Quartic polynomial is constrained such that it's 2nd derivative
    is zero at just one point. This ensures that it has just one local
    extremum.  No such or two such quartic polynomials always exist. From the
    two, the one with lower minimum is chosen.
    """
    def g(y0, y1, g0, g1, c):
        a = c+3*(y0-y1)+2*g0+g1
        b = -2*c-4*(y0-y1)-3*g0-g1
        return np.array([a, b, c, g0, y0])

    def quart_min(p):
        r = np.roots(np.polyder(p))
        is_real = np.isreal(r)
        if is_real.sum() == 1:
            minim = r[is_real][0].real
        else:
            minim = r[(r == max(-abs(r))) | (r == -max(-abs(r)))][0].real
        return minim, np.polyval(p, minim)

    D = -(g0+g1)**2-2*g0*g1+6*(y1-y0)*(g0+g1)-6*(y1-y0)**2  # discriminant of d^2y/dx^2=0
    if D < 1e-11:
        return None, None
    else:
        m = -5*g0-g1-6*y0+6*y1
        p1 = g(y0, y1, g0, g1, .5*(m+np.sqrt(2*D)))
        p2 = g(y0, y1, g0, g1, .5*(m-np.sqrt(2*D)))
        if p1[0] < 0 and p2[0] < 0:
            return None, None
        [minim1, minval1] = quart_min(p1)
        [minim2, minval2] = quart_min(p2)
        if minval1 < minval2:
            return minim1, minval1
        else:
            return minim2, minval2


class FindrootException(Exception):
    pass


def findroot(f, lim):
    """Find root of increasing function on (-inf,lim).

    Assumes f(-inf) < 0, f(lim) > 0.
    """
    d = 1.
    for _ in range(1000):
        val = f(lim-d)
        if val > 0:
            break
        d = d/2  # find d so that f(lim-d) > 0
    else:
        raise RuntimeError('Cannot find f(x) > 0')
    x = lim-d  # initial guess
    dx = 1e-10  # step for numerical derivative
    fx = f(x)
    err = abs(fx)
    for _ in range(1000):
        fxpdx = f(x+dx)
        dxf = (fxpdx-fx)/dx
        x = x-fx/dxf
        fx = f(x)
        err_new = abs(fx)
        if err_new >= err:
            return x
        err = err_new
    else:
        raise FindrootException()
