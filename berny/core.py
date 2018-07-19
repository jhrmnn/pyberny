# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections import namedtuple
from itertools import chain

import numpy as np
from numpy import dot, eye
from numpy.linalg import norm

from . import Math
from .coords import InternalCoords
from .Logger import Logger

defaults = {
    'gradientmax': 0.45e-3,
    'gradientrms': 0.3e-3,
    'stepmax': 1.8e-3,
    'steprms': 1.2e-3,
    'trust': 0.3,
    'dihedral': True,
    'superweakdih': False,
}
"""
- gradientmax, gradientrms, stepmax, steprms:
    Convergence criteria in atomic units ("step" refers to the step in
    internal coordinates, assuming radian units for angles).

- trust:
    Initial trust radius in atomic units. It is the maximum RMS of the
    quadratic step (see below).

- dihedral:
    Form dihedral angles.

- superweakdih:
    Form dihedral angles containing two or more noncovalent bonds.
"""

PESPoint = namedtuple('PESPoint', 'q E g')


def Berny(geom, log=None, debug=False, restart=None, maxsteps=100,
          verbosity=None, **params):
    """
    Coroutine that receives energy and gradients and yields the next geometry.

    :param Gometry geom: geometry to start with
    :param Logger log: used for logging if given
    :param bool debug: if True, the generator yields debug info on receiving
        the energy and gradients, otherwise it yields None
    :param dict restart: start from a state saved from previous run using ``debug=True``
    :param int maxsteps: abort after maximum number of steps
    :param int verbosity: if present and log is None, specifies the verbosity of
        the default :py:class:`~berny.Logger`
    :param params: parameters that override the :py:data:`~berny.core.defaults`

    The coroutine is to be used as follows::

        optimizer = Berny(geom)
        for geom in optimizer:
            # calculate energy and gradients (as N-by-3 matrix)
            debug = optimizer.send((energy, gradients))
    """
    log = log or Logger(verbosity=verbosity or 0)
    algo = BernyAlgo(geom, params)
    if restart:
        algo.__dict__.update(restart)
    else:
        algo.init(log=log)
    for _ in range(maxsteps):
        log.n += 1
        energy, gradients = yield algo.geom
        converged = algo.step(energy, gradients, log=log)
        if debug:
            yield vars(algo).copy()
        else:
            yield
        if converged:
            break
    else:
        log('Maximum number of steps reached')


def no_log(msg, **kwargs):
    pass


class BernyAlgo(object):
    def __init__(self, geom, params):
        self.geom = geom
        self.params = dict(chain(defaults.items(), params.items()))

    def init(s, log=no_log):
        s.trust = s.params['trust']
        s.coords = InternalCoords(
            s.geom,
            dihedral=s.params['dihedral'],
            superweakdih=s.params['superweakdih'],
        )
        s.H = s.coords.hessian_guess(s.geom)
        s.weights = s.coords.weights(s.geom)
        for line in str(s.coords).split('\n'):
            log(line)
        s.future = PESPoint(s.coords.eval_geom(s.geom), None, None)
        s.first = True

    def step(s, energy, gradients, log=no_log):
        gradients = np.array(gradients)
        log('Energy: {:.12}'.format(energy), level=1)
        B = s.coords.B_matrix(s.geom)
        B_inv = B.T.dot(Math.pinv(np.dot(B, B.T), log=log))
        current = PESPoint(s.future.q, energy, dot(B_inv.T, gradients.reshape(-1)))
        if not s.first:
            s.H = update_hessian(
                s.H, current.q-s.best.q, current.g-s.best.g, log=log
            )
            s.trust = update_trust(
                s.trust,
                current.E-s.previous.E,
                s.predicted.E-s.interpolated.E,
                s.predicted.q-s.interpolated.q,
                log=log
            )
            dq = s.best.q-current.q
            t, E = linear_search(
                current.E, s.best.E, dot(current.g, dq), dot(s.best.g, dq),
                log=log
            )
            s.interpolated = PESPoint(current.q+t*dq, E, t*s.best.g+(1-t)*current.g)
        else:
            s.interpolated = current
        if s.trust < 1e-6:
            raise RuntimeError('The trust radius got too small, check forces?')
        proj = dot(B, B_inv)
        H_proj = proj.dot(s.H).dot(proj) + 1000*(eye(len(s.coords))-proj)
        dq, dE, on_sphere = quadratic_step(
            dot(proj, s.interpolated.g), H_proj, s.weights, s.trust, log=log
        )
        s.predicted = PESPoint(s.interpolated.q+dq, s.interpolated.E+dE, None)
        dq = s.predicted.q-current.q
        log('Total step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
        q, s.geom = s.coords.update_geom(
            s.geom, current.q, s.predicted.q-current.q, B_inv, log=log
        )
        s.future = PESPoint(q, None, None)
        converged = is_converged(
            gradients, s.future.q-current.q, on_sphere, s.params, log=log
        )
        s.previous = current
        if s.first or current.E < s.best.E:
            s.best = current
        s.first = False
        return converged


def optimize(solver, geom, **kwargs):
    """
    Optimize a geometry with respect to a solver.

    :param generator solver: unprimed generator that receives geometry as a
        2-tuple of a list of 2-tuples of the atom symbol and coordinate (as a
        3-tuple), and of a list of lattice vectors (or None if molecule), and
        yields the energy and gradients (as a *N*-by-3 matrix or (*N*+3)-by-3
        matrix in case of a crystal geometry)
    :param Geometry geom: geometry to optimize
    :param kwargs: these are handed over to :py:func:`Berny`

    Returns the optimized geometry.

    Inside the function, the solver is used as follows::

        next(solver)
        energy, gradients = solver.send((list(geom), geom.lattce))
        energy, gradients = solver.send((list(geom), geom.lattce))
        ...
    """
    kwargs.setdefault('log', Logger(verbosity=kwargs.pop('verbosity', -1)))
    next(solver)
    optimizer = Berny(geom, **kwargs)
    for geom in optimizer:
        energy, gradients = solver.send((list(geom), geom.lattice))
        optimizer.send((energy, gradients))
    return geom


def update_hessian(H, dq, dg, log=no_log):
    dH = dg[None, :]*dg[:, None]/dot(dq, dg) - \
        H.dot(dq[None, :]*dq[:, None]).dot(H)/dq.dot(H).dot(dq)  # BFGS update
    log('Hessian update information:')
    log('* Change: RMS: {:.3}, max: {:.3}'.format(Math.rms(dH), abs(dH).max()))
    return H+dH


def update_trust(trust, dE, dE_predicted, dq, log=no_log):
    if dE != 0:
        r = dE/dE_predicted  # Fletcher's parameter
    else:
        r = 1.
    log("Trust update: Fletcher's parameter: {:.3}".format(r))
    if r < 0.25:
        return norm(dq)/4
    elif r > 0.75 and abs(norm(dq)-trust) < 1e-10:
        return 2*trust
    else:
        return trust


def linear_search(E0, E1, g0, g1, log=no_log):
    log('Linear interpolation:')
    log('* Energies: {:.8}, {:.8}'.format(E0, E1))
    log('* Derivatives: {:.3}, {:.3}'.format(g0, g1))
    t, E = Math.fit_quartic(E0, E1, g0, g1)
    if t is None or t < -1 or t > 2:
        t, E = Math.fit_cubic(E0, E1, g0, g1)
        if t is None or t < 0 or t > 1:
            if E0 <= E1:
                log('* No fit succeeded, staying in new point')
                return 0, E0

            else:
                log('* No fit succeeded, returning to best point')
                return 1, E1
        else:
            msg = 'Cubic interpolation was performed'
    else:
        msg = 'Quartic interpolation was performed'
    log('* {}: t = {:.3}'.format(msg, t))
    log('* Interpolated energy: {:.8}'.format(E))
    return t, E


def quadratic_step(g, H, w, trust, log=no_log):
    ev = np.linalg.eigvalsh((H+H.T)/2)
    rfo = np.vstack((np.hstack((H, g[:, None])),
                     np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo+rfo.T)/2)
    dq = V[:-1, 0]/V[-1, 0]
    l = D[0]
    if norm(dq) <= trust:
        log('Pure RFO step was performed:')
        on_sphere = False
    else:
        def steplength(l):
            return norm(np.linalg.solve(l*eye(H.shape[0])-H, g))-trust
        l = Math.findroot(steplength, ev[0])  # minimization on sphere
        dq = np.linalg.solve(l*eye(H.shape[0])-H, g)
        on_sphere = False
        log('Minimization on sphere was performed:')
    dE = dot(g, dq)+0.5*dq.dot(H).dot(dq)  # predicted energy change
    log('* Trust radius: {:.2}'.format(trust))
    log('* Number of negative eigenvalues: {}'.format((ev < 0).sum()))
    log('* Lowest eigenvalue: {:.3}'.format(ev[0]))
    log('* lambda: {:.3}'.format(l))
    log('Quadratic step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
    log('* Predicted energy change: {:.3}'.format(dE))
    return dq, dE, on_sphere


def is_converged(forces, step, on_sphere, params, log=no_log):
    criteria = [
        ('Gradient RMS', Math.rms(forces), params['gradientrms']),
        ('Gradient maximum', np.max(abs(forces)), params['gradientmax'])
    ]
    if on_sphere:
        criteria.append(('Minimization on sphere', False))
    else:
        criteria.extend([
            ('Step RMS', Math.rms(step), params['steprms']),
            ('Step maximum', np.max(abs(step)), params['stepmax'])
        ])
    log('Convergence criteria:')
    all_matched = True
    for crit in criteria:
        if len(crit) > 2:
            result = crit[1] < crit[2]
            msg = '{:.3} {} {:.3}'.format(crit[1], '<' if result else '>', crit[2])
        else:
            result = crit[2]
            msg = None
        msg = '{}: {}'.format(crit[0], msg) if msg else crit[0]
        msg = '* {} => {}'.format(msg, 'OK' if result else 'no')
        log(msg)
        if not result:
            all_matched = False
    if all_matched:
        log('* All criteria matched', level=1)
    return all_matched
