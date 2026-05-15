# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import logging
from collections import namedtuple
from collections.abc import Generator
from itertools import chain

import numpy as np
from numpy import dot, eye
from numpy.linalg import norm

from . import Math
from .coords import InternalCoords

__version__ = '0.3.2'
__all__ = ['Berny']

log = logging.getLogger(__name__)

defaults = {
    'gradientmax': 0.45e-3,
    'gradientrms': 0.15e-3,
    'stepmax': 1.8e-3,
    'steprms': 1.2e-3,
    'trust': 0.3,
    'dihedral': True,
    'superweakdih': False,
}
"""
``gradientmax``, ``gradientrms``, ``stepmax``, ``steprms``
    Convergence criteria in atomic units ("step" refers to the step in
    internal coordinates, assuming radian units for angles).

``trust``
    Initial trust radius in atomic units. It is the maximum RMS of the
    quadratic step (see below).

``dihedral``
    Form dihedral angles.

``superweakdih``
    Form dihedral angles containing two or more noncovalent bonds.
"""


OptPoint = namedtuple('OptPoint', 'q E g')


class BernyAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '{} {}'.format(self.extra['step'], msg), kwargs


class Berny(Generator):
    """Generator that receives energy and gradients and yields the next geometry.

    Args:
        geom (:class:`~berny.Geometry`): geometry to start with
        debug (bool): if :data:`True`, the generator yields debug info on receiving
            the energy and gradients, otherwise it yields :data:`None`
        restart (dict): start from a state saved from previous run
            using ``debug=True``
        maxsteps (int): abort after maximum number of steps
        logger (:class:`logging.Logger`): alternative logger to use
        params: parameters that override the :data:`~berny.berny.defaults`

    The Berny object is to be used as follows::

        optimizer = Berny(geom)
        for geom in optimizer:
            # calculate energy and gradients (as N-by-3 matrix)
            debug = optimizer.send((energy, gradients))
    """

    class State(object):
        pass

    def __init__(
        self, geom, debug=False, restart=None, maxsteps=100, logger=None, **params
    ):
        self._debug = debug
        self._maxsteps = maxsteps
        self._converged = False
        self._n = 0
        self._log = BernyAdapter(logger or log, {'step': self._n})
        s = self._state = Berny.State()
        if restart:
            vars(s).update(restart)
            return
        s.geom = geom
        s.params = dict(chain(defaults.items(), params.items()))
        s.trust = s.params['trust']
        s.coords = InternalCoords(
            s.geom, dihedral=s.params['dihedral'], superweakdih=s.params['superweakdih']
        )
        s.H = s.coords.hessian_guess(s.geom)
        s.weights = s.coords.weights(s.geom)
        s.future = OptPoint(s.coords.eval_geom(s.geom), None, None)
        s.first = True
        for line in str(s.coords).split('\n'):
            self._log.info(line)

    def __next__(self):
        assert self._n <= self._maxsteps
        if self._n == self._maxsteps or self._converged:
            raise StopIteration
        self._n += 1
        return self._state.geom

    @property
    def trust(self):
        """Current trust radius."""
        return self._state.trust

    @property
    def converged(self):
        """Whether the optimized has converged."""
        return self._converged

    def send(self, energy_and_gradients):  # noqa: D102
        self._log.extra['step'] = self._n
        log, s = self._log.info, self._state
        energy, gradients = energy_and_gradients
        gradients = np.array(gradients)
        log('Energy: {:.12}'.format(energy))
        B = s.coords.B_matrix(s.geom)
        B_inv = B.T.dot(Math.pinv(np.dot(B, B.T), log=log))
        current = OptPoint(s.future.q, energy, dot(B_inv.T, gradients.reshape(-1)))
        if not s.first:
            s.H = update_hessian(
                s.H, current.q - s.best.q, current.g - s.best.g, log=log
            )
            s.trust = update_trust(
                s.trust,
                current.E - s.previous.E,  # or should it be s.interpolated.E?
                s.predicted.E - s.interpolated.E,
                s.predicted.q - s.interpolated.q,
                log=log,
            )
            dq = s.best.q - current.q
            t, E = linear_search(
                current.E, s.best.E, dot(current.g, dq), dot(s.best.g, dq), log=log
            )
            s.interpolated = OptPoint(
                current.q + t * dq, E, current.g + t * (s.best.g - current.g)
            )
        else:
            s.interpolated = current
        if s.trust < 1e-6:
            raise RuntimeError('The trust radius got too small, check forces?')
        proj = dot(B, B_inv)
        H_proj = proj.dot(s.H).dot(proj) + 1000 * (eye(len(s.coords)) - proj)
        dq, dE, on_sphere = quadratic_step(
            dot(proj, s.interpolated.g), H_proj, s.weights, s.trust, log=log
        )
        s.predicted = OptPoint(s.interpolated.q + dq, s.interpolated.E + dE, None)
        dq = s.predicted.q - current.q
        log('Total step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
        q, s.geom = s.coords.update_geom(
            s.geom, current.q, s.predicted.q - current.q, B_inv, log=log
        )
        s.future = OptPoint(q, None, None)
        s.previous = current
        if s.first or current.E < s.best.E:
            s.best = current
        s.first = False
        self._converged = is_converged(
            current.g, s.future.q - current.q, on_sphere, s.params, log=log
        )
        if self._n == self._maxsteps:
            log('Maximum number of steps reached')
        if self._debug:
            return vars(s).copy()

    def throw(self, *args, **kwargs):  # noqa: D102
        return Generator.throw(self, *args, **kwargs)


def no_log(msg, **kwargs):
    pass


def update_hessian(H, dq, dg, log=no_log):
    dH1 = dg[None, :] * dg[:, None] / dot(dq, dg)
    dH2 = H.dot(dq[None, :] * dq[:, None]).dot(H) / dq.dot(H).dot(dq)
    dH = dH1 - dH2  # BFGS update
    log('Hessian update information:')
    log('* Change: RMS: {:.3}, max: {:.3}'.format(Math.rms(dH), abs(dH).max()))
    return H + dH


def update_trust(trust, dE, dE_predicted, dq, log=no_log):
    if dE != 0:
        r = dE / dE_predicted  # Fletcher's parameter
    else:
        r = 1.0
    log("Trust update: Fletcher's parameter: {:.3}".format(r))
    if r < 0.25:
        return norm(dq) / 4
    elif r > 0.75 and abs(norm(dq) - trust) < 1e-10:
        return 2 * trust
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
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    rfo = np.vstack((np.hstack((H, g[:, None])), np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo + rfo.T) / 2)
    dq = V[:-1, 0] / V[-1, 0]
    l = D[0]
    if norm(dq) <= trust:
        log('Pure RFO step was performed:')
        on_sphere = False
    else:

        def steplength(l):
            return norm(np.linalg.solve(l * eye(H.shape[0]) - H, g)) - trust

        l = Math.findroot(steplength, ev[0])  # minimization on sphere
        dq = np.linalg.solve(l * eye(H.shape[0]) - H, g)
        on_sphere = True
        log('Minimization on sphere was performed:')
    dE = dot(g, dq) + 0.5 * dq.dot(H).dot(dq)  # predicted energy change
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
        ('Gradient maximum', np.max(abs(forces)), params['gradientmax']),
    ]
    if on_sphere:
        criteria.append(('Minimization on sphere', False))
    else:
        criteria.extend(
            [
                ('Step RMS', Math.rms(step), params['steprms']),
                ('Step maximum', np.max(abs(step)), params['stepmax']),
            ]
        )
    log('Convergence criteria:')
    all_matched = True
    for crit in criteria:
        if len(crit) > 2:
            result = crit[1] < crit[2]
            msg = '{:.3} {} {:.3}'.format(crit[1], '<' if result else '>', crit[2])
        else:
            msg, result = crit
        msg = '{}: {}'.format(crit[0], msg) if msg else crit[0]
        msg = '* {} => {}'.format(msg, 'OK' if result else 'no')
        log(msg)
        if not result:
            all_matched = False
    if all_matched:
        log('* All criteria matched')
    return all_matched
