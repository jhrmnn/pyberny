# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections import namedtuple
import numpy as np
from itertools import chain
from numpy import dot, eye
from numpy.linalg import norm

from . import Math
from .Logger import Logger
from .coords import InternalCoords


defaults = {
    'gradientmax': 0.45e-3,
    'gradientrms': 0.3e-3,
    'stepmax': 1.8e-3,
    'steprms': 1.2e-3,
    'maxsteps': 100,
    'trust': 0.3,
}


PESPoint = namedtuple('PESPoint', 'q E g')


def Berny(geom, debug=False, log=None, **params):
    params = dict(chain(defaults.items(), params.items()))
    nsteps = 0
    log = log or Logger()
    trust = params['trust']
    coords = InternalCoords(geom)
    H = coords.hessian_guess(geom)
    weights = coords.weights(geom)
    list(map(log, str(coords).split('\n')))
    best, previous, predicted, interpolated = None, None, None, None
    future = PESPoint(coords.eval_geom(geom), None, None)
    while True:
        energy, gradients = yield geom
        if debug:
            yield locals().copy()
        else:
            yield
        gradients = np.array(gradients)
        nsteps += 1
        log.n += 1
        if nsteps > params['maxsteps']:
            break
        log('Energy: {:.12}'.format(energy))
        B = coords.B_matrix(geom)
        B_inv = Math.ginv(B, log)
        current = PESPoint(future.q, energy, dot(B_inv.T, gradients.reshape(-1)))
        if nsteps > 1:
            H = update_hessian(H, current.q-best.q, current.g-best.g, log=log)
            trust = update_trust(
                trust,
                current.E-previous.E,
                predicted.E-interpolated.E,
                predicted.q-interpolated.q,
                log=log
            )
            dq = best.q-current.q
            t, E = linear_search(
                current.E, best.E, dot(current.g, dq), dot(best.g, dq), log=log
            )
            interpolated = PESPoint(current.q+t*dq, E, t*best.g+(1-t)*current.g)
        else:
            interpolated = current
        proj = dot(B, B_inv)
        H_proj = proj.dot(H).dot(proj) + 1000*(eye(len(coords))-proj)
        dq, dE, on_sphere = quadratic_step(
            dot(proj, interpolated.g), H_proj, weights, trust, log=log
        )
        predicted = PESPoint(interpolated.q+dq, interpolated.E+dE, None)
        dq = predicted.q-current.q
        log('Total step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
        q, geom = coords.update_geom(geom, current.q, predicted.q-current.q, B_inv, log=log)
        future = PESPoint(q, None, None)
        if converged(gradients, future.q-current.q, on_sphere, params, log=log):
            break
        previous = current
        if nsteps == 1 or current.E < best.E:
            best = current


def optimize(solver, geom, conv=list, **kwargs):
    next(solver)
    optimizer = Berny(geom, **kwargs)
    for geom in optimizer:
        energy, gradients = solver.send(conv(geom))
        optimizer.send((energy, gradients))
    return geom


def no_log(_, **__):
    pass


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


def converged(forces, step, on_sphere, params, log=no_log):
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
        log('* All criteria matched')
    return all_matched
