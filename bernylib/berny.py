from collections import namedtuple
import numpy as np
from numpy import dot, eye
from numpy.linalg import norm
import json

from bernylib.Logging import info, error
from bernylib import Math
from bernylib.geomlib import InternalCoords


defaults = {
    'gradientmax': 0.45e-3,
    'gradientrms': 0.3e-3,
    'stepmax': 1.8e-3,
    'steprms': 1.2e-3,
    'maxsteps': 100,
    'trust': 0.3,
    'debug': None}


class Berny(object):
    def __init__(self, geom, params=None):
        self.geom = geom.copy()
        self.params = defaults.copy()
        self.params.update(params or {})
        self.nsteps = 0
        self.trust = np.array([self.params['trust']])
        self.int_coords = InternalCoords(self.geom)
        self.hessian = self.int_coords.hessian_guess(self.geom)
        self.weights = self.int_coords.weights(self.geom)
        info.register(self)
        self.debug = []
        for line in str(self.int_coords).split('\n'):
            info(line)

    def step(self, energy, gradients):
        gradients = np.array(gradients)
        self.nsteps += 1
        info('Energy: {:.12}'.format(energy))
        B = self.int_coords.B_matrix(self.geom)
        B_inv = Math.ginv(B)
        current = PESPoint(self.int_coords.eval_geom(self.geom),
                           energy,
                           dot(B_inv.T, gradients.reshape(-1)))
        if self.nsteps > 1:
            update_hessian(self.hessian, current.q-self.best.q, current.g-self.best.g)
            update_trust(self.trust,
                         current.E-self.previous.E,
                         self.predicted.E-self.interpolated.E,
                         self.predicted.q-self.interpolated.q)
            dq = self.best.q-current.q
            t, E = linear_search(
                current.E, self.best.E, dot(current.g, dq), dot(self.best.g, dq))
            self.interpolated = PESPoint(
                current.q+t*dq, E, t*self.best.g+(1-t)*current.g)
        else:
            self.interpolated = current
        proj = dot(B, B_inv)
        hessian_proj = proj.dot(self.hessian).dot(proj) +\
            1000*(eye(len(self.int_coords))-proj)
        dq, dE, on_sphere = quadratic_step(
            dot(proj, self.interpolated.g), hessian_proj, self.weights, self.trust[0])
        self.predicted = PESPoint(self.interpolated.q+dq, self.interpolated.E+dE, None)
        dq = self.predicted.q-current.q
        info('Total step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
        geom = self.geom.copy()
        q = self.int_coords.update_geom(self.geom, current.q, self.predicted.q-current.q, B_inv)
        future = PESPoint(q, None, None)
        if self.params['debug']:
            self.debug.append({'nstep': self.nsteps,
                               'trust': self.trust[0],
                               'hessian': self.hessian.copy(),
                               'gradients': gradients,
                               'coords': geom.coords,
                               'energy': energy,
                               'q': current.q,
                               'dq': dq})
            with open(self.params['debug'], 'w') as f:
                json.dump(self.debug, f, indent=4, cls=ArrayEncoder)
        if converged(gradients, future.q-current.q, on_sphere, self.params):
            return
        self.previous = current
        if self.nsteps == 1 or current.E < self.best.E:
            self.best = current
        return self.geom.copy()


PESPoint = namedtuple('PESPoint', 'q E g')


def update_hessian(H, dq, dg):
    dH = dg[None, :]*dg[:, None]/dot(dq, dg) - \
        H.dot(dq[None, :]*dq[:, None]).dot(H)/dq.dot(H).dot(dq)  # BFGS update
    info('Hessian update information:')
    info('* Change: RMS: {:.3}, max: {:.3}'.format(Math.rms(dH), abs(dH).max()))
    H[:, :] = H+dH


def update_trust(trust, dE, dE_predicted, dq):
    if dE != 0:
        r = dE/dE_predicted  # Fletcher's parameter
    else:
        r = 1.
    info("Trust update: Fletcher's parameter: {:.3}".format(r))
    if r < 0.25:
        tr = norm(dq)/4
    elif r > 0.75 and abs(norm(dq)-trust) < 1e-10:
        tr = 2*trust
    else:
        tr = trust
    trust[:] = tr


def linear_search(E0, E1, g0, g1):
    info('Linear interpolation:')
    info('* Energies: {:.8}, {:.8}'.format(E0, E1))
    info('* Derivatives: {:.3}, {:.3}'.format(g0, g1))
    t, E = Math.fit_quartic(E0, E1, g0, g1)
    if t is None or t < -1 or t > 2:
        t, E = Math.fit_cubic(E0, E1, g0, g1)
        if t is None or t < 0 or t > 1:
            if E0 <= E1:
                info('* No fit succeeded, staying in new point')
                return 0, E0

            else:
                info('* No fit succeeded, returning to best point')
                return 1, E1
        else:
            msg = 'Cubic interpolation was performed'
    else:
        msg = 'Quartic interpolation was performed'
    info('* {}: t = {:.3}'.format(msg, t))
    info('* Interpolated energy: {:.8}'.format(E))
    return t, E


def quadratic_step(g, H, w, trust):
    ev = np.linalg.eigvalsh((H+H.T)/2)
    rfo = np.vstack((np.hstack((H, g[:, None])),
                     np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo+rfo.T)/2)
    dq = V[:-1, 0]/V[-1, 0]
    l = D[0]
    if norm(dq) <= trust:
        info('Pure RFO step was performed:')
        on_sphere = False
    else:
        def steplength(l):
            return norm(np.linalg.solve(l*eye(H.shape[0])-H, g))-trust
        l = Math.findroot(steplength, ev[0])  # minimization on sphere
        dq = np.linalg.solve(l*eye(H.shape[0])-H, g)
        on_sphere = False
        info('Minimization on sphere was performed:')
    dE = dot(g, dq)+0.5*dq.dot(H).dot(dq)  # predicted energy change
    info('* Trust radius: {:.2}'.format(trust))
    info('* Number of negative eigenvalues: {}'.format((ev < 0).sum()))
    info('* Lowest eigenvalue: {:.3}'.format(ev[0]))
    info('* lambda: {:.3}'.format(l))
    info('Quadratic step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
    info('* Predicted energy change: {:.3}'.format(dE))
    return dq, dE, on_sphere


def converged(forces, step, on_sphere, params):
    criteria = [
        ('Gradient RMS', Math.rms(forces), params['gradientrms']),
        ('Gradient maximum', np.max(abs(forces)), params['gradientmax'])]
    if on_sphere:
        criteria.append(('Minimization on sphere', False))
    else:
        criteria.extend([
            ('Step RMS', Math.rms(step), params['steprms']),
            ('Step maximum', np.max(abs(step)), params['stepmax'])])
    info('Convergence criteria:')
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
        info(msg)
        if not result:
            all_matched = False
    if all_matched:
        info('* All criteria matched')
    return all_matched


def optimize_morse(geom, r0=None, **params):
    geom = geom.copy()
    berny = Berny(geom)
    debug = []
    while True:
        energy, gradients = geom.morse(r0=r0)
        debug.append({'energy': energy,
                      'gradients': gradients,
                      'geom': geom.copy()})
        try:
            geom = berny.step(energy, gradients)
        except Math.FindrootException:
            error('Could not find root of RFO, bad Hessian, quitting')
            break
        if not geom:
            break
    return debug


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is np.nan:
            return None
        try:
            return obj.tolist()
        except AttributeError:
            return super().default(obj)
