import Math
from collections import namedtuple
import numpy as np
from numpy import dot, eye
from numpy.linalg import norm
from geomlib import bohr, InternalCoords


PESPoint = namedtuple('PESPoint', 'q E g')

defaults = {
    'gradientmax': 0.45e-3,
    'gradientrms': 0.3e-3,
    'stepmax': 1.8e-3,
    'steprms': 1.2e-3,
    'maxsteps': 100,
    'trust': 0.3,
    'debug': False}


class Berny:
    def __init__(self, geom, params=None):
        self.params = defaults.copy()
        self.params.update(params or {})
        self.nsteps = 0
        self.trust = self.params['trust']
        self.int_coords = InternalCoords(geom)
        self.hessian = self.int_coords.hessian_guess(geom)
        self.weights = self.int_coords.weights(geom)
        print(self.int_coords)

    def step(self, geom, energy, gradients):
        gradients = gradients*bohr
        self.nsteps += 1
        print('Energy: {:.12}'.format(energy))
        B = self.int_coords.B_matrix(geom)
        B_inv = Math.ginv(B)
        current = PESPoint(self.int_coords.eval(geom),
                           energy,
                           dot(B_inv.T, gradients.reshape(-1)))
        if self.nsteps > 1:
            self.hessian = update_hessian(self.hessian,
                                          current.q-self.best.q,
                                          current.g-self.best.g)
            self.trust = update_trust(self.trust,
                                      current.E-self.previous.E,
                                      self.predicted.E-self.interpolated.E,
                                      self.predicted.q-self.interpolated.q)
            dq = self.best.q-current.q
            t, E = linear_search(current.E, self.best.E,
                                 dot(current.g, dq),
                                 dot(self.best.g, dq))
            self.interpolated = PESPoint(current.q+t*dq,
                                         E,
                                         t*self.best.g+(1-t)*current.g)
        else:
            self.interpolated = current
        proj = dot(B, B_inv)
        hessian_proj = proj.dot(self.hessian).dot(proj) +\
            1000*(eye(len(self.int_coords))-proj)
        dq, dE, on_sphere = quadratic_step(dot(proj, self.interpolated.g),
                                           hessian_proj,
                                           self.weights,
                                           self.trust)
        self.predicted = PESPoint(self.interpolated.q+dq, self.interpolated.E+dE, None)
        dq = self.predicted.q-current.q
        print('Total predicted step: RMS: {:.3}, max: {:.3}'
              .format(Math.rms(dq), max(abs(dq))))
        geom, q = self.int_coords.update_geom(geom, current, self.predicted, B_inv)
        dq = q-current.q
        print('Total actual step: RMS: {:.3}, max: {:.3}'
              .format(Math.rms(dq), max(abs(dq))))
        if converged(gradients, q-current.q, on_sphere, self.params):
            return
        self.previous = current
        if self.nsteps == 1 or current.E < self.best.E:
            self.best = current
        return geom


def update_hessian(H, dq, dg):
    dH = dg[None, :]*dg[:, None]/dot(dq, dg) - \
        H.dot(dq[None, :]*dq[:, None]).dot(H)/dq.dot(H).dot(dq)  # BFGS update
    print('Hessian update information:')
    print('* Change: RMS: {}, max: {}'.format(Math.rms(dH), abs(dH).max()))
    return H+dH


def update_trust(dE, dE_predicted, dq, trust):
    if dE != 0:
        r = dE/dE_predicted  # Fletcher's parameter
    else:
        r = 1.
    if r < 0.25:
        trust = norm(dq)/4
    elif r > 0.75 and abs(norm(dq)-trust) < 1e-10:
        trust = 2*trust
    print("Trust update: Fletcher's parameter: {}".format(r))
    return trust


def linear_search(E0, E1, g0, g1):
    print('Linear interpolation:')
    print('* Energies: {}, {}'.format(E0, E1))
    print('* Derivatives: {}, {}'.format(g0, g1))
    t, E = Math.fit_quartic(E0, E1, g0, g1)
    if t is None or t < -1 or t > 2:
        t, E = Math.fit_cubic(E0, E1, g0, g1)
        if t is None or t < 0 or t > 1:
            if E0 <= E1:
                print('* No fit succeeded, staying in new point')
                return 0, E0

            else:
                print('* No fit succeeded, returning to best point')
                return 1, E1
        else:
            msg = 'Cubic interpolation was performed'
    else:
        msg = 'Quartic interpolation was performed'
    print('* {}: t = {}'.format(msg, t))
    print('* Interpolated energy: {}'.format(E))
    return t, E


def quadratic_step(g, H, w, trust):
    ev = np.linalg.eigvalsh((H+H.T)/2)
    rfo = np.vstack((np.hstack((H, g[:, None])),
                     np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo+rfo.T)/2)
    dq = V[:-1, 0]/V[-1, 0]
    l = D[0]
    if norm(dq) <= trust:
        print('Pure RFO step was performed:')
        on_sphere = False
    else:
        def steplength(l):
            return norm(np.linalg.solve(l*eye(H.shape[0])-H, g))-trust
        l = Math.findroot(steplength, ev[0])  # minimization on sphere
        dq = np.linalg.solve(l*eye(H.shape[0])-H, g)
        on_sphere = False
        print('Minimization on sphere was performed:')
    dE = dot(g, dq)+0.5*dq.dot(H).dot(dq)  # predicted energy change
    print('* Trust radius: {}'.format(trust))
    print('* Number of negative eigenvalues: {}'.format((ev < 0).sum()))
    print('* Lowest eigenvalue: {}'.format(ev[0]))
    print('* lambda: {}'.format(l))
    print('Quadratic step: RMS: {}, max: {}'.format(Math.rms(dq), max(abs(dq))))
    print('* Predicted energy change: {}'.format(dE))
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
    print('Convergence criteria:')
    all_matched = True
    for crit in criteria:
        if len(crit) > 2:
            result = crit[1] < crit[2]
            msg = '{} {} {}'.format(crit[1], '<' if result else '>', crit[2])
        else:
            result = crit[2]
            msg = None
        msg = '{}: {}'.format(crit[0], msg) if msg else crit[0]
        msg = '* {} => {}'.format(msg, 'OK' if result else 'no')
        print(msg)
        if not result:
            all_matched = False
    if all_matched:
        print('* All criteria matched')
    return all_matched
