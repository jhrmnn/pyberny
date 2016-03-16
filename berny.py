import Math
from collections import namedtuple
from numpy import dot, eye
from geomlib import bohr


PESPoint = namedtuple('PESPoint', 'q E g')


class Berny:
    def __init__(self, geom, params):
        self.params = params
        self.step = 0
        self.trust = params['trust']
        self.int_coords = geom.internal_coords()
        print(self.int_coords)
        self.hessian = geom.hessian_guess()
        self.best = None

    def step(self, geom, energy, forces):
        self.step += 1
        print('Energy: {.12}'.format(energy))
        B = self.int_coords.B_matrix(geom)
        B_inv = Math.ginv(B)
        current = PESPoint(self.int_coords.eval(geom),
                           energy,
                           dot(B_inv.T, -forces*bohr))
        if self.step > 1:
            self.update_hessian(current, self.best)
            self.update_trust(current, self.best)
            dq = self.best.q-current.q
            t, E = Math.linear_search(current.E, self.best.E,
                                      dot(current.g, dq),
                                      dot(self.best.g, dq))
            interpolated = PESPoint(current.q+t*dq,
                                    E,
                                    t*self.best.g+(1-t)*current.g)
        else:
            interpolated = current
        proj = B*B_inv
        hessian_proj = proj*self.hessian*proj+1000*(eye(self.hessian.shape)-proj)
        dq, dE = Math.quadratic_step(proj*interpolated.g,
                                     hessian_proj,
                                     self.int_coords,
                                     self.trust)
        predicted = PESPoint(interpolated.q+dq, interpolated.E+dE, None)
        dq = predicted.q-current.q
        print('Total step: RMS: {.3}, max: {.3}'.format(Math.rms(dq), max(abs(dq))))
        geom, q = self.int_coords.update_geom(geom, current, predicted, B_inv)
        new = PESPoint(q, predicted.E, None)
        self.test_conv(current, interpolated, predicted, new)
        if self.step == 1 or current.E < self.best.E:
            self.best = current
        return geom

    def test_conv(self, current, interpolated, predicted, new):
        pass
