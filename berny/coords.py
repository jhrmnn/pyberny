# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import division

from collections import OrderedDict
from itertools import combinations, product

import numpy as np
from numpy import dot, pi
from numpy.linalg import norm

from . import Math
from .species_data import get_property

angstrom = 1/0.52917721092  #:


class InternalCoord(object):
    def __init__(self, C=None):
        if C is not None:
            self.weak = sum(
                not C[self.idx[i], self.idx[i+1]] for i in range(len(self.idx)-1)
            )

    def __eq__(self, other):
        self.idx == other.idx

    def __hash__(self):
        return hash(self.idx)

    def __repr__(self):
        args = list(map(str, self.idx))
        if self.weak is not None:
            args.append('weak=' + str(self.weak))
        return '{}({})'.format(self.__class__.__name__, ', '.join(args))


class Bond(InternalCoord):
    def __init__(self, i, j, **kwargs):
        if i > j:
            i, j = j, i
        self.i = i
        self.j = j
        self.idx = i, j
        InternalCoord.__init__(self, **kwargs)

    def hessian(self, rho):
        return 0.45*rho[self.i, self.j]

    def weight(self, rho, coords):
        return rho[self.i, self.j]

    def center(self, ijk):
        return np.round(ijk[[self.i, self.j]].sum(0))

    def eval(self, coords, grad=False):
        v = (coords[self.i]-coords[self.j])*angstrom
        r = norm(v)
        if not grad:
            return r
        return r, [v/r, -v/r]


class Angle(InternalCoord):
    def __init__(self, i, j, k, **kwargs):
        if i > k:
            i, j, k = k, j, i
        self.i = i
        self.j = j
        self.k = k
        self.idx = i, j, k
        InternalCoord.__init__(self, **kwargs)

    def hessian(self, rho):
        return 0.15*(rho[self.i, self.j]*rho[self.j, self.k])

    def weight(self, rho, coords):
        f = 0.12
        return np.sqrt(rho[self.i, self.j]*rho[self.j, self.k]) *\
            (f+(1-f)*np.sin(self.eval(coords)))

    def center(self, ijk):
        return np.round(2*ijk[self.j])

    def eval(self, coords, grad=False):
        v1 = (coords[self.i]-coords[self.j])*angstrom
        v2 = (coords[self.k]-coords[self.j])*angstrom
        dot_product = np.dot(v1, v2)/(norm(v1)*norm(v2))
        if dot_product < -1:
            dot_product = -1
        elif dot_product > 1:
            dot_product = 1
        phi = np.arccos(dot_product)
        if not grad:
            return phi
        if abs(phi) > pi-1e-6:
            grad = [
                (pi-phi)/(2*norm(v1)**2)*v1,
                (1/norm(v1)-1/norm(v2))*(pi-phi)/(2*norm(v1))*v1,
                (pi-phi)/(2*norm(v2)**2)*v2
            ]
        else:
            grad = [
                1/np.tan(phi)*v1/norm(v1)**2-v2/(norm(v1)*norm(v2)*np.sin(phi)),
                (v1+v2)/(norm(v1)*norm(v2)*np.sin(phi)) -
                1/np.tan(phi)*(v1/norm(v1)**2+v2/norm(v2)**2),
                1/np.tan(phi)*v2/norm(v2)**2-v1/(norm(v1)*norm(v2)*np.sin(phi))
            ]
        return phi, grad


class Dihedral(InternalCoord):
    def __init__(self, i, j, k, l, weak=None, angles=None, C=None, **kwargs):
        if j > k:
            i, j, k, l = l, k, j, i
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.idx = (i, j, k, l)
        self.weak = weak
        self.angles = angles
        InternalCoord.__init__(self, **kwargs)

    def hessian(self, rho):
        return 0.005*rho[self.i, self.j]*rho[self.j, self.k]*rho[self.k, self.l]

    def weight(self, rho, coords):
        f = 0.12
        th1 = Angle(self.i, self.j, self.k).eval(coords)
        th2 = Angle(self.j, self.k, self.l).eval(coords)
        return (rho[self.i, self.j]*rho[self.j, self.k]*rho[self.k, self.l])**(1/3) * \
            (f+(1-f)*np.sin(th1))*(f+(1-f)*np.sin(th2))

    def center(self, ijk):
        return np.round(ijk[[self.j, self.k]].sum(0))

    def eval(self, coords, grad=False):
        v1 = (coords[self.i]-coords[self.j])*angstrom
        v2 = (coords[self.l]-coords[self.k])*angstrom
        w = (coords[self.k]-coords[self.j])*angstrom
        ew = w/norm(w)
        a1 = v1-dot(v1, ew)*ew
        a2 = v2-dot(v2, ew)*ew
        sgn = np.sign(np.linalg.det(np.array([v2, v1, w])))
        sgn = sgn or 1
        dot_product = dot(a1, a2)/(norm(a1)*norm(a2))
        if dot_product < -1:
            dot_product = -1
        elif dot_product > 1:
            dot_product = 1
        phi = np.arccos(dot_product)*sgn
        if not grad:
            return phi
        if abs(phi) > pi-1e-6:
            g = Math.cross(w, a1)
            g = g/norm(g)
            A = dot(v1, ew)/norm(w)
            B = dot(v2, ew)/norm(w)
            grad = [
                g/(norm(g)*norm(a1)),
                -((1-A)/norm(a1)-B/norm(a2))*g,
                -((1+B)/norm(a2)+A/norm(a1))*g,
                g/(norm(g)*norm(a2))
            ]
        elif abs(phi) < 1e-6:
            g = Math.cross(w, a1)
            g = g/norm(g)
            A = dot(v1, ew)/norm(w)
            B = dot(v2, ew)/norm(w)
            grad = [
                g/(norm(g)*norm(a1)),
                -((1-A)/norm(a1)+B/norm(a2))*g,
                ((1+B)/norm(a2)-A/norm(a1))*g,
                -g/(norm(g)*norm(a2))
            ]
        else:
            A = dot(v1, ew)/norm(w)
            B = dot(v2, ew)/norm(w)
            grad = [
                1/np.tan(phi)*a1/norm(a1)**2-a2/(norm(a1)*norm(a2)*np.sin(phi)),
                ((1-A)*a2-B*a1)/(norm(a1)*norm(a2)*np.sin(phi)) -
                1/np.tan(phi)*((1-A)*a1/norm(a1)**2-B*a2/norm(a2)**2),
                ((1+B)*a1+A*a2)/(norm(a1)*norm(a2)*np.sin(phi)) -
                1/np.tan(phi)*((1+B)*a2/norm(a2)**2+A*a1/norm(a1)**2),
                1/np.tan(phi)*a2/norm(a2)**2-a1/(norm(a1)*norm(a2)*np.sin(phi))
            ]
        return phi, grad


def get_clusters(C):
    nonassigned = list(range(len(C)))
    clusters = []
    while nonassigned:
        queue = set([nonassigned[0]])
        clusters.append([])
        while queue:
            node = queue.pop()
            clusters[-1].append(node)
            nonassigned.remove(node)
            queue.update(n for n in np.flatnonzero(C[node]) if n in nonassigned)
    C = np.zeros_like(C)
    for cluster in clusters:
        for i in cluster:
            C[i, cluster] = True
    return clusters, C


class InternalCoords(object):
    def __init__(self, geom, allowed=None, dihedral=True, superweakdih=False):
        self._coords = []
        n = len(geom)
        geom = geom.supercell()
        dist = geom.dist(geom)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in geom.species])
        bondmatrix = dist < 1.3*(radii[None, :]+radii[:, None])
        self.fragments, C = get_clusters(bondmatrix)
        radii = np.array([get_property(sp, 'vdw_radius') for sp in geom.species])
        shift = 0.
        C_total = C.copy()
        while not C_total.all():
            bondmatrix |= ~C_total & (dist < radii[None, :]+radii[:, None]+shift)
            C_total = get_clusters(bondmatrix)[1]
            shift += 1.
        for i, j in combinations(range(len(geom)), 2):
            if bondmatrix[i, j]:
                bond = Bond(i, j, C=C)
                self.append(bond)
        for j in range(len(geom)):
            for i, k in combinations(np.flatnonzero(bondmatrix[j, :]), 2):
                ang = Angle(i, j, k, C=C)
                if ang.eval(geom.coords) > pi/4:
                    self.append(ang)
        if dihedral:
            for bond in self.bonds:
                self.extend(get_dihedrals(
                    [bond.i, bond.j],
                    geom.coords,
                    bondmatrix,
                    C,
                    superweak=superweakdih,
                ))
        if geom.lattice is not None:
            self._reduce(n)

    def append(self, coord):
        self._coords.append(coord)

    def extend(self, coords):
        self._coords.extend(coords)

    def __iter__(self):
        return self._coords.__iter__()

    def __len__(self):
        return len(self._coords)

    @property
    def bonds(self):
        return [c for c in self if isinstance(c, Bond)]

    @property
    def angles(self):
        return [c for c in self if isinstance(c, Angle)]

    @property
    def dihedrals(self):
        return [c for c in self if isinstance(c, Dihedral)]

    @property
    def dict(self):
        return OrderedDict([
            ('bonds', self.bonds),
            ('angles', self.angles),
            ('dihedrals', self.dihedrals)
        ])

    def __repr__(self):
        return "<InternalCoords '{}'>".format(', '.join(
            '{}: {}'.format(name, len(coords)) for name, coords in self.dict.items()
        ))

    def __str__(self):
        ncoords = sum(len(coords) for coords in self.dict.values())
        s = 'Internal coordinates:\n'
        s += '* Number of fragments: {}\n'.format(len(self.fragments))
        s += '* Number of internal coordinates: {}\n'.format(ncoords)
        for name, coords in self.dict.items():
            for degree, adjective in [(0, 'strong'), (1, 'weak'), (2, 'superweak')]:
                n = len([None for c in coords if min(2, c.weak) == degree])
                if n > 0:
                    s += '* Number of {} {}: {}\n'.format(adjective, name, n)
        return s.rstrip()

    def eval_geom(self, geom, template=None):
        geom = geom.supercell()
        q = np.array([coord.eval(geom.coords) for coord in self])
        if template is None:
            return q
        swapped = []  # dihedrals swapped by pi
        candidates = set()  # potentially swapped angles
        for i, dih in enumerate(self):
            if not isinstance(dih, Dihedral):
                continue
            diff = q[i]-template[i]
            if abs(abs(diff)-2*pi) < pi/2:
                q[i] -= 2*pi*np.sign(diff)
            elif abs(abs(diff)-pi) < pi/2:
                q[i] -= pi*np.sign(diff)
                swapped.append(dih)
                candidates.update(dih.angles)
        for i, ang in enumerate(self):
            if not isinstance(ang, Angle) or ang not in candidates:
                continue
            # candidate angle was swapped if each dihedral that contains it was
            # either swapped or all its angles are candidates
            if all(dih in swapped or all(a in candidates for a in dih.angles)
                   for dih in self.dihedrals if ang in dih.angles):
                q[i] = 2*pi-q[i]
        return q

    def _reduce(self, n):
        idxs = np.int64(np.floor(np.array(range(3**3*n))/n))
        idxs, i = np.divmod(idxs, 3)
        idxs, j = np.divmod(idxs, 3)
        k = idxs % 3
        ijk = np.vstack((i, j, k)).T-1
        self._coords = [
            coord for coord in self._coords
            if np.all(np.isin(coord.center(ijk), [0, -1]))
        ]
        idxs = set(i for coord in self._coords for i in coord.idx)
        self.fragments = [frag for frag in self.fragments if set(frag) & idxs]

    def hessian_guess(self, geom):
        geom = geom.supercell()
        rho = geom.rho()
        return np.diag([coord.hessian(rho) for coord in self])

    def weights(self, geom):
        geom = geom.supercell()
        rho = geom.rho()
        return np.array([coord.weight(rho, geom.coords) for coord in self])

    def B_matrix(self, geom):
        geom = geom.supercell()
        B = np.zeros((len(self), len(geom), 3))
        for i, coord in enumerate(self):
            _, grads = coord.eval(geom.coords, grad=True)
            idx = [k % len(geom) for k in coord.idx]
            for j, grad in zip(idx, grads):
                B[i, j] += grad
        return B.reshape(len(self), 3*len(geom))

    def update_geom(self, geom, q, dq, B_inv, log=lambda _: None):
        geom = geom.copy()
        thre = 1e-6
        # target = CartIter(q=q+dq)
        # prev = CartIter(geom.coords, q, dq)
        for i in range(20):
            coords_new = geom.coords+B_inv.dot(dq).reshape(-1, 3)/angstrom
            dcart_rms = Math.rms(coords_new-geom.coords)
            geom.coords = coords_new
            q_new = self.eval_geom(geom, template=q)
            dq_rms = Math.rms(q_new-q)
            q, dq = q_new, dq-(q_new-q)
            if dcart_rms < thre:
                msg = 'Perfect transformation to cartesians in {} iterations'
                break
            if i == 0:
                keep_first = geom.copy(), q, dcart_rms, dq_rms
        else:
            msg = 'Transformation did not converge in {} iterations'
            geom, q, dcart_rms, dq_rms = keep_first
        log(msg.format(i+1))
        log('* RMS(dcart): {:.3}, RMS(dq): {:.3}'.format(dcart_rms, dq_rms))
        return q, geom


def get_dihedrals(center, coords, bondmatrix, C, superweak=False):
    neigh_l = [n for n in np.flatnonzero(bondmatrix[center[0], :]) if n not in center]
    neigh_r = [n for n in np.flatnonzero(bondmatrix[center[-1], :]) if n not in center]
    angles_l = [Angle(i, center[0], center[1]).eval(coords) for i in neigh_l]
    angles_r = [Angle(center[-2], center[-1], j).eval(coords) for j in neigh_r]
    nonlinear_l = [n for n, ang in zip(neigh_l, angles_l) if ang < pi-1e-3 and ang >= 1e-3]
    nonlinear_r = [n for n, ang in zip(neigh_r, angles_r) if ang < pi-1e-3 and ang >= 1e-3]
    linear_l = [n for n, ang in zip(neigh_l, angles_l) if ang >= pi-1e-3 or ang < 1e-3]
    linear_r = [n for n, ang in zip(neigh_r, angles_r) if ang >= pi-1e-3 or ang < 1e-3]
    assert len(linear_l) <= 1
    assert len(linear_r) <= 1
    if center[0] < center[-1]:
        nweak = len(list(
            None for i in range(len(center)-1)
            if not C[center[i], center[i+1]]
        ))
        dihedrals = []
        for nl, nr in product(nonlinear_l, nonlinear_r):
            if nl == nr:
                continue
            weak = nweak + \
                (0 if C[nl, center[0]] else 1) + \
                (0 if C[center[0], nr] else 1)
            if not superweak and weak > 1:
                continue
            dihedrals.append(Dihedral(
                nl,
                center[0],
                center[-1],
                nr,
                weak=weak,
                angles=(
                    Angle(nl, center[0], center[1], C=C),
                    Angle(nl, center[-2], center[-1], C=C)
                )
            ))
    else:
        dihedrals = []
    if len(center) > 3:
        pass
    elif linear_l and not linear_r:
        dihedrals.extend(get_dihedrals(linear_l + center, coords, bondmatrix, C))
    elif linear_r and not linear_l:
        dihedrals.extend(get_dihedrals(center + linear_r, coords, bondmatrix, C))
    return dihedrals
