import numpy as np
from numpy import dot, pi
from numpy.linalg import norm, inv
from collections import defaultdict, OrderedDict
from itertools import chain, product, repeat, combinations
import os
import csv
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from bernylib import Math
from bernylib.Logging import info

bohr = 0.52917721092


class Molecule(object):
    def __init__(self, species, coords):
        self.species = species
        self.coords = np.array(coords)

    def __repr__(self):
        return "<{} '{}'>".format(self.__class__.__name__, self.formula)

    def __getattr__(self, attr):
        if attr == 'formula':
            counter = defaultdict(int)
            for specie in self.species:
                counter[specie] += 1
            return ''.join('{}{}'.format(sp, n if n > 1 else '')
                           for sp, n in sorted(counter.items()))
        else:
            raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(self.__class__.__name__, attr))

    def __iter__(self):
        for specie, coord in zip(self.species, self.coords):
            yield specie, coord

    def __len__(self):
        return len(self.species)

    def __format__(self, fmt):
        fp = StringIO()
        self.dump(fp, fmt)
        return fp.getvalue()

    dumps = __format__

    def dump(self, fp, fmt):
        if fmt == '':
            fp.write(repr(self))
        elif fmt == 'xyz':
            fp.write('{}\n'.format(len(self)))
            fp.write('Formula: {}\n'.format(self.formula))
            for specie, coord in self:
                fp.write('{:>2} {}\n'.format(
                    specie, ' '.join('{:15.8}'.format(x) for x in coord)))
        elif fmt == 'aims':
            fp.write('# Formula: {}\n'.format(self.formula))
            for specie, coord in self:
                fp.write('atom {} {:>2}\n'.format(
                    ' '.join('{:15.8}'.format(x) for x in coord), specie))
        else:
            raise ValueError("Unknown format: '{}'".format(fmt))

    def copy(self):
        return Molecule(list(self.species), self.coords.copy())

    def supercell(self, *args, **kwargs):
        return self.copy()

    def draw(self, method='imolecule', **kwargs):
        if method == 'imolecule':
            import imolecule
            imolecule.draw(self.to_json(), 'json', **kwargs)

    def to_json(self):
        bond = self.bondmatrix()
        return {'atoms': [{'element': specie, 'location': coord.tolist()}
                          for specie, coord in self],
                'bonds': [{'atoms': [i, j], 'order': 1}
                          for i, j in combinations(range(len(self)), 2)
                          if bond[i, j]]}

    def dist(self, geom, ret_diff=False):
        diff = self.coords[:, None, :]-geom.coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, 2))
        dist[np.diag_indices(len(self))] = np.inf
        if ret_diff:
            return dist, diff
        else:
            return dist

    def bondmatrix(self, scale=1.3):
        dist = self.dist(self)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in self.species])
        return dist < 1.3*(radii[None, :]+radii[:, None])

    def rho(self):
        geom = self.supercell()
        dist = geom.dist(geom)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in geom.species])
        return np.exp(-dist/(radii[None, :]+radii[:, None])+1)

    def morse(self, cutoff=20., r0=None):
        def V(r, r0):
            return (1-np.exp(-(r-r0)))**2-1

        def dVdr(r, r0):
            return 2*(1-np.exp(-(r-r0)))*np.exp(-(r-r0))

        geom = self.supercell(cutoff=cutoff)
        dist, diff = self.dist(geom, ret_diff=True)
        if not r0:
            radii = np.array([get_property(sp, 'covalent_radius') for sp in geom.species])
            r0 = radii[:len(self), None]+radii[None, :]
        E = np.sum(V(dist, r0))/2
        dEdR = np.sum(dVdr(dist, r0)[:, :, None]*diff/dist[:, :, None], 1)
        return E, dEdR


def load(fp, fmt):
    if fmt == 'xyz':
        n = int(fp.readline())
        fp.readline()
        species = []
        coords = []
        for _ in range(n):
            l = fp.readline().split()
            species.append(l[0])
            coords.append([float(x) for x in l[1:4]])
        return Molecule(species, coords)
    if fmt == 'aims':
        species = []
        coords = []
        lattice = []
        while True:
            l = fp.readline()
            if l == '':
                break
            l = l.strip()
            if not l or l.startswith('#'):
                continue
            l = l.split()
            what = l[0]
            if what == 'atom':
                species.append(l[4])
                coords.append([float(x) for x in l[1:4]])
            elif what == 'lattice_vector':
                lattice.append([float(x) for x in l[1:4]])
        if lattice:
            assert len(lattice) == 3
            return Crystal(species, coords, lattice)
        else:
            return Molecule(species, coords)


def readfile(path, fmt=None):
    if not fmt:
        ext = os.path.splitext(path)[1]
        if ext == '.xyz':
            fmt = 'xyz'
        if ext == '.aims' or os.path.basename(path) == 'geometry.in':
            fmt = 'aims'
    with open(path) as f:
        return load(f, fmt)


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


class Crystal(Molecule):
    def __init__(self, species, coords, lattice):
        self.lattice = np.array(lattice)
        super(Crystal, self).__init__(species, coords)

    def copy(self):
        return Crystal(list(self.species), self.coords.copy(), self.lattice.copy())

    def super_circum(self, radius):
        rec_lattice = 2*pi*inv(self.lattice.T)
        layer_sep = np.array(
            [sum(vec*rvec/norm(rvec))
             for vec, rvec in zip(self.lattice, rec_lattice)])
        return np.array([int(n) for n in np.ceil(radius/layer_sep+1/2)])

    def supercell(self, ranges=((-1, 1), (-1, 1), (-1, 1)), cutoff=None):
        if cutoff:
            ranges = [(-r, r) for r in self.super_circum(cutoff)]
        else:
            assert ranges
        latt_vectors = np.array([(0, 0, 0)] + [
            sum(k*vec for k, vec in zip(shift, self.lattice))
            for shift
            in product(*[range(a, b+1) for a, b in ranges])
            if shift != (0, 0, 0)])
        species = list(chain.from_iterable(repeat(self.species, len(latt_vectors))))
        coords = (self.coords[None, :, :]+latt_vectors[:, None, :]).reshape((-1, 3))
        lattice = self.lattice*np.array([b-a for a, b in ranges])[:, None]
        return Crystal(species, coords, lattice)


class InternalCoord(object):
    def __init__(self, C=None):
        if C is not None:
            self.weak = sum(not C[self.idx[i], self.idx[i+1]]
                            for i in range(len(self.idx)-1))

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
    def __init__(self, i, j, *args, **kwargs):
        if i > j:
            i, j = j, i
        self.i = i
        self.j = j
        self.idx = (i, j)
        super(Bond, self).__init__(*args, **kwargs)

    def hessian(self, rho):
        return 0.45*rho[self.i, self.j]

    def weight(self, rho, coords):
        return rho[self.i, self.j]

    def eval(self, coords, grad=False):
        v = (coords[self.i]-coords[self.j])/bohr
        r = norm(v)
        if not grad:
            return r
        return r, [v/r, -v/r]


class Angle(InternalCoord):
    def __init__(self, i, j, k, *args, **kwargs):
        if i > k:
            i, j, k = k, j, i
        self.i = i
        self.j = j
        self.k = k
        self.idx = (i, j, k)
        super(Angle, self).__init__(*args, **kwargs)

    def hessian(self, rho):
        return 0.15*(rho[self.i, self.j]*rho[self.j, self.k])

    def weight(self, rho, coords):
        f = 0.12
        return np.sqrt(rho[self.i, self.j]*rho[self.j, self.k]) *\
            (f+(1-f)*np.sin(self.eval(coords)))

    def eval(self, coords, grad=False):
        v1 = (coords[self.i]-coords[self.j])/bohr
        v2 = (coords[self.k]-coords[self.j])/bohr
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
                (pi-phi)/(2*norm(v2)**2)*v2]
        else:
            grad = [
                1/np.tan(phi)*v1/norm(v1)**2-v2/(norm(v1)*norm(v2)*np.sin(phi)),
                (v1+v2)/(norm(v1)*norm(v2)*np.sin(phi)) -
                1/np.tan(phi)*(v1/norm(v1)**2+v2/norm(v2)**2),
                1/np.tan(phi)*v2/norm(v2)**2-v1/(norm(v1)*norm(v2)*np.sin(phi))]
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
        super(Dihedral, self).__init__(**kwargs)

    def hessian(self, rho):
        return 0.005*rho[self.i, self.j]*rho[self.j, self.k] *\
            rho[self.k, self.l]

    def weight(self, rho, coords):
        f = 0.12
        th1 = Angle(self.i, self.j, self.k).eval(coords)
        th2 = Angle(self.j, self.k, self.l).eval(coords)
        return (rho[self.i, self.j]*rho[self.j, self.k]*rho[self.k, self.l])**(1./3) *\
            (f+(1-f)*np.sin(th1))*(f+(1-f)*np.sin(th2))

    def eval(self, coords, grad=False):
        v1 = (coords[self.i]-coords[self.j])/bohr
        v2 = (coords[self.k]-coords[self.l])/bohr
        w = (coords[self.k]-coords[self.j])/bohr
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
                g/(norm(g)*norm(a2))]
        elif abs(phi) < 1e-6:
            g = Math.cross(w, a1)
            g = g/norm(g)
            A = dot(v1, ew)/norm(w)
            B = dot(v2, ew)/norm(w)
            grad = [
                g/(norm(g)*norm(a1)),
                -((1-A)/norm(a1)+B/norm(a2))*g,
                ((1+B)/norm(a2)-A/norm(a1))*g,
                -g/(norm(g)*norm(a2))]
        else:
            A = dot(v1, ew)/norm(w)
            B = dot(v2, ew)/norm(w)
            grad = [
                1/np.tan(phi)*a1/norm(a1)**2-a2/(norm(a1)*norm(a2)*np.sin(phi)),
                ((1-A)*a2-B*a1)/(norm(a1)*norm(a2)*np.sin(phi)) -
                1/np.tan(phi)*((1-A)*a1/norm(a1)**2-B*a2/norm(a2)**2),
                ((1+B)*a1+A*a2)/(norm(a1)*norm(a2)*np.sin(phi)) -
                1/np.tan(phi)*((1+B)*a2/norm(a2)**2+A*a1/norm(a1)**2),
                1/np.tan(phi)*a2/norm(a2)**2-a1/(norm(a1)*norm(a2)*np.sin(phi))]
        return phi, grad


class InternalCoords(list):
    def __init__(self, geom, allowed=None):
        super(InternalCoords, self).__init__()
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
        for bond in self.bonds:
            self.extend(get_dihedrals([bond.i, bond.j], geom.coords, bondmatrix, C))

    def __getattr__(self, attr):
        if attr == 'bonds':
            return [c for c in self if isinstance(c, Bond)]
        elif attr == 'angles':
            return [c for c in self if isinstance(c, Angle)]
        elif attr == 'dihedrals':
            return [c for c in self if isinstance(c, Dihedral)]
        elif attr == 'dict':
            return OrderedDict([('bonds', self.bonds),
                                ('angles', self.angles),
                                ('dihedrals', self.dihedrals)])
        else:
            raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(self.__class__.__name__, attr))

    def __repr__(self):
        return "<InternalCoords '{}'>".format(', '.join(
            '{}: {}'.format(name, len(coords)) for name, coords in self.dict.items()))

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

    def update_geom(self, geom, q, dq, B_inv):
        thre = 1e-6
        target = CartIter(q=q+dq)
        prev = CartIter(geom.coords, q, dq)
        for i in range(20):
            cur = CartIter(prev.cart+B_inv.dot(prev.dq).reshape(-1, 3)*bohr)
            cur.dcart = cur.cart-prev.cart
            geom.coords = cur.cart
            cur.q = self.eval_geom(geom, template=prev.q)
            cur.dq = target.q-cur.q
            if Math.rms(cur.dcart) < thre:
                msg = 'Perfect transformation to cartesians in {} iterations'
                break
            if i == 0:
                iter_first = cur
            prev = cur
        else:
            msg = 'Transformation did not converge in {} iterations'
            cur = iter_first
        info(msg.format(i+1))
        info('* RMS(dcart): {:.3}, RMS(dq): {:.3}'
             .format(Math.rms(cur.dcart), Math.rms(cur.dq)))
        geom.coords = cur.cart
        return cur.q


class CartIter(object):
    def __init__(self, cart=None, q=None, dq=None, dcart=None):
        self.cart = cart
        self.q = q
        self.dcart = dcart
        self.dq = dq


def get_dihedrals(center, coords, bondmatrix, C):
    neigh_l = [n for n in np.flatnonzero(bondmatrix[center[0], :]) if n != center[1]]
    neigh_r = [n for n in np.flatnonzero(bondmatrix[center[-1], :]) if n != center[-2]]
    angles_l = [Angle(i, center[0], center[1]).eval(coords) for i in neigh_l]
    angles_r = [Angle(center[-2], center[-1], j).eval(coords) for j in neigh_r]
    nonlinear_l = [n for n, ang in zip(neigh_l, angles_l) if ang < pi-1e-3]
    nonlinear_r = [n for n, ang in zip(neigh_r, angles_r) if ang < pi-1e-3]
    linear_l = [n for n, ang in zip(neigh_l, angles_l) if ang > pi-1e-3]
    linear_r = [n for n, ang in zip(neigh_r, angles_r) if ang > pi-1e-3]
    assert len(linear_l) <= 1
    assert len(linear_r) <= 1
    if center[0] < center[-1]:
        nweak = len(list(None for i in range(len(center)-1) if not C[center[i], center[i+1]]))
        dihedrals = [Dihedral(nl, center[0], center[-1], nr,
                              weak=nweak +
                              (0 if C[nl, center[0]] else 1) +
                              (0 if C[center[0], nr] else 1),
                              angles=(Angle(nl, center[0], center[1], C=C),
                                      Angle(nl, center[-2], center[-1], C=C)))
                     for nl, nr in product(nonlinear_l, nonlinear_r)
                     if nl != nr]
    else:
        dihedrals = []
    if len(center) > 3:
        pass
    elif linear_l and not linear_r:
        dihedrals.extend(get_dihedrals(linear_l + center, coords, bondmatrix, C))
    elif linear_r and not linear_l:
        dihedrals.extend(get_dihedrals(center + linear_r, coords, bondmatrix, C))
    return dihedrals


def get_property(idx, name):
    if isinstance(idx, int):
        select, value = 'number', str(idx)
    else:
        select, value = 'symbol', idx
    row = next(row for row in species_data if row[select] == value)
    value = row[name]
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return value

species_data = [row for row in csv.DictReader(StringIO("""\
number,symbol,name,vdw_radius,covalent_radius,mass,ionization_energy
1,H,hydrogen,1.2,0.38,1.0079,13.5984
2,He,helium,1.4,0.32,4.0026,24.5874
3,Li,lithium,1.82,1.34,6.941,5.3917
4,Be,beryllium,1.53,0.9,9.0122,9.3227
5,B,boron,1.92,0.82,10.811,8.298
6,C,carbon,1.7,0.77,12.0107,11.2603
7,N,nitrogen,1.55,0.75,14.0067,14.5341
8,O,oxygen,1.52,0.73,15.9994,13.6181
9,F,fluorine,1.47,0.71,18.9984,17.4228
10,Ne,neon,1.54,0.69,20.1797,21.5645
11,Na,sodium,2.27,1.54,22.9897,5.1391
12,Mg,magnesium,1.73,1.3,24.305,7.6462
13,Al,aluminium,1.84,1.18,26.9815,5.9858
14,Si,silicon,2.1,1.11,28.0855,8.1517
15,P,phosphorus,1.8,1.06,30.9738,10.4867
16,S,sulfur,1.8,1.02,32.065,10.36
17,Cl,chlorine,1.75,0.99,35.453,12.9676
18,Ar,argon,1.88,0.97,39.948,15.7596
19,K,potassium,2.75,1.96,39.0983,4.3407
20,Ca,calcium,2.31,1.74,40.078,6.1132
21,Sc,scandium,2.11,1.44,44.9559,6.5615
22,Ti,titanium,,1.36,47.867,6.8281
23,V,vanadium,,1.25,50.9415,6.7462
24,Cr,chromium,,1.27,51.9961,6.7665
25,Mn,manganese,,1.39,54.938,7.434
26,Fe,iron,,1.25,55.845,7.9024
27,Co,cobalt,,1.26,58.9332,7.881
28,Ni,nickel,1.63,1.21,58.6934,7.6398
29,Cu,copper,1.4,1.38,63.546,7.7264
30,Zn,zinc,1.39,1.31,65.39,9.3942
31,Ga,gallium,1.87,1.26,69.723,5.9993
32,Ge,germanium,2.11,1.22,72.64,7.8994
33,As,arsenic,1.85,1.19,74.9216,9.7886
34,Se,selenium,1.9,1.16,78.96,9.7524
35,Br,bromine,1.85,1.14,79.904,11.8138
36,Kr,krypton,2.02,1.1,83.8,13.9996
37,Rb,rubidium,3.03,2.11,85.4678,4.1771
38,Sr,strontium,2.49,1.92,87.62,5.6949
39,Y,yttrium,,1.62,88.9059,6.2173
40,Zr,zirconium,,1.48,91.224,6.6339
41,Nb,niobium,,1.37,92.9064,6.7589
42,Mo,molybdenum,,1.45,95.94,7.0924
43,Tc,technetium,,1.56,98,7.28
44,Ru,ruthenium,,1.26,101.07,7.3605
45,Rh,rhodium,,1.35,102.9055,7.4589
46,Pd,palladium,1.63,1.31,106.42,8.3369
47,Ag,silver,1.72,1.53,107.8682,7.5762
48,Cd,cadmium,1.58,1.48,112.411,8.9938
49,In,indium,1.93,1.44,114.818,5.7864
50,Sn,tin,2.17,1.41,118.71,7.3439
51,Sb,antimony,2.06,1.38,121.76,8.6084
52,Te,tellurium,2.06,1.35,127.6,9.0096
53,I,iodine,1.98,1.33,126.9045,10.4513
54,Xe,xenon,2.16,1.3,131.293,12.1298
55,Cs,caesium,3.43,2.25,132.9055,3.8939
56,Ba,barium,2.68,1.98,137.327,5.2117
57,La,lanthanum,,1.69,138.9055,5.5769
58,Ce,cerium,,,140.116,5.5387
59,Pr,praseodymium,,,140.9077,5.473
60,Nd,neodymium,,,144.24,5.525
61,Pm,promethium,,,145,5.582
62,Sm,samarium,,,150.36,5.6437
63,Eu,europium,,,151.964,5.6704
64,Gd,gadolinium,,,157.25,6.1501
65,Tb,terbium,,,158.9253,5.8638
66,Dy,dysprosium,,,162.5,5.9389
67,Ho,holmium,,,164.9303,6.0215
68,Er,erbium,,,167.259,6.1077
69,Tm,thulium,,,168.9342,6.1843
70,Yb,ytterbium,,,173.04,6.2542
71,Lu,lutetium,,1.6,174.967,5.4259
72,Hf,hafnium,,1.5,178.49,6.8251
73,Ta,tantalum,,1.38,180.9479,7.5496
74,W,tungsten,,1.46,183.84,7.864
75,Re,rhenium,,1.59,186.207,7.8335
76,Os,osmium,,1.28,190.23,8.4382
77,Ir,iridium,,1.37,192.217,8.967
78,Pt,platinum,1.75,1.28,195.078,8.9587
79,Au,gold,1.66,1.44,196.9665,9.2255
80,Hg,mercury,1.55,1.49,200.59,10.4375
81,Tl,thallium,1.96,1.48,204.3833,6.1082
82,Pb,lead,2.02,1.47,207.2,7.4167
83,Bi,bismuth,2.07,1.46,208.9804,7.2856
84,Po,polonium,1.97,,209,8.417
85,At,astatine,2.02,,210,9.3
86,Rn,radon,2.2,1.45,222,10.7485
87,Fr,francium,3.48,,223,4.0727
88,Ra,radium,2.83,,226,5.2784
89,Ac,actinium,,,227,5.17
90,Th,thorium,,,232.0381,6.3067
91,Pa,protactinium,,,231.0359,5.89
92,U,uranium,1.86,,238.0289,6.1941
"""))]
