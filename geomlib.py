import numpy as np
from collections import defaultdict
from itertools import chain, product, repeat, combinations
import os
import csv
import io

bohr = 0.52917721092


class Molecule:
    def __init__(self, species, coords):
        self.species = species
        self.coords = np.array(coords)

    def __repr__(self):
        counter = defaultdict(int)
        for specie in self.species:
            counter[specie] += 1
        return ''.join('{}{}'.format(sp, n if n > 1 else '')
                       for sp, n in sorted(counter.items()))

    def __iter__(self):
        for specie, coord in zip(self.species, self.coords):
            yield specie, coord

    def copy(self):
        return Molecule(self.species.copy(), self.coords.copy())

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
                          for i, j in combinations(range(len(self.species)), 2)
                          if bond[i, j]]}

    def dist(self, geom, ret_diff=False):
        diff = self.coords[:, None, :]-geom.coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, 2))
        dist[np.diag_indices(len(self.coords))] = np.inf
        if ret_diff:
            return dist, diff
        else:
            return dist

    def bondmatrix(self, scale=1.3):
        dist = self.dist(self)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in self.species])
        return dist < 1.3*(radii[None, :]+radii[:, None])

    def morse(self, cutoff=20.):
        def V(r, r0):
            return (1-np.exp(-(r-r0)))**2-1

        def dVdr(r, r0):
            return 2*(1-np.exp(-(r-r0)))*np.exp(-(r-r0))

        supercell = self.supercell(cutoff=cutoff)
        dist, diff = self.dist(supercell, ret_diff=True)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in supercell.species])
        r0 = radii[:len(self.species), None]+radii[None, :]
        E = np.sum(V(dist, r0))/2
        dEdR = np.sum(dVdr(dist, r0)[:, :, None]*diff/dist[:, :, None], 1)
        return E, dEdR

    def internal_coords(self, allowed=None):
        supercell = self.supercell([(-1, 1), (-1, 1), (-1, 1)])
        dist = supercell.dist(supercell)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in supercell.species])
        bondmatrix = dist < 1.3*(radii[None, :]+radii[:, None])
        fragments, C = get_clusters(bondmatrix)
        radii = np.array([get_property(sp, 'vdw_radius') for sp in supercell.species])
        shift = 0.
        C_total = C.copy()
        while not C_total.all():
            bondmatrix |= ~C_total & (dist < radii[None, :]+radii[:, None]+shift)
            C_total = get_clusters(bondmatrix)[1]
            shift += 1.
        int_coords = {
            'fragments': fragments,
            'bonds': [
                Bond(i, j, weak=0 if C[i, j] else 1)
                for i, j in combinations(range(len(self.species)), 2)
                if bondmatrix[i, j]],
            'angles': [
                Angle(j, i, k, weak=(0 if C[j, i] else 1)+(0 if C[i, k] else 1))
                for i in range(len(self.species))
                for j, k in combinations(np.flatnonzero(bondmatrix[i, :]), 2)]}
        int_coords['dihedrals'] = list(chain.from_iterable(
            self._get_dihedrals([bond.i, bond.j], bondmatrix, C)
            for bond in int_coords['bonds']))
        return InternalCoords(**int_coords)

    def _get_dihedrals(self, center, bondmatrix, C):
        neigh_l = [n for n in np.flatnonzero(bondmatrix[center[0], :]) if n != center[1]]
        neigh_r = [n for n in np.flatnonzero(bondmatrix[center[-1], :]) if n != center[-2]]
        angles_l = [Angle(i, center[0], center[1]).eval(self.coords) for i in neigh_l]
        angles_r = [Angle(center[-2], center[-1], j).eval(self.coords) for j in neigh_r]
        nonlinear_l = [n for n, ang in zip(neigh_l, angles_l) if ang < np.pi-1e-3]
        nonlinear_r = [n for n, ang in zip(neigh_r, angles_r) if ang < np.pi-1e-3]
        linear_l = [n for n, ang in zip(neigh_l, angles_l) if ang > np.pi-1e-3]
        linear_r = [n for n, ang in zip(neigh_r, angles_r) if ang > np.pi-1e-3]
        assert len(linear_l) <= 1
        assert len(linear_r) <= 1
        if center[0] < center[-1]:
            nweak = len(list(None for i in range(len(center)-1) if C[center[i], center[i+1]]))
            dihedrals = [Dihedral(nl, center[0], center[-1], nr,
                                  weak=nweak +
                                  (0 if C[nl, center[0]] else 1) +
                                  (0 if C[center[0], nr] else 1))
                         for nl, nr in product(nonlinear_l, nonlinear_r)
                         if nl != nr]
        if len(center) > 3:
            pass
        elif linear_l and not linear_r:
            dihedrals.extend(self.get_dihedrals(linear_l + center, bondmatrix, C))
        elif linear_r and not linear_l:
            dihedrals.extend(self.get_dihedrals(center + linear_r, bondmatrix, C))
        return dihedrals

    def eval_internal(self, int_coords):
        supercell = self.supercell([(-1, 1), (-1, 1), (-1, 1)])
        dist = supercell.dist(supercell)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in supercell.species])
        rho = np.exp(-dist/(radii[None, :]+radii[:, None])+1)
        return int_coords.evaluate(supercell, rho)


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


def readfile(path, fmt=None):
    if not fmt:
        ext = os.path.splitext(path)[1]
        if ext == '.xyz':
            fmt = 'xyz'
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
        super().__init__(species, coords)

    def copy(self):
        return Crystal(self.species.copy(), self.coords.copy(), self.lattice.copy())

    def super_circum(self, radius):
        rec_lattice = 2*np.pi*np.linalg.inv(self.lattice.T)
        layer_sep = np.array(
            [sum(vec*rvec/np.linalg.norm(rvec))
             for vec, rvec in zip(self.lattice, rec_lattice)])
        return np.array([int(n) for n in np.ceil(radius/layer_sep+1/2)])

    def supercell(self, ranges=None, cutoff=None):
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


class InternalCoord:
    def __init__(self, weak=0, weight=None):
        self.weak = weak
        self.weight = weight


class Bond(InternalCoord):
    def __init__(self, i, j, *args, **kwargs):
        self.i = i
        self.j = j
        super().__init__(*args, **kwargs)


class Angle(InternalCoord):
    def __init__(self, i, j, k, *args, **kwargs):
        self.i = i
        self.j = j
        self.k = k
        super().__init__(*args, **kwargs)

    def eval(self, geom, grad=False):
        do_grad = grad
        v1 = geom.coords[self.i]-geom.cords[self.j]
        v2 = geom.coords[self.k]-geom.cords[self.j]
        sc = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        if sc < -1:
            sc = -1
        if sc > 1:
            sc = 1
        phi = np.arccos(sc)
        if not do_grad:
            return phi


class Dihedral(InternalCoord):
    def __init__(self, i, j, k, l, *args, **kwargs):
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        super().__init__(*args, **kwargs)


class InternalCoords:
    def __init__(self, bonds, angles, dihedrals, fragments):
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.fragments = fragments

    def __iter__(self):
        for name in ['bonds', 'angles', 'dihedrals']:
            yield name, getattr(self, name)

    def __repr__(self):
        return "<InternalCoords '{}'>".format(', '.join(
            '{}: {}'.format(name, len(coords)) for name, coords in self))

    def __str__(self):
        ncoords = sum(len(coords) for _, coords in self)
        s = ''
        s += 'Internal coordinates:\n'
        s += '* Number of fragments: {}\n'.format(len(self.fragments))
        s += '* Number of internal coordinates: {}\n'.format(ncoords)
        for name, coords in self:
            for degree, adjective in [(0, 'strong'), (1, 'weak'), (2, 'superweak')]:
                n = len([None for c in coords if min(2, c.weak) == degree])
                if n > 0:
                    s += '* Number of {} {}: {}\n'.format(adjective, name, n)
        return s.rstrip()

    def to_dict(self):
        return {'bonds': self.bonds,
                'angles': self.angles,
                'dihedrals': self.dihedrals,
                'fragments': self.fragments}

    def evaluate(self, geom, rho):
        q = []
        for bond in self.bonds:
            q.append(bond(*geom.coords[(bond.i, bond.j), :]))


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

species_data = [row for row in csv.DictReader(io.StringIO("""\
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
