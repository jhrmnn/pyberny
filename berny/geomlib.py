# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
from itertools import chain, groupby, product, repeat

import numpy as np
from numpy import pi
from numpy.linalg import inv, norm

from .species_data import get_property

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


class Molecule(object):
    def __init__(self, species, coords):
        self.species = species
        self.coords = np.array(coords)

    @classmethod
    def from_atoms(cls, atoms, unit=1.):
        species = [sp for sp, _ in atoms]
        coords = [np.array(coord)*unit for _, coord in atoms]
        return cls(species, coords)

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.formula)

    @property
    def formula(self):
        composition = sorted(
            (sp, len(list(g)))
            for sp, g in groupby(sorted(self.species))
        )
        return ''.join(
            '{}{}'.format(sp, n if n > 1 else '') for sp, n in composition
        )

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

    def dump(self, f, fmt):
        if fmt == '':
            f.write(repr(self))
        elif fmt == 'xyz':
            f.write('{}\n'.format(len(self)))
            f.write('Formula: {}\n'.format(self.formula))
            for specie, coord in self:
                f.write('{:>2} {}\n'.format(
                    specie, ' '.join('{:15.8}'.format(x) for x in coord)
                ))
        elif fmt == 'aims':
            f.write('# Formula: {}\n'.format(self.formula))
            for specie, coord in self:
                f.write('atom {} {:>2}\n'.format(
                    ' '.join('{:15.8}'.format(x) for x in coord), specie
                ))
        elif fmt == 'mopac':
            f.write('* Formula: {}\n'.format(self.formula))
            for specie, coord in self:
                f.write('{:>2} {}\n'.format(
                    specie, ' '.join('{:15.8} 1'.format(x) for x in coord)
                ))
        else:
            raise ValueError("Unknown format: '{}'".format(fmt))

    def copy(self):
        return Molecule(list(self.species), self.coords.copy())

    def write(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == 'xyz':
            fmt = 'xyz'
        elif ext == 'aims' or os.path.basename(filename) == 'geometry.in':
            fmt = 'aims'
        elif ext == 'mopac':
            fmt = 'mopac'
        with open(filename, 'w') as f:
            self.dump(f, fmt)

    def supercell(self, *args, **kwargs):
        return self.copy()

    def dist_diff(self, geom):
        diff = self.coords[:, None, :]-geom.coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, 2))
        dist[np.diag_indices(len(self))] = np.inf
        return dist, diff

    def dist(self, geom):
        return self.dist_diff(geom)[0]

    def bondmatrix(self, scale=1.3):
        dist = self.dist(self)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in self.species])
        return dist < 1.3*(radii[None, :]+radii[:, None])

    def rho(self):
        geom = self.supercell()
        dist = geom.dist(geom)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in geom.species])
        return np.exp(-dist/(radii[None, :]+radii[:, None])+1)

    @property
    def masses(self):
        return np.array([get_property(sp, 'mass') for sp in self.species])

    @property
    def cms(self):
        masses = self.masses
        return np.sum(masses[:, None]*self.coords, 0)/masses.sum()

    @property
    def inertia(self):
        coords_w = np.sqrt(self.masses)[:, None]*(self.coords-self.cms)
        A = np.array([np.diag(np.full(3, r)) for r in np.sum(coords_w**2, 1)])
        B = coords_w[:, :, None]*coords_w[:, None, :]
        return np.sum(A-B, 0)


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


def loads(s, fmt):
    fp = StringIO(s)
    return load(fp, fmt)


def readfile(path, fmt=None):
    if not fmt:
        ext = os.path.splitext(path)[1]
        if ext == '.xyz':
            fmt = 'xyz'
        if ext == '.aims' or os.path.basename(path) == 'geometry.in':
            fmt = 'aims'
    with open(path) as f:
        return load(f, fmt)


class Crystal(Molecule):
    def __init__(self, species, coords, lattice):
        Molecule.__init__(self, species, coords)
        self.lattice = np.array(lattice)

    def copy(self):
        return Crystal(list(self.species), self.coords.copy(), self.lattice.copy())

    def super_circum(self, radius):
        rec_lattice = 2*pi*inv(self.lattice.T)
        layer_sep = np.array(
            [sum(vec*rvec/norm(rvec)) for vec, rvec in zip(self.lattice, rec_lattice)]
        )
        return np.array(np.ceil(radius/layer_sep+0.5), dtype=int)

    def supercell(self, ranges=((-1, 1), (-1, 1), (-1, 1)), cutoff=None):
        if cutoff:
            ranges = [(-r, r) for r in self.super_circum(cutoff)]
        latt_vectors = np.array([(0, 0, 0)] + [
            sum(k*vec for k, vec in zip(shift, self.lattice))
            for shift
            in product(*[range(a, b+1) for a, b in ranges])
            if shift != (0, 0, 0)
        ])
        species = list(chain.from_iterable(repeat(self.species, len(latt_vectors))))
        coords = (self.coords[None, :, :]+latt_vectors[:, None, :]).reshape((-1, 3))
        lattice = self.lattice*np.array([b-a for a, b in ranges])[:, None]
        return Crystal(species, coords, lattice)
