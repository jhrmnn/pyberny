# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from io import StringIO
from itertools import chain, groupby, product, repeat
from typing import Any

import numpy as np
from numpy import pi
from numpy.linalg import inv, norm

from .species_data import get_property

__all__ = ['Geometry', 'loads', 'readfile']


class Geometry:
    """Represents a single molecule or a crystal.

    Iterating over a geometry yields 2-tuples of symbols and coordinates.
    :func:`len` returns the number of atoms in a geometry. The class supports
    :func:`format` with the same available formats as :meth:`dump`.

    Args:
        species: list of element symbols.
        coords: atomic coordinates in angstroms, shape ``(N, 3)``.
        lattice: lattice vectors, shape ``(3, 3)``, or :data:`None` for a
            molecule.
    """

    def __init__(
        self,
        species: list[str],
        coords: Any,
        lattice: Any = None,
    ) -> None:
        self.species = species
        self.coords = np.array(coords, dtype=float)
        self.lattice = np.array(lattice, dtype=float) if lattice is not None else None

    @classmethod
    def from_atoms(
        cls,
        atoms: Iterable[tuple[str, Any]],
        lattice: Any = None,
        unit: float = 1.0,
    ) -> Geometry:
        """Alternative constructor.

        Args:
            atoms: iterable of ``(element_symbol, coordinate)`` 2-tuples.
            lattice: lattice vectors, or :data:`None` for a molecule.
            unit: value to multiply atomic coordinates with.
        """
        atoms = list(atoms)
        species = [sp for sp, _ in atoms]
        coords = [np.array(coord, dtype=float) * unit for _, coord in atoms]
        return cls(species, coords, lattice)

    def __repr__(self) -> str:
        s = repr(self.formula)
        if self.lattice is not None:
            s += ' in a lattice'
        return f'<{self.__class__.__name__} {s}>'

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from zip(self.species, self.coords)

    def __len__(self) -> int:
        return len(self.species)

    @property
    def formula(self) -> str:
        """Chemical formula of the molecule or a unit cell."""
        composition = sorted(
            (sp, len(list(g))) for sp, g in groupby(sorted(self.species))
        )
        return ''.join(f"{sp}{n if n > 1 else ''}" for sp, n in composition)

    def __format__(self, fmt: str) -> str:
        """Return the geometry represented as a string, delegates to :meth:`dump`."""
        fp = StringIO()
        self.dump(fp, fmt)
        return fp.getvalue()

    dumps = __format__

    def dump(self, f: Any, fmt: str) -> None:
        """Save the geometry into a file.

        :param file f: file object
        :param str fmt: geometry format, one of ``""``, ``"xyz"``, ``"aims"``,
            ``"mopac"``.
        """
        if fmt == '':
            f.write(repr(self))
        elif fmt == 'xyz':
            f.write(f'{len(self)}\n')
            f.write(f'Formula: {self.formula}\n')
            for specie, coord in self:
                coords_str = ' '.join(f'{x:15.8}' for x in coord)
                f.write(f'{specie:>2} {coords_str}\n')
        elif fmt == 'aims':
            f.write(f'# Formula: {self.formula}\n')
            if self.lattice is not None:
                for vec in self.lattice:
                    vec_str = ' '.join(f'{x:15.8}' for x in vec)
                    f.write(f'lattice_vector {vec_str}\n')
            for specie, coord in self:
                coords_str = ' '.join(f'{x:15.8}' for x in coord)
                f.write(f'atom {coords_str} {specie:>2}\n')
        elif fmt == 'mopac':
            f.write(f'* Formula: {self.formula}\n')
            for specie, coord in self:
                coords_str = ' '.join(f'{x:15.8} 1' for x in coord)
                f.write(f'{specie:>2} {coords_str}\n')
        else:
            raise ValueError(f'Unknown format: {fmt!r}')

    def copy(self) -> Geometry:
        """Make a copy of the geometry."""
        return Geometry(
            list(self.species),
            self.coords.copy(),
            self.lattice.copy() if self.lattice is not None else None,
        )

    def write(self, filename: str) -> None:
        """Write the geometry into a file, delegates to :meth:`dump`.

        Args:
            filename: path that will be overwritten.
        """
        ext = os.path.splitext(filename)[1]
        if ext == '.xyz':
            fmt = 'xyz'
        elif ext == '.aims' or os.path.basename(filename) == 'geometry.in':
            fmt = 'aims'
        elif ext == '.mopac':
            fmt = 'mopac'
        else:
            raise ValueError('Unknown file extension')
        with open(filename, 'w') as f:
            self.dump(f, fmt)

    def super_circum(self, radius):
        """
        Supercell dimensions such that the supercell circumsribes a sphere.

        :param float radius: circumscribed radius in angstroms

        Returns :data:`None` when geometry is not a crystal.
        """
        if self.lattice is None:
            return None
        rec_lattice = 2 * pi * inv(self.lattice.T)
        layer_sep = np.array(
            [
                sum(vec * rvec / norm(rvec))
                for vec, rvec in zip(self.lattice, rec_lattice)
            ]
        )
        return np.array(np.ceil(radius / layer_sep + 0.5), dtype=int)

    def supercell(self, ranges=((-1, 1), (-1, 1), (-1, 1)), cutoff=None):
        """
        Create a crystal supercell.

        :param list ranges: list of 2-tuples specifying the range of multiples
            of the unit-cell vectors
        :param float cutoff: if given, the ranges are determined such that
            the supercell contains a sphere with the radius qual to the cutoff

        Returns a copy of itself when geometry is not a crystal.
        """
        if self.lattice is None:
            return self.copy()
        if cutoff:
            ranges = [(-r, r) for r in self.super_circum(cutoff)]
        latt_vectors = np.array(
            [(0, 0, 0)]
            + [
                sum(k * vec for k, vec in zip(shift, self.lattice))
                for shift in product(*[range(a, b + 1) for a, b in ranges])
                if shift != (0, 0, 0)
            ]
        )
        species = list(chain.from_iterable(repeat(self.species, len(latt_vectors))))
        coords = (self.coords[None, :, :] + latt_vectors[:, None, :]).reshape((-1, 3))
        lattice = self.lattice * np.array([b - a for a, b in ranges])[:, None]
        return Geometry(species, coords, lattice)

    def dist_diff(self, other=None):
        r"""
        Calculate distances and vectors between atoms.

        Args:
            other (:class:`~berny.Geometry`): calculate distances between two
                geometries if given or within a geometry if not

        Returns:
            :math:`R_{ij}:=|\mathbf R_i-\mathbf R_j|` and
            :math:`R_{ij\alpha}:=(\mathbf R_i)_\alpha-(\mathbf R_j)_\alpha`.
        """
        if other is None:
            other = self
        diff = self.coords[:, None, :] - other.coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, 2))
        dist[np.diag_indices(len(self))] = np.inf
        return dist, diff

    def dist(self, other=None):
        """Alias for the first element of :meth:`dist_diff`."""
        return self.dist_diff(other)[0]

    def bondmatrix(self, scale=1.3):
        r"""
        Calculate the covalent connectedness matrix.

        :param float scale: threshold for accepting a distance as a covalent bond

        Returns:
            :math:`b_{ij}:=R_{ij}<\text{scale}\times (R_i^\text{cov}+R_j^\text{cov})`.
        """
        dist = self.dist(self)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in self.species])
        return dist < scale * (radii[None, :] + radii[:, None])

    def rho(self):
        r"""
        Calculate a measure of covalentness.

        Returns:
            :math:`\rho_{ij}:=\exp\big(-R_{ij}/(R_i^\text{cov}+R_j^\text{cov})\big)`.
        """
        geom = self.supercell()
        dist = geom.dist(geom)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in geom.species])
        return np.exp(-dist / (radii[None, :] + radii[:, None]) + 1)

    @property
    def masses(self):
        """Numpy array of atomic masses."""
        return np.array([get_property(sp, 'mass') for sp in self.species])

    @property
    def cms(self):
        r"""Calculate the center of mass, :math:`\mathbf R_\text{CMS}`."""
        masses = self.masses
        return np.sum(masses[:, None] * self.coords, 0) / masses.sum()

    @property
    def inertia(self):
        r"""Calculate the moment of inertia.

        .. math::
            I_{\alpha\beta}:=
            \sum_im_i\big(r_i^2\delta_{\alpha\beta}-(\mathbf r_i)_\alpha(\mathbf
            r_i)_\beta\big),\qquad
            \mathbf r_i=\mathbf R_i-\mathbf R_\text{CMS}
        """
        coords_w = np.sqrt(self.masses)[:, None] * (self.coords - self.cms)
        A = np.array([np.diag(np.full(3, r)) for r in np.sum(coords_w**2, 1)])
        B = coords_w[:, :, None] * coords_w[:, None, :]
        return np.sum(A - B, 0)


def load(fp: Any, fmt: str) -> Geometry:
    """Read a geometry from a file object.

    Args:
        fp: file object.
        fmt: geometry format, one of ``"xyz"`` or ``"aims"``.
    """
    if fmt == 'xyz':
        n = int(fp.readline())
        fp.readline()
        species = []
        coords = []
        for _ in range(n):
            l = fp.readline().split()
            species.append(l[0])
            coords.append([float(x) for x in l[1:4]])
        return Geometry(species, coords)
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
            return Geometry(species, coords, lattice)
        return Geometry(species, coords)
    raise ValueError(f'Unknown format: {fmt!r}')


def loads(s: str, fmt: str) -> Geometry:
    """Read a geometry from a string, delegates to :func:`load`.

    Args:
        s: string with geometry.
        fmt: geometry format (see :func:`load`).
    """
    fp = StringIO(s)
    return load(fp, fmt)


def readfile(path: str, fmt: str | None = None) -> Geometry:
    """Read a geometry from a file path, delegates to :func:`load`.

    Args:
        path: path to a geometry file.
        fmt: format; if not given, derived from the file extension.
    """
    if not fmt:
        ext = os.path.splitext(path)[1]
        if ext == '.xyz':
            fmt = 'xyz'
        elif ext == '.aims' or os.path.basename(path) == 'geometry.in':
            fmt = 'aims'
        else:
            raise ValueError(f'Cannot infer format from path {path!r}')
    with open(path) as f:
        return load(f, fmt)
