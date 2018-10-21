# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
__version__ = '0.2.0'


def optimize(optimizer, solver):
    """
    Optimize a geometry with respect to a solver.

    :param generator optimizer: Optimizer object with the same generator interface
        as :py:func:`berny.Berny`
    :param generator solver: unprimed generator that receives geometry as a
        2-tuple of a list of 2-tuples of the atom symbol and coordinate (as a
        3-tuple), and of a list of lattice vectors (or None if molecule), and
        yields the energy and gradients (as a N-by-3 matrix or (N+3)-by-3
        matrix in case of a crystal geometry)

    Returns the optimized geometry.

    The function is equivalent to::

        next(solver)
        for geom in optimizer:
            energy, gradients = solver.send((list(geom), geom.lattice))
            optimizer.send((energy, gradients))
    """
    next(solver)
    for geom in optimizer:
        energy, gradients = solver.send((list(geom), geom.lattice))
        optimizer.send((energy, gradients))
    return geom
