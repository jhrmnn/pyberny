# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
__version__ = '0.2.0'


def optimize(optimizer, solver, trajectory=None):
    """Optimize a geometry with respect to a solver.

    Args:
        optimizer (:class:`~collections.abc.Generator`): Optimizer object with
            the same generator interface as :class:`~berny.Berny`
        solver (:class:`~collections.abc.Generator`): unprimed generator that
            receives geometry as a 2-tuple of a list of 2-tuples of the atom
            symbol and coordinate (as a 3-tuple), and of a list of lattice
            vectors (or :data:`None` if molecule), and yields the energy and
            gradients (as a :math:`N`-by-3 matrix or :math:`(N+3)`-by-3 matrix
            in case of a crystal geometry).

            See :class:`~berny.solvers.MopacSolver` for an example.
        trajectory (str): filename for the XYZ trajectory

    Returns:
        The optimized geometry.

    The function is equivalent to::

        next(solver)
        for geom in optimizer:
            energy, gradients = solver.send((list(geom), geom.lattice))
            optimizer.send((energy, gradients))
    """
    if trajectory:
        trajectory = open(trajectory, 'w')
    try:
        next(solver)
        for geom in optimizer:
            energy, gradients = solver.send((list(geom), geom.lattice))
            if trajectory:
                geom.dump(trajectory, 'xyz')
            optimizer.send((energy, gradients))
    finally:
        if trajectory:
            trajectory.close()
    return geom
