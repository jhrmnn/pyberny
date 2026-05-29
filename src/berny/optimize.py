# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack

from .berny import Berny
from .geomlib import Geometry
from .solvers import SolverInput, SolverOutput, TS_SolverOutput


def optimize(
    optimizer: Berny,
    solver: Generator[SolverOutput | TS_SolverOutput, SolverInput, None],
    trajectory: str | None = None,
) -> Geometry:
    """Optimize a geometry with respect to a solver.

    Args:
        optimizer: optimizer with the generator interface of :class:`~berny.Berny`
        solver: unprimed generator that receives geometry as a 2-tuple of a
            list of 2-tuples of the atom symbol and coordinate (as a 3-tuple),
            and of a list of lattice vectors (or :data:`None` if molecule), and
            yields the energy and gradients (as a :math:`N`-by-3 matrix or
            :math:`(N+3)`-by-3 matrix in case of a crystal geometry), and
            optionally a Cartesian Hessian as a :math:`(3N)`-by-:math:`(3N)`
            matrix (for TS-capable solvers).

            See :class:`~berny.solvers.MopacSolver` for an example.
        trajectory: filename for the XYZ trajectory

    Returns:
        The optimized geometry.

    The function is equivalent to::

        next(solver)
        for geom in optimizer:
            result = solver.send((list(geom), geom.lattice))
            optimizer.send(result)
    """
    with ExitStack() as stack:
        traj_fp = stack.enter_context(open(trajectory, 'w')) if trajectory else None
        next(solver)
        for geom in optimizer:
            result = solver.send((list(geom), geom.lattice))
            if traj_fp is not None:
                geom.dump(traj_fp, 'xyz')
            optimizer.send(result)
    result_geom: Geometry = geom
    return result_geom
