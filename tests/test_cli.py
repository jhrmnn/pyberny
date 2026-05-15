from io import StringIO

from berny import Berny
from berny.cli import handler
from berny.geomlib import Geometry


def test_handler_converges_returns_none():
    # Until 2026, handler called berny.send(energy, gradients) — two
    # positional args to a method that takes a single tuple — so the CLI
    # crashed on every step. It also let StopIteration propagate on
    # convergence instead of returning None to the driver.
    geom = Geometry(
        ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]
    )
    berny = Berny(geom)
    next(berny)
    f = StringIO('0.0\n0 0 0\n0 0 0\n0 0 0\n')
    result = handler(berny, f)
    assert result is None
    assert berny.converged
