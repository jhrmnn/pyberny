from pytest import approx
import numpy as np
from berny import optimize, geomlib
from berny.solvers import MopacSolver

ethanol = geomlib.loads("""\
9

C	1.1879	-0.3829	0.0000
C	0.0000	0.5526	0.0000
O	-1.1867	-0.2472	0.0000
H	-1.9237	0.3850	0.0000
H	2.0985	0.2306	0.0000
H	1.1184	-1.0093	0.8869
H	1.1184	-1.0093	-0.8869
H	-0.0227	1.1812	0.8852
H	-0.0227	1.1812	-0.8852
""", 'xyz')


def test_basic():
    solver = MopacSolver()
    final = optimize(solver, ethanol, steprms=0.01, stepmax=0.05)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([14.94631992, 52.57923507, 61.10485588], rel=1e-3)
