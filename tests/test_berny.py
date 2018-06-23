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


def test_ethanol():
    solver = MopacSolver()
    final = optimize(solver, ethanol, steprms=0.01, stepmax=0.05)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([14.95, 52.58, 61.10], rel=1e-3)


aniline = geomlib.loads("""\
14
Aniline
  H      1.5205     -0.1372      2.5286
  C      0.9575     -0.0905      1.5914
  C     -0.4298     -0.1902      1.6060
  H     -0.9578     -0.3156      2.5570
  C     -1.1520     -0.1316      0.4215
  H     -2.2452     -0.2104      0.4492
  C     -0.4779      0.0324     -0.7969
  N     -1.2191      0.2008     -2.0081
  H     -2.0974     -0.2669     -1.9681
  H     -0.6944     -0.0913     -2.8025
  C      0.9208      0.1292     -0.8109
  H      1.4628      0.2560     -1.7555
  C      1.6275      0.0685      0.3828
  H      2.7196      0.1470      0.3709
""", 'xyz')


def test_aniline():
    solver = MopacSolver()
    final = optimize(solver, aniline, steprms=0.01, stepmax=0.05)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([90.94, 193.1, 283.9], rel=1e-3)
