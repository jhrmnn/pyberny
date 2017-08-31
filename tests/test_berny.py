from pytest import approx
import numpy as np
from berny import optimize, geomlib

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


def MopacSolver(cmd='mopac', method='PM7'):
    import os
    import tempfile
    import subprocess
    import shutil

    tmpdir = tempfile.mkdtemp()
    kcal, ev, angstrom = 627.503, 27.2107, 0.52917721092
    try:
        atoms = yield
        while True:
            mopac_input = '{} 1SCF GRADIENTS\n\n\n'.format(method) + '\n'.join(
                '{} {} 1 {} 1 {} 1'.format(el, *coord) for el, coord in atoms
            )
            input_file = os.path.join(tmpdir, 'job.mop')
            with open(input_file, 'w') as f:
                f.write(mopac_input)
            subprocess.check_output([cmd, input_file])
            with open(os.path.join(tmpdir, 'job.out')) as f:
                energy = float(next(l for l in f if 'TOTAL ENERGY' in l).split()[3])/ev
                next(l for l in f if 'FINAL  POINT  AND  DERIVATIVES' in l)
                next(f)
                next(f)
                gradients = [
                    [float(next(f).split()[6])/kcal*angstrom for _ in range(3)]
                    for _ in range(len(atoms))
                ]

            atoms = yield energy, gradients
    finally:
        shutil.rmtree(tmpdir)


def test_basic():
    solver = MopacSolver()
    final = optimize(solver, ethanol, steprms=0.01, stepmax=0.05)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([14.94631992, 52.57923507, 61.10485588], rel=1e-8)
