# PyBerny

![checks](https://img.shields.io/github/checks-status/jhrmnn/pyberny/master.svg)
[![coverage](https://img.shields.io/codecov/c/github/jhrmnn/pyberny.svg)](https://codecov.io/gh/jhrmnn/pyberny)
![python](https://img.shields.io/pypi/pyversions/pyberny.svg)
[![pypi](https://img.shields.io/pypi/v/pyberny.svg)](https://pypi.org/project/pyberny/)
[![commits since](https://img.shields.io/github/commits-since/jhrmnn/pyberny/latest.svg)](https://github.com/jhrmnn/pyberny/releases)
[![last commit](https://img.shields.io/github/last-commit/jhrmnn/pyberny.svg)](https://github.com/jhrmnn/pyberny/commits/master)
[![license](https://img.shields.io/github/license/jhrmnn/pyberny.svg)](https://github.com/jhrmnn/pyberny/blob/master/LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![doi](https://img.shields.io/badge/doi-10.5281%2Fzenodo.3695037-blue)](http://doi.org/10.5281/zenodo.3695037)

PyBerny is an optimizer of molecular geometries with respect to the total energy, using nuclear gradient information.

In each step, it takes energy and Cartesian gradients as an input, and returns a new equilibrium structure estimate.

The package implements a single optimization algorithm, which is an amalgam of several techniques, comprising the quasi-Newton method, redundant internal coordinates, an iterative Hessian approximation, a trust region scheme, linear search, and coordinate weighting. The algorithm is described in more detail in the [documentation](https://jhrmnn.github.io/pyberny/master/algorithm.html).

Several desirable features are missing or incomplete at the moment, some of them being actively worked on (help is always welcome): [crystal geometries](https://github.com/jhrmnn/pyberny/issues/5), [coordinate constraints](https://github.com/jhrmnn/pyberny/issues/14), [coordinate weighting](https://github.com/jhrmnn/pyberny/issues/32), [transition state search](https://github.com/jhrmnn/pyberny/issues/4).

PyBerny is available in [PySCF](https://sunqm.github.io/pyscf/geomopt.html#pyberny) and [QCEngine](http://docs.qcarchive.molssi.org/projects/QCEngine/en/latest/index.html?highlight=pyberny#backends).

## Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/):

```
pip install -U pyberny
```

## Example

The snippet below optimizes a geometry from `geom.xyz` end-to-end using
[MOPAC](http://openmopac.net) as the energy/gradient backend:

```python
from berny import Berny, geomlib, optimize
from berny.solvers import MopacSolver

optimizer = Berny(geomlib.readfile('geom.xyz'))
relaxed = optimize(optimizer, MopacSolver())
```

To plug in a different backend, replace `MopacSolver()` with any
coroutine that follows the same interface (see the
[documentation](https://jhrmnn.github.io/pyberny/master/getting-started.html)
for the manual generator pattern and the solver protocol).

## Citing

If you use PyBerny in published work, please cite it via its Zenodo DOI:
[10.5281/zenodo.3695037](https://doi.org/10.5281/zenodo.3695037). The
linked record resolves to the latest release and lists per-version DOIs
for citing a specific version.

## Links

- Documentation: https://jhrmnn.github.io/pyberny
