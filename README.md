# PyBerny

[![build](https://img.shields.io/travis/com/jhrmnn/pyberny/master.svg)](https://travis-ci.com/jhrmnn/pyberny)
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

The package implements a single optimization algorithm, which is an amalgam of several techniques, comprising the quasi-Newton method, redundant internal coordinates, an iterative Hessian approximation, a trust region scheme, and linear search. The algorithm is described in more detailed in the [documentation](https://jhrmnn.github.io/pyberny/algorithm.html).

Several desirable features are missing at the moment but planned, some of them being actively worked on (help is always welcome): [crystal geometries](https://github.com/jhrmnn/pyberny/issues/5), [coordinate constraints](https://github.com/jhrmnn/pyberny/issues/14), [coordinate weighting](https://github.com/jhrmnn/pyberny/issues/32), [transition state search](https://github.com/jhrmnn/pyberny/issues/4).

PyBerny is available in [PySCF](https://sunqm.github.io/pyscf/geomopt.html#pyberny), [ASE](https://wiki.fysik.dtu.dk/ase/dev/ase/optimize.html?highlight=berny#pyberny), and [QCEngine](http://docs.qcarchive.molssi.org/projects/QCEngine/en/latest/index.html?highlight=pyberny#backends).

## Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/):

```
pip install -U pyberny
```

## Example

```python
from berny import Berny, geomlib

optimizer = Berny(geomlib.readfile('geom.xyz'))
for geom in optimizer:
    # get energy and gradients for geom
    optimizer.send((energy, gradients))
```

## Links

- Documentation: https://jhrmnn.github.io/pyberny
