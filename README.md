# Berny

[![build](https://img.shields.io/travis/azag0/pyberny/master.svg)](https://travis-ci.org/azag0/pyberny)
[![coverage](https://img.shields.io/codecov/c/github/azag0/pyberny.svg)](https://codecov.io/gh/azag0/pyberny)
![python](https://img.shields.io/pypi/pyversions/pyberny.svg)
[![pypi](https://img.shields.io/pypi/v/pyberny.svg)](https://pypi.org/project/pyberny/)
[![commits since](https://img.shields.io/github/commits-since/azag0/pyberny/latest.svg)](https://github.com/azag0/pyberny/releases)
[![last commit](https://img.shields.io/github/last-commit/azag0/pyberny.svg)](https://github.com/azag0/pyberny/commits/master)
[![license](https://img.shields.io/github/license/azag0/pyberny.svg)](https://github.com/azag0/pyberny/blob/master/LICENSE)

Berny is an optimizer of molecular geometries with respect to the total energy, using nuclear gradient information.

In each step, it takes energy and Cartesian gradients as an input, and returns a new equilibrium structure estimate.

The package implements a single optimization algorithm, which is an amalgam of several techniques, comprising the quasi-Newton method, redundant internal coordinates, an iterative Hessian approximation, a trust region scheme, and linear search. The algorithm is described in more detailed in the [documentation](https://azag0.github.io/pyberny/algorithm.html).

Several desirable features are missing at the moment but planned, some of them being actively worked on (help is always welcome): [crystal geometries](https://github.com/azag0/pyberny/issues/5), [coordinate constraints](https://github.com/azag0/pyberny/issues/14), [coordinate weighting](https://github.com/azag0/pyberny/issues/32), [transition state search](https://github.com/azag0/pyberny/issues/4).

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

- Documentation: <https://azag0.github.io/pyberny>
