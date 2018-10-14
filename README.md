# `berny` — Molecular optimizer

[![build](https://img.shields.io/travis/azag0/pyberny/master.svg)](https://travis-ci.org/azag0/pyberny)
[![coverage](https://img.shields.io/codecov/c/github/azag0/pyberny.svg)](https://codecov.io/gh/azag0/pyberny)
![python](https://img.shields.io/pypi/pyversions/pyberny.svg)
[![pypi](https://img.shields.io/pypi/v/pyberny.svg)](https://pypi.org/project/pyberny/)
[![commits since](https://img.shields.io/github/commits-since/azag0/pyberny/latest.svg)](https://github.com/azag0/pyberny/releases)
[![last commit](https://img.shields.io/github/last-commit/azag0/pyberny.svg)](https://github.com/azag0/pyberny/commits/master)
[![license](https://img.shields.io/github/license/azag0/pyberny.svg)](https://github.com/azag0/pyberny/blob/master/LICENSE)

This Python 2/3 package can optimize molecular and crystal structures with respect to total energy, using nuclear gradient information.

In each step, it takes energy and Cartesian gradients as an input, and returns a new structure estimate.

The algorithm is an amalgam of several techniques, comprising redundant internal coordinates, iterative Hessian estimate, trust region, line search, and coordinate weighing, mostly inspired by the optimizer in the [Gaussian](http://gaussian.com) program.

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
