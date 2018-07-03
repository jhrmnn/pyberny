# `berny` — Molecular optimizer

This Python 2/3 package can optimize molecular and crystal structures with respect to total energy, using nuclear gradient information.

In each step, it takes energy and Cartesian gradients as an input, and returns a new structure estimate.

The algorithm is an amalgam of several techniques, comprising redundant internal coordinates, iterative Hessian estimate, trust region, line search, and coordinate weighing, mostly inspired by the optimizer in the [Gaussian](http://gaussian.com) program.

## Installing

Install and update using [pip](https://pip.pypa.io/en/stable/quickstart/):

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

-   Documentation: <https://pyberny.readthedocs.io/en/latest/>
