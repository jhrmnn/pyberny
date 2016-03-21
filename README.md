# `berny` — Molecular optimizer

This Python 2/3 package can optimize molecular and crystal structures with respect to total energy, using nuclear gradient information.

In each step, it takes energy and Cartesian gradients as an input, and returns a new structure estimate.

The algorithm is an amalgam of several techniques, comprising redundant internal coordinates, iterative Hessian estimate, trust region, line search, and coordinate weighing, mostly inspired by the optimizer in the [Gaussian](http://gaussian.com) program.

## Dependencies

-   Python 2.7 or 3.5 with Numpy

## Usage

The Python API is defined in the files in directory `bernylib`. The simplest usage could look like the following:

```python
from bernylib import berny, geomlib
geom = geomlib.readfile('start.xyz')
optimizer = berny.Berny(geom, params={'debug': 'debug.json'})
while True:
    # calculate energy and gradients of geom
    geom_next = optimizer.step(energy, gradients)
    if not geom_next:  # minimum reached
        geom.write('final.xyz')
        break
    geom = geom_next
```

A different option is to use the package via a command-line or socket interface defined in `berny`:

```
usage: berny [-h] [--init] [-f {xyz,aims}] [-s host port] [paramfile]

positional arguments:
  paramfile             Optional optimization parameters as JSON

optional arguments:
  -h, --help            show this help message and exit
  --init                Initialize Berny optimizer.
  -f {xyz,aims}, --format {xyz,aims}
                        Format of geometry
  -s host port, --socket host port
                        Listen on given address
```

A call with `--init`  corresponds to initializing the `Berny` object where the geometry is taken from standard input, asssuming format `--format`.  The object is then pickled to `berny.pickle` and the program quits. Subsequent calls to `berny` recover the `Berny` object, read energy and gradients from the standard input (first line is energy, subsequent lines correspond to Cartesian gradients of individual atoms, all in atomic units) and write the new structure estimate to standard output. An example usage could look like this:

```bash
#!/bin/bash
./berny --init params.json <start.xyz
cat start.xyz >current.xyz
while true; do
	# calculate energy and gradients of current.xyz
    cat energy_gradients.txt | ./berny >next.xyz
    if [[ $? == 0 ]]; then  # minimum reached
        break
    fi
	mv next.xyz current.xyz
done
```

Alternatively, one can start an optimizer server with the `--socket` option on a given address and port. This initiates the `Berny` object and waits for connections, in which it expects to receive energy and gradients as a request (in the same format as above) and responds with a new structure estimate in a given format. Example usage would be

```bash
#!/bin/bash
./berny -s localhost 25000 -f xyz <start.xyz &
cat start.xyz >current.xyz
while true; do
	# calculate energy and gradients of current.xyz
    cat energy_gradients.txt | nc localhost 25000 >next.xyz
    if [[ ! -s next.xyz ]]; then  # minimum reached
    	break
    fi
	mv next.xyz current.xyz
done
```

## Parameters

All parameters have default values given in `bernylib.berny.defaults`.

-   `gradientmax = 0.45e-3`, `gradientrms = 0.3e-3`, `stepmax = 1.8e-3`, `steprms = 1.2e-3`. Convergence criteria in atomic units (`step` refers to the step in internal coordinates, assuming radian units for angles).


-   `maxsteps = 100`. Maximum number of optimization steps. If the maximum is reached, the optimization is

-   `trust = 0.3`. Initial trust radius in atomic units. It is the maximum RMS of the quadratic step (see below).

-   `debug = None`. Path to a file where debugging info in pretty printed JSON format should be stored. If none, the debug info is not recorded.

## Algorithm

### Redundant internal coordinates

1.  All bonds shorter than 1.3 times the sum of covalent radii are created.
2.  If there are unconnected fragments, all bonds between unconnected fragments shorter than the sum of van der Waals radii plus *d* are created, with *d* starting at 0 and increasing by 1 angstrom, until all fragments are connected.
3.  All angles greater than 45° are created.
4.  All dihedrals with 1–2–3, 2–3–4 angles both greater than 45° are created. If one of the angles is zero, so that three atoms lie on a line, they are used as a new base for a dihedral. This process is recursively repeated.
5.  In the case of a crystal, just the internal coordinate closest to the original unit cell is retained from all periodic images.

### Generalized inverse

The Wilson matrix **B**, which relates differences in the internal redundant coordinates to differences in the Cartesian coordinates, is in general non-square and non-invertible. A generalized inverse of a matrix is obtained by taking its singular value decomposition and inverting only the nonzero singular values. For invertible matrices, this is equivalent to an ordinary inverse. In practice, the zero values are in fact nonzero but several orders of magnitude smaller than the true nonzero values.