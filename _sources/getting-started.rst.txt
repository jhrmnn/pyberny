Getting started
===============

Dependencies
------------

Python 2.7 or >=3.5 with Numpy

Usage
-----

The Python API consists of coroutine :py:func:`~berny.Berny` and function
:py:func:`~berny.optimize`::

   from berny import Berny, geomlib
   from berny.solvers import MopacSolver

   optimizer = Berny(geomlib.readfile('start.xyz'), debug=True)
   solver = MopacSolver()
   for geom in optimizer:
       energy, gradients = solver(geom)
       optimizer.send((energy, gradients))
    relaxed = geom

or equivalently::

   from berny import Berny, geomlib, optimize
   from berny.solvers import MopacSolver

   relaxed = optimize(Berny(geomlib.readfile('start.xyz')), MopacSolver())

A different option is to use the package via a command-line or socket
interface defined in ``berny``::

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

A call with ``--init`` corresponds to initializing the ``Berny`` object
where the geometry is taken from standard input, asssuming format
``--format``. The object is then pickled to ``berny.pickle`` and the
program quits. Subsequent calls to ``berny`` recover the ``Berny``
object, read energy and gradients from the standard input (first line is
energy, subsequent lines correspond to Cartesian gradients of individual
atoms, all in atomic units) and write the new structure estimate to
standard output. An example usage could look like this:

.. code:: bash

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

Alternatively, one can start an optimizer server with the ``--socket``
option on a given address and port. This initiates the ``Berny`` object
and waits for connections, in which it expects to receive energy and
gradients as a request (in the same format as above) and responds with a
new structure estimate in a given format. Example usage would be

.. code:: bash

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
