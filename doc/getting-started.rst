Getting started
===============

Dependencies
------------

Python >=3.9 with NumPy.

Usage
-----

The Python API consists of coroutine :class:`~berny.Berny` and function
:func:`~berny.optimize`::

   from berny import Berny, geomlib
   from berny.solvers import MopacSolver

   optimizer = Berny(geomlib.readfile('start.xyz'))
   solver = MopacSolver()
   next(solver)
   for geom in optimizer:
       energy, gradients = solver.send((list(geom), geom.lattice))
       optimizer.send((energy, gradients))
   relaxed = geom

or equivalently::

   from berny import Berny, geomlib, optimize
   from berny.solvers import MopacSolver

   relaxed = optimize(Berny(geomlib.readfile('start.xyz')), MopacSolver())

For PySCF, use upstream PySCF's own bridge to pyberny (it imports ``Berny``
internally and handles unit conversion, ghost atoms, symmetry, and the
gradient scanner)::

   from pyscf import gto, scf
   from pyscf.geomopt.berny_solver import optimize

   mol = gto.M(atom='start.xyz', basis='3-21G')
   mol_opt = optimize(scf.RHF(mol))

The 19-molecule benchmark under ``tests/data/birkholz_schlegel/`` reproduces
the geometries of [BirkholzTCA16]_; ``scripts/benchmark.py`` runs the suite
through that PySCF bridge (and optionally :func:`~berny.solvers.MopacSolver`)
and prints a step-count comparison table.

A different option is to use the package via a command-line or socket
interface defined by the ``berny`` command:

.. code:: none

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

A call with ``--init`` corresponds to initializing the :class:`~berny.Berny`
object where the geometry is taken from standard input, assuming format
``--format``. The object is then pickled to ``berny.pickle`` and the program
quits. Subsequent calls to ``berny`` recover the :class:`~berny.Berny` object,
read energy and gradients from the standard input (first line is energy,
subsequent lines correspond to Cartesian gradients of individual atoms, all in
atomic units) and write the new structure estimate to standard output. An
example usage could look like this:

.. code:: bash

   #!/bin/bash
   berny --init params.json <start.xyz
   cat start.xyz >current.xyz
   while true; do
       # calculate energy and gradients of current.xyz
       cat energy_gradients.txt | berny >next.xyz
       if [[ $? == 0 ]]; then  # minimum reached
           break
       fi
       mv next.xyz current.xyz
   done

Alternatively, one can start an optimizer server with the ``--socket``
option on a given address and port. This initiates the :class:`~berny.Berny`
object and waits for connections, in which it expects to receive energy and
gradients as a request (in the same format as above) and responds with a new
structure estimate in a given format. Example usage would be

.. code:: bash

   #!/bin/bash
   berny -s localhost 25000 -f xyz <start.xyz &
   cat start.xyz >current.xyz
   while true; do
       # calculate energy and gradients of current.xyz
       cat energy_gradients.txt | nc localhost 25000 >next.xyz
       if [[ ! -s next.xyz ]]; then  # minimum reached
           break
       fi
       mv next.xyz current.xyz
   done
