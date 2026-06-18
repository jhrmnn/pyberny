# Oligomer benchmark set

This benchmark exercises pyberny's optimizer on length series of common
oligomers — rigid acenes, poly-ynes, poly(phenylene ethynylene) (PPE),
polythiophene, three oligo-peptides (poly-glycine / -alanine / -serine),
and four commodity polymers (polyethylene, polypropylene, nylon-6, PEG).
Each family is a chain-length sweep (``_n1`` … ``_nN``), so the set probes
how step count and convergence scale with system size within a chemically
homogeneous series.

## Geometries

Unlike the ``birkholz_schlegel`` and ``baker_shajan_2023`` sets, the 91
starting ``.xyz`` geometries are **not** redistributed inside this package.
They live in the upstream repository

> https://github.com/ghutchis/oligomer-benchmarks
> (Geoff Hutchison; proposed for pyberny in issue #126)

which is vendored as the git submodule ``external/oligomer-benchmarks``.
The submodule records only a commit pointer, so this repository does not
copy or relicense the coordinate data. Pinned commit:

    0147460b52f06c6b58b69c0ee51bf0ac19a684cf

``berny.benchmarks`` resolves each geometry through the submodule's
``xyz/<family>/<molecule>.xyz`` layout via the per-molecule ``file`` key in
``reference.json``. CI checks out submodules before running this benchmark;
a plain ``git clone`` without ``--recurse-submodules`` will leave
``external/oligomer-benchmarks`` empty and the benchmark unable to find its
geometries (run ``git submodule update --init`` to populate it).

## reference.json

``reference.json`` records, per molecule, ``atoms`` (header count of the
upstream ``.xyz``), ``family`` (the source subdirectory), ``file`` (path
relative to the submodule's ``xyz/`` root), ``charge``, ``mult``, and the
solver step-count baseline ``xtb_gfn2_steps``. All 91 molecules are
closed-shell neutral C/H/N/O/S organics, so every row has ``charge=0`` and
``mult=1``.

``xtb_gfn2_steps`` is ``null`` for every molecule: this set ships
**unseeded**. As with the other benchmarks, a ``null`` reference disables
the regression gate for that row (see ``scripts/benchmark.py``), so the
first ``workflow_dispatch`` run of ``Benchmark`` (benchmark ``oligomers``,
solver ``xtb``) is a baseline pass that simply reports measured step
counts without failing. Seed ``xtb_gfn2_steps`` from that baseline to
activate the per-row regression check, exactly as ``birkholz_schlegel`` and
``baker_shajan_2023`` were seeded.

There are no ``paper_steps``/``paper_steps_method``/``paper_steps_basis``
entries because this set has no published reference optimization; the pyscf
runner (which reads those keys) is therefore not wired up for it, and the
CI dropdown offers ``oligomers`` under the ``xtb`` solver only.
