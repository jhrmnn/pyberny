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

``xtb_gfn2_steps`` records the GFN2-xTB pyberny step count for each
molecule. It was seeded from the first ``workflow_dispatch`` baseline run of
the ``Benchmark`` workflow (benchmark ``oligomers``, solver ``xtb``) on
commit ``937a289``: 86 of the 91 molecules converged and carry their
measured step count (3–82 steps, 1599 total); a ``null`` reference disables
the regression gate for a row (see ``scripts/benchmark.py``), so subsequent
runs check the 86 seeded molecules within the usual 7 %/±2-step tolerance
while leaving the five non-convergers ungated. As noted in
``birkholz_schlegel/SOURCE.md``, GFN2-xTB step counts are not bitwise
reproducible across runners, so an occasional row drifting past tolerance is
flaky rather than a regression. Reseed by rerunning the baseline dispatch.

The five ``null`` rows are documented non-convergers under GFN2-xTB at the
default settings (same convention as ``bisphenol_a`` in
``birkholz_schlegel``): ``nylon6_n8`` and ``PPE_n8`` fail at the tblite SCF
(``SCF not converged in 250 cycles``), while ``nylon6_n5``, ``nylon6_n6``,
and ``polyglycine_n8`` -- large floppy chains -- do not satisfy the
geometry-convergence criteria within pyberny's default step ceiling.

There are no ``paper_steps``/``paper_steps_method``/``paper_steps_basis``
entries because this set has no published reference optimization; the pyscf
runner (which reads those keys) is therefore not wired up for it, and the
CI dropdown offers ``oligomers`` under the ``xtb`` solver only.
