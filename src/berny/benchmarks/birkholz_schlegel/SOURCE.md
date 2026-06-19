# Birkholz–Schlegel 2016 benchmark geometries

The 19 ``.xyz`` files in this directory are starting geometries of the
benchmark set published as electronic supplementary material (ESM) of:

> A. B. Birkholz and H. B. Schlegel,
> *Exploration of some refinements to geometry optimization methods*,
> **Theor. Chem. Acc.** 135, 84 (2016).
> [doi:10.1007/s00214-016-1847-3](https://doi.org/10.1007/s00214-016-1847-3)

Source ESM file: ``214_2016_1847_MOESM1_ESM.txt`` (hosted by Springer
Nature). Each ``.xyz`` here was produced by ``scripts/split_birkholz_si.py``
splitting that file into one molecule per file; element symbols are copied
as-is and coordinates are parsed and rewritten with ``{:18.10f}`` formatting
(numerically identical to the ESM to all retained digits, but not character-
for-character verbatim).

The paper describes a 20-molecule test set; the SI we have ships only 19 —
**Aspartame** (paper Table 1, Standard column: 30 steps) is referenced in
the paper but missing from the ESM and is therefore absent from this
benchmark.

``reference.json`` records charge, multiplicity, and the paper's
``paper_steps`` (from Table 1, "Standard" column) for each molecule, along
with the QM ``paper_steps_method`` and ``paper_steps_basis`` used for that
row (HF/3-21G for all molecules except EASC and Vitamin C, which use
B3LYP/6-31G(d,p)). The ``pyberny_steps`` and ``xtb_gfn2_steps`` fields are
populated by running ``scripts/benchmark.py`` and committing the measured
counts as a regression baseline.

``pyberny_steps`` records the PySCF step counts from a GitHub Actions run
using the method and basis in ``paper_steps_method``/``paper_steps_basis``
(HF/3-21G or B3LYP/6-31G(d,p) per molecule). Two molecules are left
``null``: ``azadirachtin`` is excluded from PySCF runs because its ~526 s/call
cost would exceed GitHub's 6-hour job cap; ``ochratoxin_a`` did not converge
within the 100-step default ceiling.

``xtb_gfn2_steps`` records the GFN2-xTB step counts (evaluated through the
``tblite`` library; see ``berny.solvers.XTBSolver``). All 19 molecules *converge*
under xTB within the benchmark's 130-step ceiling -- including ``ochratoxin_a``,
which does not converge under PySCF. A single GFN2-xTB energy+gradient call is
sub-second even for the 95-atom ``azadirachtin`` (~0.75 s), so unlike PySCF
nothing has to be excluded on cost grounds.

Three molecules are nonetheless left ``null`` -- ``bisphenol_a``, ``maltose``
and ``penicillin_v``. These have flat minima where xTB's step count is not
reproducible: ``tblite``'s OpenMP reductions are not bitwise-deterministic, and
near a flat minimum the resulting ~1e-9 Ha noise reroutes the trust-radius path,
so repeated runs on a *single* host already scatter well past the 7%/2-step gate
(``bisphenol_a`` ranged 63-85 over four runs; ``penicillin_v`` 52-64;
``maltose`` 76-86), so they have no meaningful value to gate on. The other 16
molecules are reproducible to 0-1 steps across runs. Even those stable counts
are not guaranteed bitwise across *different* hosts, so the same 7%/2-step drift
tolerance applies; the committed values are a single-host baseline and should be
confirmed (and refreshed if needed) from the first CI ``workflow_dispatch`` xTB
run.

xTB needs no fast/full batch split (no easc-style per-call outlier to
quarantine): ``scripts/plan_batches.py`` bins it into four cost-balanced shards
(per-call ``tblite`` time × ``xtb_gfn2_steps``, falling back to ``paper_steps``
for the three ``null`` rows) used for both ``birkholz-fast`` and
``birkholz-full``. The ``null`` molecules still *run* in those shards; only the
regression gate skips them.

Coordinate data is treated as factual and is redistributed under pyberny's
MPL-2.0 license, with attribution to Birkholz & Schlegel via this file and
via the citation embedded in the comment line of each ``.xyz``. The
original Springer ESM ``.txt`` is *not* redistributed by this project.

Cite the paper when reporting results obtained with this benchmark.
