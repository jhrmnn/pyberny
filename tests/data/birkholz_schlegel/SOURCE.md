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
B3LYP/6-31G(d,p)). The ``pyberny_steps`` and ``mopac_pm7_steps`` fields are
populated by running ``scripts/benchmark.py`` and committing the measured
counts as a regression baseline.

``mopac_pm7_steps`` is left ``null`` for ``bisphenol_a``, ``ochratoxin_a``
and ``raffinose``: these did not converge with PM7 within pyberny's default
100-step ceiling and so have no meaningful step count to record.

Coordinate data is treated as factual and is redistributed under pyberny's
MPL-2.0 license, with attribution to Birkholz & Schlegel via this file and
via the citation embedded in the comment line of each ``.xyz``. The
original Springer ESM ``.txt`` is *not* redistributed by this project.

Cite the paper when reporting results obtained with this benchmark.
