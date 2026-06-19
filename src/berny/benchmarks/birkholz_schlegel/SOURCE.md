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

``mopac_pm7_steps`` is left ``null`` for ``azadirachtin`` and ``raffinose``:
both have very flat minima where, once ``MopacSolver`` switched to the
high-precision ``AUX`` file (lowering the energy noise floor by ~1000×), the
discrete Fletcher trust-radius rule reacts to the real ~1e-6 Ha energy
changes and limit-cycles on the trust-region sphere at the minimum.
``azadirachtin`` never satisfies the step-convergence criteria within the
benchmark's 130-step ceiling; ``raffinose`` does eventually converge, but at a
step count that is not reproducible across hosts (e.g. 76 vs 87 on otherwise
identical GitHub Actions runs), so it has no meaningful baseline to gate on.
Both are documented non-convergers rather than a regression: the flat-minimum
sawtooth is orthogonal to this AUX/noise-floor change and is addressed
separately by the sphere-convergence gate, not by ``energy_noise``. The
reference values come from a run on GitHub Actions' ``ubuntu-latest`` (MOPAC
23.2.5, BLAS threads pinned to the runner's physical-core count); MOPAC PM7 is
not bitwise-reproducible across hosts, so ``scripts/benchmark.py`` and
``scripts/aggregate_benchmark.py`` allow each row to drift from its reference
by up to 7% (with an absolute floor of 2 steps) before failing the run.

``pyberny_steps`` records the PySCF step counts from a GitHub Actions run
using the method and basis in ``paper_steps_method``/``paper_steps_basis``
(HF/3-21G or B3LYP/6-31G(d,p) per molecule). Two molecules are left
``null``: ``azadirachtin`` is excluded from PySCF runs because its ~526 s/call
cost would exceed GitHub's 6-hour job cap; ``ochratoxin_a`` did not converge
within the 100-step default ceiling.

Coordinate data is treated as factual and is redistributed under pyberny's
MPL-2.0 license, with attribution to Birkholz & Schlegel via this file and
via the citation embedded in the comment line of each ``.xyz``. The
original Springer ESM ``.txt`` is *not* redistributed by this project.

Cite the paper when reporting results obtained with this benchmark.
