# Shajan et al. 2023 Baker-set benchmark geometries

The 30 ``.xyz`` files in this directory are the starting geometries of
Baker's classic 30-molecule test set as redistributed in the Supporting
Information of:

> A. Shajan, M. Manathunga, A. W. Goetz, K. M. Merz, Jr.
> *Geometry Optimization: A Comparison of Different Open-Source
> Geometry Optimizers*. chemrxiv, v4, 22 August 2023.
> [doi:10.26434/chemrxiv-2023-7r7qn-v4](https://doi.org/10.26434/chemrxiv-2023-7r7qn-v4)
> (preprint; not peer-reviewed).

Source SI archive: ``input_files_to_different_geometry_optimizers.zip``,
folder ``quick-geopt-si/ASE_inputs/``. The SI is released under CC-BY 4.0.
Each ``.xyz`` here was produced by ``scripts/convert_baker_si.py`` reading
that ZIP: element symbols are copied as-is and coordinates are reformatted
with ``{:18.10f}``. The converter also corrects one upstream typo
(``Acetone.xyz`` declared 12 atoms in its header but listed only 10; the
committed file has the right count).

``reference.json`` records charge, multiplicity, and the paper's
``paper_steps`` (from Table 1, "ASE/Berny" column) for each molecule. All
30 molecules in the Baker set are closed-shell neutral organics, so every
row has ``charge=0`` and ``mult=1``. The reference method is HF with the
6-31G** basis set; the paper used QUICK as the gradient backend.

Paper-side convergence in the ASE/Berny column was driven by a single
criterion:

    opt = Berny(geom, logfile='FILE.log')
    opt.run(fmax=0.00045*Hartree/Bohr)

i.e. ASE's wrapper only sets ``fmax`` (max Cartesian gradient component
in Hartree/Bohr); PyBerny's other thresholds (``gradientrms``, ``stepmax``,
``steprms``) keep their defaults. ``scripts/benchmark.py`` does **not**
go through ASE — it drives PyBerny via ``pyscf.geomopt.berny_solver``
with PySCF/HF/6-31G** for energies and gradients and PyBerny's standard
multi-criterion convergence on internal coordinates. Step counts can
therefore drift from the paper's numbers for two independent reasons:

1. SCF engine: PySCF instead of QUICK changes numerical gradients at the
   digits that matter for the step counter.
2. Convergence: PyBerny's native four-criterion test is stricter than
   ASE/Berny's ``fmax``-only mapping, so a molecule that satisfies
   ``fmax`` may need extra cycles to also satisfy ``stepmax`` etc.

``scripts/benchmark.py`` allows each row to drift from the paper-reported
``paper_steps`` by up to 7% (with an absolute floor of 2 steps) before
failing the run -- same rule as the Birkholz-Schlegel benchmark.

``pyberny_steps`` and ``mopac_pm7_steps`` were seeded from the first
manual workflow_dispatch baseline run (PR #84, baker × both at HF/6-31G\*\*
through PySCF and PM7 through MOPAC). PySCF reaches the same minimum the
paper found for every one of the 30 molecules with a total of 208 steps
vs. the paper's 190 (per-row delta typically ≤1 step, with histidine,
dimethylpentane, menthone, disilyl\_ether, trisilacyclohexane\_135,
pterin needing a few extra cycles past ASE/Berny's ``fmax``-only test).
MOPAC PM7 totals 274 steps over 29 molecules; ``caffeine`` is left
``null`` because PM7 hits the 110-step ceiling on this molecule, the same
documented-non-converger convention used by ``bisphenol_a`` in the
``birkholz_schlegel`` set.

Coordinate data is treated as factual and is redistributed under
pyberny's MPL-2.0 license, with attribution to Shajan et al. via this
file and via the citation embedded in the comment line of each ``.xyz``.
The original SI ZIP and PDF are *not* redistributed by this project.

Cite the paper when reporting results obtained with this benchmark.
