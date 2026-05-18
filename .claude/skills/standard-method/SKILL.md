---
name: standard-method
description: Fetch the Birkholz & Schlegel 2016 paper and its supporting information — the "standard method" (SM) that PyBerny's optimization algorithm follows — plus the papers it cites. Use when the user references the SM, asks about algorithm details/equations/convergence criteria, asks for test structures, asks about any of the SM's cited references (RFO, BFGS/SR1/PSB/Bofill, trust region, redundant internal coordinates, etc.), or asks anything that would be answered by reading [BirkholzTCA16] (cited in `doc/algorithm.rst`).
---

# Standard method paper (Birkholz & Schlegel 2016) and supporting literature

`doc/algorithm.rst` notes that PyBerny "loosely follows" the **standard method (SM)** described in the appendix of:

> Birkholz, A. B. & Schlegel, H. B. *Exploration of some refinements to geometry optimization methods.* Theor. Chem. Acc. 135, (2016). DOI: [10.1007/s00214-016-1847-3](http://dx.doi.org/10.1007/s00214-016-1847-3)

The SM appendix is self-contained for headline formulas but defers several pieces to other papers (RFO derivation, trust-region details, Bofill MSP, pRFO/EF for TS) or omits them entirely (initial Hessian, bond detection, B-matrix derivatives, coordinate formulas). The bundle therefore also includes those cited / canonical sources.

None of these papers are checked into the repo (copyright). They are downloaded from `PAPERS_URL`.

## How to access

The container is ephemeral, so re-download as needed:

```bash
wget -q "$PAPERS_URL" -O /tmp/standard-method.zip
unzip -o /tmp/standard-method.zip -d /tmp/
```

The `Read` tool renders PDFs visually (via poppler-utils) — equations, bold matrices, subscripts and fractions all come through cleanly. Always pass a `pages` range for multi-page PDFs.

## Bundle contents (under `/tmp/standard-method/`)

### Primary

- `390.pdf` — Birkholz & Schlegel 2016, 12 pp. Appendix "Standard method" (pp. 8–12) is what PyBerny implements. Numbered Eqs. 14–33 cover B-matrix, generalised inverse, back-transformation, RFO, trust region, BFGS/SR1/PSB/MSP, convergence.
- `214_2016_1847_MOESM1_ESM.txt` — SI. Plain-text XYZ coordinates (Å) for the 20 test molecules (Artemisinin, Aspartame, …), each preceded by name and `charge multiplicity`.

### SM appendix's *explicit* deferrals (`see Ref. [X]`)

- `Banerjee, Adams, Simons - 1985 - Search for Stationary Points on Surfaces.pdf` — SM ref [26]. RFO derivation and pRFO for transition states. The Banerjee article begins a few pages into the PDF (the journal-issue layout puts a different article first).
- `baker1996.pdf` — SM ref [17]. Baker & Chan 1996, eigenvector-following for TS optimization (referenced alongside [26] for pRFO).
- `Dennis, Numerical Methods for Unconstrained Optimization and Nonlinear Equations.pdf` — SM ref [27]. Dennis & Schnabel textbook, 395 pp. Adaptive trust-region material is in chapter 6.
- `bofill1994.pdf` — SM ref [14]. Bofill 1994, *J. Comput. Chem.* 15(1):1. Source of the MSP / "Bofill" Hessian update (Eqs. 8–13 of that paper define the φ-mixed family that the SM cites as Eqs. 32–33).

### SM appendix's *implicit* / canonical-source references

- `peng1996.pdf` — SM ref [16]. Peng, Ayala, Schlegel & Frisch 1996. The RIC machinery PyBerny implements (bond-detection rule, recursive dihedrals through linear atoms, projector + α(1−P) penalty in Eq. 9 with α = 1000 au — *this is the source of PyBerny's hard-coded `1000·(I − proj)` in `berny.py:191`*, not an ad-hoc constant). Iterative back-transformation is Eq. 11.
- `1992_Pulay_2856.pdf` — SM ref [3]. Pulay & Fogarasi 1992, *J. Chem. Phys.* 96:2856. Generalised inverse, projector approach (basis for SM Eqs. 18–21).
- `Schlegel - 1984 - Estimating the Hessian for gradient-type geometry optimizations.pdf` — SM ref [7]. Analytic B-matrix derivatives for stretch/bend/torsion coordinates (the SM doesn't write these out; PyBerny's `coord.eval(..., grad=True)` implements them).
- `Schlegel11_Geometry_optimization.pdf` — SM ref [1]. Schlegel 2011 *WIREs Comput. Mol. Sci.* review. Useful overview context, especially for line searches (not in the SM but used by PyBerny).
- `Swart, Bickelhaupt - 2006 - Optimization of Strong and Weak Coordinates.pdf` — `[SwartIJQC06]` from `doc/algorithm.rst`. The Lindh-style diagonal initial Hessian PyBerny uses; fills the gap the SM leaves silent.

## When to consult which paper

| Need | File |
|---|---|
| Exact SM convergence criteria, sketch of full algorithm | `390.pdf` appendix |
| Test geometries to reproduce results / build test cases | `214_…_MOESM1_ESM.txt` |
| Why the augmented-Hessian eigenvector gives the RFO step | Banerjee 1985 |
| pRFO / eigenvector-following for TS | Banerjee 1985, Baker 1996 |
| Trust-region update theory & convergence guarantees | Dennis & Schnabel, ch. 6 |
| Bofill / MSP update derivation | Bofill 1994 |
| RIC construction details (bonds, dihedrals through linears, `α(1−P)` penalty, iterative back-transformation) | Peng 1996 |
| Generalised inverse + projector formulation | Pulay-Fogarasi 1992 |
| Analytic B-matrix derivatives | Schlegel 1984 |
| Background, line-search rationale, overall context | Schlegel 2011 |
| Initial Hessian model | Swart-Bickelhaupt 2006 |

## Notes for reviewing / extending PyBerny against the SM

- PyBerny's `1000·(I − proj)` penalty in `berny.py:191` comes from Peng 1996 Eq. 9 (α = 1000 au) — not ad-hoc.
- The SM omits the initial Hessian, bond detection, and B-matrix derivatives — those are PyBerny's own choices, sourced from Swart-Bickelhaupt, Peng et al., and Schlegel 1984 respectively.
- Issue #29 ("fully conform to the SM") and issue #30 (linear bends with dummy atoms) are the open TODOs.

If `PAPERS_URL` is unset, ask the user — do not invent a URL.
