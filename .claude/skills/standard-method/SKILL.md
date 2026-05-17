---
name: standard-method
description: Fetch the Birkholz & Schlegel 2016 paper and its supporting information — the "standard method" (SM) that PyBerny's optimization algorithm follows. Use when the user references the SM, asks about algorithm details/equations/convergence criteria, asks for test structures, or asks anything that would be answered by reading [BirkholzTCA16] (cited in `doc/algorithm.rst`).
---

# Standard method paper (Birkholz & Schlegel 2016)

`doc/algorithm.rst` notes that PyBerny "loosely follows" the **standard method (SM)** described in the appendix of:

> Birkholz, A. B. & Schlegel, H. B. *Exploration of some refinements to geometry optimization methods.* Theor. Chem. Acc. 135, (2016). DOI: [10.1007/s00214-016-1847-3](http://dx.doi.org/10.1007/s00214-016-1847-3)

The paper and its SI are not checked into the repo (copyright). They are made available via the `PAPERS_URL` environment variable.

## How to access

The container is ephemeral, so re-download as needed:

```bash
wget -q "$PAPERS_URL" -O /tmp/standard-method.zip
unzip -o /tmp/standard-method.zip -d /tmp/
```

Resulting files under `/tmp/standard-method/`:

- `390.pdf` — main paper (~874 KB). Use `Read` with a `pages` range; the appendix describes the SM step by step.
- `214_2016_1847_MOESM1_ESM.txt` — supporting information (~52 KB). Plain-text XYZ coordinates (Å) for the test molecules used in the paper (Artemisinin, etc.), each preceded by name and `charge multiplicity`.

## When to consult it

Reach for these files when you need authoritative detail that goes beyond `doc/algorithm.rst`, e.g.:

- Exact convergence criteria the SM prescribes (referenced in step 11 of the algorithm sketch).
- Trust-region update formula details, RFO step formulation, BFGS variant.
- A reference geometry to reproduce a result or build a test case.
- Resolving the "fully conform to the SM" TODO ([issue #29](https://github.com/jhrmnn/pyberny/issues/29)).

If `PAPERS_URL` is unset, ask the user — do not invent a URL.
