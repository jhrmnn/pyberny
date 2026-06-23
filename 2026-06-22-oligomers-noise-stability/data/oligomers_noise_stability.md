# Noise-stability sweep: oligomers / xtb

- Noise amplitudes (Angstrom RMS per Cartesian coord): 0.02, 0.05, 0.1, 0.2, 0.3
- Seeds per (molecule, amplitude): 3
- "Same minimum" window: |E - E_ref| <= 0.1 kcal/mol
- Wall time: 0.0 s

## Headline

- Noisy trials: 645; converged: 554 (85.9%), hit step ceiling: 18 (2.8%), errored before optimizing: 73 (11.3%)
- Converged trials landing in a different basin than the unperturbed run: 44 (6.82%)
- Molecules with any coordinate-build/other error: naphthalene, tetracene, heptacene, octacene, PPE_n2, PPE_n3, PPE_n4, PPE_n5, polyyne_n6, polyyne_n7, polyyne_n8, polyalanine_n1, polyalanine_n3, polyalanine_n4, polyalanine_n5, polyglycine_n1, polyglycine_n3, polyglycine_n5, polyglycine_n6, polyethylene_n2, polyethylene_n3, polyethylene_n4, PEG_n4
- Molecules with any step-ceiling non-convergence: polyyne_n7, polyyne_n8, polyalanine_n5, polyglycine_n4, polyglycine_n5, polyglycine_n6, PEG_n3, PEG_n4, thiophene_n3, thiophene_n4
- Molecules with any different-basin outcome: naphthalene, anthracene, tetracene, pentacene, hexacene, heptacene, octacene, nonacene, polyalanine_n1, polyalanine_n2, polyalanine_n3, polyglycine_n1, polyglycine_n2, polyglycine_n3, polyglycine_n5, polyglycine_n6, PEG_n1, PEG_n2, PEG_n4, thiophene_n3

## Stability vs noise amplitude (aggregate over all molecules)

| sigma (A) | trials | converged | ceiling | error | same basin | diff basin | max |dE| (kcal/mol) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 129 | 118 | 1 | 10 | 115 | 3 | 0.182 |
| 0.05 | 129 | 117 | 2 | 10 | 114 | 3 | 0.182 |
| 0.1 | 129 | 118 | 3 | 8 | 110 | 8 | 0.182 |
| 0.2 | 129 | 113 | 3 | 13 | 104 | 9 | 98.711 |
| 0.3 | 129 | 88 | 9 | 32 | 67 | 21 | 142.452 |

## Per-molecule summary

| Molecule | Atoms | Ref steps | Noisy trials | Converged | Ceiling | Error | Diff basin | Steps (min/mean/max) | Max |dE| (kcal/mol) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| PPE_n5 | 62 | 8 | 15 | 0 | 0 | 15 | 0 | - | 0.000 |
| PPE_n3 | 38 | 10 | 15 | 1 | 0 | 14 | 0 | 47/47.0/47 | 0.000 |
| PPE_n4 | 50 | 10 | 15 | 2 | 0 | 13 | 0 | 47/62.0/77 | 0.000 |
| polyglycine_n6 | 54 | 66 | 15 | 7 | 7 | 1 | 1 | 57/81.1/100 | 8.977 |
| PPE_n2 | 26 | 7 | 15 | 9 | 0 | 6 | 0 | 14/27.3/44 | 0.000 |
| polyglycine_n5 | 47 | 51 | 15 | 11 | 2 | 2 | 5 | 40/49.3/64 | 0.159 |
| polyglycine_n3 | 33 | 34 | 15 | 12 | 0 | 3 | 3 | 27/43.4/67 | 0.121 |
| polyalanine_n5 | 62 | 27 | 15 | 12 | 1 | 2 | 0 | 27/34.3/47 | 0.000 |
| polyyne_n7 | 16 | 6 | 15 | 12 | 1 | 2 | 0 | 8/23.4/32 | 0.001 |
| polyyne_n8 | 18 | 7 | 15 | 12 | 2 | 1 | 0 | 9/35.3/64 | 0.002 |
| PEG_n4 | 31 | 6 | 15 | 13 | 1 | 1 | 1 | 7/22.2/47 | 1.311 |
| polyalanine_n4 | 52 | 21 | 15 | 13 | 0 | 2 | 0 | 22/36.9/61 | 0.000 |
| polyglycine_n1 | 19 | 7 | 15 | 14 | 0 | 1 | 14 | 23/35.9/52 | 0.182 |
| polyalanine_n3 | 42 | 35 | 15 | 14 | 0 | 1 | 2 | 37/52.5/81 | 134.144 |
| thiophene_n3 | 23 | 9 | 15 | 14 | 1 | 0 | 2 | 10/22.3/50 | 98.711 |
| heptacene | 50 | 22 | 15 | 14 | 0 | 1 | 1 | 19/38.2/94 | 58.456 |
| naphthalene | 18 | 8 | 15 | 14 | 0 | 1 | 1 | 10/19.9/48 | 84.356 |
| octacene | 54 | 23 | 15 | 14 | 0 | 1 | 1 | 18/31.0/67 | 45.103 |
| polyalanine_n1 | 22 | 22 | 15 | 14 | 0 | 1 | 1 | 22/32.5/57 | 31.241 |
| tetracene | 30 | 8 | 15 | 14 | 0 | 1 | 1 | 11/20.3/44 | 142.452 |
| PEG_n3 | 24 | 8 | 15 | 14 | 1 | 0 | 0 | 9/20.3/41 | 0.001 |
| polyethylene_n2 | 17 | 4 | 15 | 14 | 0 | 1 | 0 | 5/15.8/47 | 0.000 |
| polyethylene_n3 | 23 | 4 | 15 | 14 | 0 | 1 | 0 | 6/17.1/46 | 0.000 |
| polyethylene_n4 | 29 | 4 | 15 | 14 | 0 | 1 | 0 | 7/17.4/37 | 0.000 |
| polyglycine_n4 | 40 | 41 | 15 | 14 | 1 | 0 | 0 | 37/54.9/86 | 0.000 |
| polyyne_n6 | 14 | 7 | 15 | 14 | 0 | 1 | 0 | 9/22.6/32 | 0.002 |
| thiophene_n4 | 30 | 12 | 15 | 14 | 1 | 0 | 0 | 12/23.6/52 | 0.000 |
| nonacene | 60 | 16 | 15 | 15 | 0 | 0 | 3 | 14/29.5/68 | 137.899 |
| pentacene | 36 | 7 | 15 | 15 | 0 | 0 | 2 | 10/23.9/66 | 87.502 |
| PEG_n1 | 10 | 7 | 15 | 15 | 0 | 0 | 1 | 7/15.0/48 | 0.841 |
| PEG_n2 | 17 | 6 | 15 | 15 | 0 | 0 | 1 | 6/19.3/67 | 84.290 |
| anthracene | 24 | 7 | 15 | 15 | 0 | 0 | 1 | 9/22.3/68 | 135.734 |
| hexacene | 42 | 10 | 15 | 15 | 0 | 0 | 1 | 10/25.9/77 | 78.578 |
| polyalanine_n2 | 32 | 19 | 15 | 15 | 0 | 0 | 1 | 20/33.3/73 | 63.681 |
| polyglycine_n2 | 26 | 23 | 15 | 15 | 0 | 0 | 1 | 24/36.2/76 | 19.383 |
| PPE_n1 | 14 | 8 | 15 | 15 | 0 | 0 | 0 | 10/18.5/39 | 0.000 |
| diacetylene | 6 | 6 | 15 | 15 | 0 | 0 | 0 | 9/13.3/19 | 0.001 |
| polyethylene_n1 | 11 | 4 | 15 | 15 | 0 | 0 | 0 | 5/15.7/39 | 0.000 |
| polyyne_n3 | 8 | 7 | 15 | 15 | 0 | 0 | 0 | 8/14.8/21 | 0.000 |
| polyyne_n4 | 10 | 8 | 15 | 15 | 0 | 0 | 0 | 9/16.5/24 | 0.001 |
| polyyne_n5 | 12 | 7 | 15 | 15 | 0 | 0 | 0 | 8/17.5/27 | 0.002 |
| thiophene_n1 | 9 | 9 | 15 | 15 | 0 | 0 | 0 | 8/15.5/41 | 0.000 |
| thiophene_n2 | 16 | 9 | 15 | 15 | 0 | 0 | 0 | 11/23.9/57 | 0.029 |

## Representative errors

- **PPE_n3**: `RuntimeError: Cannot find f(x) > 0`
- **PPE_n5**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[1, 40], linear_l=[0, 2], linear_r=[0]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **naphthalene**: `TBLiteRuntimeError: SCF not converged in 250 cycles`
- **polyalanine_n1**: `LinAlgError: SVD did not converge`
- **polyalanine_n5**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[10, 43], linear_l=[], linear_r=[40, 42]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **polyglycine_n5**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[12, 15], linear_l=[], linear_r=[11, 16]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **polyglycine_n6**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[5, 11], linear_l=[34, 38], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **polyyne_n6**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[9, 6, 8], linear_l=[7, 10], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **polyyne_n7**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[5, 6], linear_l=[], linear_r=[7, 9]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **polyyne_n8**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[13, 14], linear_l=[12], linear_r=[15, 17]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **tetracene**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[1, 28], linear_l=[2, 16], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`

