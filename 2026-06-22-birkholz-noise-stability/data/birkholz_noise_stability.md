# Noise-stability sweep: birkholz / xtb

- Noise amplitudes (Angstrom RMS per Cartesian coord): 0.02, 0.05, 0.1, 0.2, 0.3
- Seeds per (molecule, amplitude): 3
- "Same minimum" window: |E - E_ref| <= 0.1 kcal/mol
- Wall time: 0.0 s

## Headline

- Noisy trials: 285; converged: 246 (86.3%), hit step ceiling: 16 (5.6%), errored before optimizing: 23 (8.1%)
- Converged trials landing in a different basin than the unperturbed run: 76 (26.67%)
- Molecules with any coordinate-build/other error: artemisinin, avobenzone, azadirachtin, bisphenol_a, cetirizine, codeine, diisobutyl_phthalate, estradiol, ochratoxin_a, tamoxifen, zn_edta
- Molecules with any step-ceiling non-convergence: artemisinin, azadirachtin, diisobutyl_phthalate, inosine_cation, maltose, ochratoxin_a, penicillin_v, raffinose, sphingomyelin
- Molecules with any different-basin outcome: artemisinin, avobenzone, azadirachtin, cetirizine, codeine, diisobutyl_phthalate, easc, estradiol, inosine_cation, maltose, mg_porphin, ochratoxin_a, penicillin_v, raffinose, sphingomyelin, tamoxifen, vitamin_c

## Stability vs noise amplitude (aggregate over all molecules)

| sigma (A) | trials | converged | ceiling | error | same basin | diff basin | max |dE| (kcal/mol) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 57 | 57 | 0 | 0 | 45 | 12 | 3.545 |
| 0.05 | 57 | 57 | 0 | 0 | 42 | 15 | 2.370 |
| 0.1 | 57 | 56 | 1 | 0 | 41 | 15 | 3.826 |
| 0.2 | 57 | 45 | 5 | 7 | 32 | 13 | 6.498 |
| 0.3 | 57 | 31 | 10 | 16 | 10 | 21 | 305.361 |

## Per-molecule summary

| Molecule | Atoms | Ref steps | Noisy trials | Converged | Ceiling | Error | Diff basin | Steps (min/mean/max) | Max |dE| (kcal/mol) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| diisobutyl_phthalate | 42 | 34 | 15 | 10 | 3 | 2 | 7 | 34/53.0/80 | 5.828 |
| azadirachtin | 95 | 60 | 15 | 10 | 1 | 4 | 5 | 57/69.9/89 | 2.520 |
| sphingomyelin | 84 | 83 | 15 | 11 | 4 | 0 | 5 | 78/84.5/97 | 3.826 |
| artemisinin | 42 | 27 | 15 | 11 | 1 | 3 | 2 | 25/47.1/94 | 1.686 |
| raffinose | 66 | 47 | 15 | 12 | 3 | 0 | 7 | 50/65.9/98 | 82.054 |
| ochratoxin_a | 45 | 56 | 15 | 12 | 1 | 2 | 4 | 25/63.5/97 | 26.109 |
| zn_edta | 33 | 40 | 15 | 12 | 0 | 3 | 0 | 32/42.5/62 | 0.002 |
| cetirizine | 52 | 51 | 15 | 13 | 0 | 2 | 9 | 32/53.9/95 | 1.467 |
| tamoxifen | 57 | 45 | 15 | 13 | 0 | 2 | 1 | 35/53.3/84 | 1.701 |
| bisphenol_a | 33 | 72 | 15 | 13 | 0 | 2 | 0 | 25/37.5/60 | 0.089 |
| inosine_cation | 31 | 58 | 15 | 14 | 1 | 0 | 14 | 32/46.7/79 | 28.932 |
| maltose | 45 | 77 | 15 | 14 | 1 | 0 | 11 | 33/62.9/94 | 4.830 |
| avobenzone | 45 | 48 | 15 | 14 | 0 | 1 | 1 | 37/53.8/78 | 62.976 |
| codeine | 43 | 36 | 15 | 14 | 0 | 1 | 1 | 36/45.5/61 | 50.099 |
| estradiol | 44 | 27 | 15 | 14 | 0 | 1 | 1 | 28/42.1/91 | 38.091 |
| penicillin_v | 42 | 52 | 15 | 14 | 1 | 0 | 1 | 43/57.3/99 | 2.988 |
| mg_porphin | 37 | 16 | 15 | 15 | 0 | 0 | 3 | 17/33.9/92 | 305.361 |
| easc | 26 | 38 | 15 | 15 | 0 | 0 | 2 | 33/52.7/92 | 0.164 |
| vitamin_c | 20 | 30 | 15 | 15 | 0 | 0 | 2 | 31/44.9/83 | 1.885 |

## Representative errors

- **artemisinin**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[9, 10], linear_l=[36, 37], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **artemisinin**: `LinAlgError: SVD did not converge`
- **azadirachtin**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[6, 54], linear_l=[4, 53], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **azadirachtin**: `TBLiteRuntimeError: SCF not converged in 250 cycles`
- **cetirizine**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[16, 22], linear_l=[13, 15], linear_r=[13]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **ochratoxin_a**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[28, 42], linear_l=[41, 44], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **tamoxifen**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[34, 37], linear_l=[6], linear_r=[6, 39]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **tamoxifen**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[49, 52], linear_l=[54, 56], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **zn_edta**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[1, 16], linear_l=[18, 26], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`

