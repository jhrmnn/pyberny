# Noise-stability sweep: baker / xtb

- Noise amplitudes (Angstrom RMS per Cartesian coord): 0.02, 0.05, 0.1, 0.2, 0.3
- Seeds per (molecule, amplitude): 6
- "Same minimum" window: |E - E_ref| <= 0.1 kcal/mol
- Wall time: 2636.3 s

## Headline

- Noisy trials: 900; converged: 886 (98.4%), hit step ceiling: 6 (0.7%), errored before optimizing: 8 (0.9%)
- Converged trials landing in a different basin than the unperturbed run: 159 (17.67%)
- Molecules with any coordinate-build/other error: achtar10, dimethylpentane, histidine, menthone, neopentane, pterin
- Molecules with any step-ceiling non-convergence: caffeine, difluoronaphthalene_15, ethanol, histidine, hydroxybicyclopentane_2
- Molecules with any different-basin outcome: acanil01, achtar10, benzaldehyde, benzene, benzidine, caffeine, difluorobenzene_13, difuropyrazine, ethanol, furan, histidine, hydroxybicyclopentane_2, mesityl_oxide, methylamine, naphthalene, neopentane, trifluorobenzene_135

## Stability vs noise amplitude (aggregate over all molecules)

| sigma (A) | trials | converged | ceiling | error | same basin | diff basin | max |dE| (kcal/mol) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 180 | 180 | 0 | 0 | 156 | 24 | 6.241 |
| 0.05 | 180 | 180 | 0 | 0 | 152 | 28 | 6.241 |
| 0.1 | 180 | 180 | 0 | 0 | 150 | 30 | 6.241 |
| 0.2 | 180 | 178 | 0 | 2 | 146 | 32 | 6.241 |
| 0.3 | 180 | 168 | 6 | 6 | 123 | 45 | 115.234 |

## Per-molecule summary

| Molecule | Atoms | Ref steps | Noisy trials | Converged | Ceiling | Error | Diff basin | Steps (min/mean/max) | Max |dE| (kcal/mol) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| caffeine | 24 | 7 | 30 | 28 | 2 | 0 | 22 | 11/41.9/74 | 90.540 |
| histidine | 20 | 19 | 30 | 28 | 1 | 1 | 2 | 20/31.0/81 | 97.310 |
| achtar10 | 16 | 30 | 30 | 28 | 0 | 2 | 1 | 22/32.5/65 | 1.069 |
| dimethylpentane | 23 | 12 | 30 | 28 | 0 | 2 | 0 | 12/28.5/73 | 0.000 |
| ethanol | 9 | 5 | 30 | 29 | 1 | 0 | 2 | 6/14.7/80 | 24.909 |
| hydroxybicyclopentane_2 | 14 | 9 | 30 | 29 | 1 | 0 | 1 | 8/15.9/40 | 43.784 |
| neopentane | 17 | 3 | 30 | 29 | 0 | 1 | 1 | 5/19.0/66 | 3.827 |
| difluoronaphthalene_15 | 18 | 4 | 30 | 29 | 1 | 0 | 0 | 10/18.2/56 | 0.002 |
| menthone | 29 | 13 | 30 | 29 | 0 | 1 | 0 | 13/26.0/89 | 0.000 |
| pterin | 17 | 7 | 30 | 29 | 0 | 1 | 0 | 15/26.6/52 | 0.031 |
| benzidine | 26 | 8 | 30 | 30 | 0 | 0 | 30 | 14/25.5/53 | 115.234 |
| mesityl_oxide | 17 | 6 | 30 | 30 | 0 | 0 | 30 | 25/40.1/83 | 1.305 |
| methylamine | 7 | 5 | 30 | 30 | 0 | 0 | 30 | 8/13.8/25 | 6.241 |
| acanil01 | 19 | 6 | 30 | 30 | 0 | 0 | 28 | 12/32.7/76 | 0.752 |
| benzene | 12 | 3 | 30 | 30 | 0 | 0 | 2 | 9/17.1/42 | 93.344 |
| difluorobenzene_13 | 12 | 4 | 30 | 30 | 0 | 0 | 2 | 9/17.4/58 | 87.694 |
| difuropyrazine | 16 | 7 | 30 | 30 | 0 | 0 | 2 | 12/22.2/46 | 77.428 |
| furan | 9 | 5 | 30 | 30 | 0 | 0 | 2 | 10/16.9/48 | 48.219 |
| naphthalene | 18 | 4 | 30 | 30 | 0 | 0 | 2 | 9/20.2/86 | 56.874 |
| benzaldehyde | 14 | 5 | 30 | 30 | 0 | 0 | 1 | 8/18.3/40 | 77.676 |
| trifluorobenzene_135 | 12 | 4 | 30 | 30 | 0 | 0 | 1 | 7/16.0/41 | 87.749 |
| acetone | 10 | 5 | 30 | 30 | 0 | 0 | 0 | 7/23.5/80 | 0.024 |
| acetylene | 4 | 4 | 30 | 30 | 0 | 0 | 0 | 5/10.7/17 | 0.000 |
| allene | 7 | 5 | 30 | 30 | 0 | 0 | 0 | 9/15.1/24 | 0.000 |
| ammonia | 4 | 4 | 30 | 30 | 0 | 0 | 0 | 4/6.7/13 | 0.000 |
| disilyl_ether | 9 | 16 | 30 | 30 | 0 | 0 | 0 | 16/30.2/47 | 0.087 |
| ethane | 8 | 4 | 30 | 30 | 0 | 0 | 0 | 4/11.8/35 | 0.000 |
| hydroxysulfane | 4 | 7 | 30 | 30 | 0 | 0 | 0 | 6/8.9/16 | 0.000 |
| trisilacyclohexane_135 | 18 | 7 | 30 | 30 | 0 | 0 | 0 | 7/19.3/50 | 0.001 |
| water | 3 | 4 | 30 | 30 | 0 | 0 | 0 | 4/6.4/13 | 0.000 |

## Representative errors

- **achtar10**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[1, 6], linear_l=[5, 13], linear_r=[]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **achtar10**: `LinAlgError: SVD did not converge`
- **menthone**: `CoordinateError: Cannot build dihedrals through a near-linear chain with a branching terminus: the linear-chain extension supports at most one near-linear neighbour per end (center=[0, 6], linear_l=[], linear_r=[5, 12]). This typically arises in fully linear-rich systems such as long poly(phenylene-ethynylene) chains.`
- **pterin**: `TBLiteRuntimeError: SCF not converged in 250 cycles`

