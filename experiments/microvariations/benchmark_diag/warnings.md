# Benchmark trajectory warning sweep

For every molecule in the two MOPAC benchmark sets, `Berny(geom, maxsteps=110)` + `MopacSolver(charge, mult)` ran with INFO-level logging captured to `<set>/<molecule>.log`. The columns below summarise the warning lines that fired during the trajectory. `pinv@final` is the dangerous case discovered on estradiol: a `Pseudoinverse gap of only:` warning at the same step that declared convergence means pyberny halted on a rank-deficient internal-gradient projection.


## birkholz_schlegel

| molecule | ref | steps | conv | pinv | pinv@final | back-xform | neg-eig steps | severe dq | maxsteps |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| artemisinin | 25 | 25 | yes | 0 | - | 0 | 0 | 0 | - |
| avobenzone | 94 | 90 | yes | 0 | - | 0 | 4 | 0 | - |
| azadirachtin | 60 | 61 | yes | 0 | - | 0 | 0 | 0 | - |
| bisphenol_a | - | 110 | no | 0 | - | 0 | 2 | 0 | yes |
| cetirizine | 58 | 57 | yes | 0 | - | 0 | 0 | 0 | - |
| codeine | 31 | 31 | yes | 0 | - | 0 | 2 | 0 | - |
| diisobutyl_phthalate | 38 | 38 | yes | 0 | - | 0 | 0 | 0 | - |
| easc | 53 | 53 | yes | 0 | - | 0 | 0 | 0 | - |
| estradiol | 11 | 11 | yes | 4 | YES | 2 | 0 | 1 | - |
| inosine_cation | 47 | 47 | yes | 0 | - | 1 | 0 | 1 | - |
| maltose | 54 | 54 | yes | 0 | - | 2 | 3 | 2 | - |
| mg_porphin | 34 | 34 | yes | 0 | - | 0 | 0 | 0 | - |
| ochratoxin_a | - | 110 | no | 0 | - | 0 | 5 | 0 | yes |
| penicillin_v | 54 | 54 | yes | 0 | - | 0 | 0 | 0 | - |
| raffinose | - | 110 | no | 0 | - | 6 | 5 | 6 | yes |
| sphingomyelin | 95 | 87 | yes | 0 | - | 0 | 1 | 0 | - |
| tamoxifen | 70 | 72 | yes | 0 | - | 0 | 0 | 0 | - |
| vitamin_c | 29 | 29 | yes | 0 | - | 0 | 0 | 0 | - |
| zn_edta | 99 | 99 | yes | 0 | - | 0 | 0 | 0 | - |

## baker_shajan_2023

| molecule | ref | steps | conv | pinv | pinv@final | back-xform | neg-eig steps | severe dq | maxsteps |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| acanil01 | 43 | 43 | yes | 0 | - | 0 | 2 | 0 | - |
| acetone | 7 | 7 | yes | 0 | - | 0 | 0 | 0 | - |
| acetylene | 9 | 9 | yes | 0 | - | 0 | 0 | 0 | - |
| achtar10 | 9 | 9 | yes | 0 | - | 0 | 0 | 0 | - |
| allene | 12 | 12 | yes | 7 | YES | 6 | 0 | 0 | - |
| ammonia | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| benzaldehyde | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| benzene | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| benzidine | 24 | 24 | yes | 0 | - | 0 | 2 | 0 | - |
| caffeine | - | 110 | no | 0 | - | 0 | 43 | 0 | yes |
| difluorobenzene_13 | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| difluoronaphthalene_15 | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| difuropyrazine | 7 | 7 | yes | 0 | - | 0 | 0 | 0 | - |
| dimethylpentane | 10 | 10 | yes | 0 | - | 0 | 0 | 0 | - |
| disilyl_ether | 7 | 7 | yes | 1 | YES | 2 | 0 | 1 | - |
| ethane | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| ethanol | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| furan | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| histidine | 19 | 19 | yes | 0 | - | 0 | 0 | 0 | - |
| hydroxybicyclopentane_2 | 9 | 9 | yes | 0 | - | 0 | 0 | 0 | - |
| hydroxysulfane | 6 | 6 | yes | 0 | - | 0 | 0 | 0 | - |
| menthone | 16 | 16 | yes | 0 | - | 0 | 0 | 0 | - |
| mesityl_oxide | 8 | 8 | yes | 0 | - | 0 | 0 | 0 | - |
| methylamine | 18 | 18 | yes | 0 | - | 0 | 4 | 0 | - |
| naphthalene | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |
| neopentane | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| pterin | 8 | 8 | yes | 0 | - | 0 | 0 | 0 | - |
| trifluorobenzene_135 | 4 | 4 | yes | 0 | - | 0 | 0 | 0 | - |
| trisilacyclohexane_135 | 7 | 7 | yes | 0 | - | 0 | 0 | 0 | - |
| water | 5 | 5 | yes | 0 | - | 0 | 0 | 0 | - |

## Findings

### Pseudoinverse warning at the convergence-declaring step (estradiol-style)

These are the cases where `converged=True` was returned even though `Math.pinv` had silently zeroed a singular direction at the final step:

- **birkholz/estradiol** - final-step pinv gap `2.60e+03`
- **baker/allene** - final-step pinv gap `3.00e+07`
- **baker/disilyl_ether** - final-step pinv gap `9.40e+05`

### Heavy back-transform struggle (>=3 warnings)

On estradiol the precursor to the pinv catastrophe was three consecutive `Transformation did not converge in 20 iterations` lines. Molecules below show the same precursor signature:

- birkholz/raffinose: 6 back-transform warning(s)
- baker/allene: 6 back-transform warning(s)

### Severe back-transform step blow-up (RMS(dq) >= 0.05)

These steps reported a back-transform with internal-coord RMS displacement at least an order of magnitude above what a well-behaved Cartesian<->internal mapping produces. They are often saddle-pass attempts on rigid macrocycles or near-linear angles; if they coincide with `pinv@final = YES` they are the mechanism by which the pinv pathology fires.

- birkholz/estradiol: 7 (dq=0.13)
- birkholz/inosine_cation: 11 (dq=0.091)
- birkholz/maltose: 27 (dq=0.082),30 (dq=0.09)
- birkholz/raffinose: 42 (dq=0.089),43 (dq=0.054),46 (dq=0.22),47 (dq=0.12),48 (dq=0.074),49 (dq=0.074)
- baker/disilyl_ether: 6 (dq=1.1)

### Negative-eigenvalue events (sphere-minimization saddle passes)

Healthy minimum-finding trajectories report all-positive BFGS Hessian eigenvalues at every step. Negative-eigenvalue events happen when the BFGS update produces a spurious unstable mode (or when the geometry really is near a saddle). Pyberny then switches to RFO sphere-minimization to descend along that unstable direction. Occasional events are normal; a sustained count is a sign of a confusing PES region:

- birkholz/avobenzone: 4 event(s) at step(s) 14,17,20,49
- birkholz/bisphenol_a: 2 event(s) at step(s) 13,15
- birkholz/codeine: 2 event(s) at step(s) 15,18
- birkholz/maltose: 3 event(s) at step(s) 27,30,31
- birkholz/ochratoxin_a: 5 event(s) at step(s) 41,44,47,51,54
- birkholz/raffinose: 5 event(s) at step(s) 42,45,46,47,48
- birkholz/sphingomyelin: 1 event(s) at step(s) 11
- baker/acanil01: 2 event(s) at step(s) 12,13
- baker/benzidine: 2 event(s) at step(s) 12,13
- baker/caffeine: 43 event(s) at step(s) 27,30,33,35,36,38,39,41,45,47,49,51,52,53,56,57,58,59,62,63,64,65,69,71,74,75,77,78,79,80,81,83,86,87,89,93,95,98,99,101,105,106,107
- baker/methylamine: 4 event(s) at step(s) 5,7,8,9

### Hit maxsteps ceiling

- birkholz/bisphenol_a
- birkholz/ochratoxin_a
- birkholz/raffinose
- baker/caffeine

