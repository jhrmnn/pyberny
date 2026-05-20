# Internal-coordinate triggers along benchmark trajectories

Per-step diagnostics for every "bad case" molecule from `benchmark_diag/warnings.json`. Each row in the per-molecule mechanism table reports the warning class fired that step and the strongest geometric flag, classified per the rules at the bottom.


## Roll-up: warning steps and their geometric trigger

| set | molecule | class | converged | steps | warning step(s) | first geometric flag at/before warning | mechanism |
|---|---|---|---|---:|---|---|---|
| baker_shajan_2023 | allene | pinv@final | yes | 12 | 3,5,6,7,8... | angle C1-C0-C2=180.0 deg; 3-coord C1 sum=360.0 deg | linear-angle-dihedral |
| baker_shajan_2023 | disilyl_ether | pinv@final | yes | 7 | 5,6,7 | angle Si0-O2-Si1=171.7 deg | linear-angle-dihedral |
| birkholz_schlegel | estradiol | pinv@final (cross-validate) | yes | 11 | 6,7,8,9,10... | angle H37-C36-O40=171.5 deg; 3-coord C5 sum=360.0 deg; bond 36-40 ratio=1.54 | linear-angle-dihedral |
| birkholz_schlegel | raffinose | heavy back-xform + saddles | yes | 85 | 42,43,45,46,47 | - | unattributed |
| baker_shajan_2023 | caffeine | sustained neg-eig | ERR (FindrootError: ) | 75 | 27,31,33,37,40... | - | crashed |
| birkholz_schlegel | maltose | severe dq (converges right) | yes | 54 | 27,30,31 | - | unattributed |
| birkholz_schlegel | inosine_cation | severe dq (converges right) | yes | 47 | 11 | 3-coord C21 sum=360.0 deg | planar-center |
| birkholz_schlegel | ochratoxin_a | hit maxsteps | no | 110 | 41,44,47,51,54 | 3-coord C22 sum=360.0 deg | planar-center |
| birkholz_schlegel | bisphenol_a | hit maxsteps | no | 110 | 13,15 | 3-coord C16 sum=360.0 deg | planar-center |
| birkholz_schlegel | artemisinin | control (clean) | yes | 25 | - | - | clean |
| birkholz_schlegel | vitamin_c | control (clean) | yes | 29 | - | - | clean |
| baker_shajan_2023 | benzene | control (clean) | yes | 4 | - | - | clean |

## Per-molecule co-occurrence of warnings and geometric flags

Each row is one optimizer step. `mechanism` is the classification applied at that step. Only steps that fired *any* warning are tabulated. Steps with no warnings and no geometric flags are not shown.


### baker_shajan_2023/allene

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 3 | back-xform | Y | Y | . | Y | linear-angle-dihedral |
| 5 | pinv | Y | Y | . | Y | linear-angle-dihedral |
| 6 | pinv,back-xform | Y | Y | . | Y | linear-angle-dihedral |
| 7 | pinv,back-xform | Y | Y | . | Y | linear-angle-dihedral |
| 8 | pinv | Y | Y | . | Y | linear-angle-dihedral |
| 9 | pinv,back-xform | Y | Y | . | Y | linear-angle-dihedral |
| 10 | back-xform | Y | Y | . | Y | linear-angle-dihedral |
| 11 | pinv | Y | Y | . | Y | linear-angle-dihedral |
| 12 | pinv,back-xform | Y | Y | . | Y | linear-angle-dihedral |

### baker_shajan_2023/disilyl_ether

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 5 | back-xform | . | . | . | Y | unattributed |
| 6 | back-xform,severe-dq | Y | . | . | Y | linear-angle-dihedral |
| 7 | pinv | Y | . | . | Y | linear-angle-dihedral |

### birkholz_schlegel/estradiol

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 6 | back-xform | . | Y | . | Y | planar-center |
| 7 | back-xform,severe-dq | Y | Y | . | Y | linear-angle-dihedral |
| 8 | pinv | Y | Y | . | Y | linear-angle-dihedral |
| 9 | pinv | Y | Y | . | Y | linear-angle-dihedral |
| 10 | pinv | Y | Y | . | Y | linear-angle-dihedral |
| 11 | pinv | Y | Y | . | Y | linear-angle-dihedral |

### birkholz_schlegel/raffinose

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 42 | back-xform,neg-eig,severe-dq | . | . | . | Y | unattributed |
| 43 | back-xform,severe-dq | . | . | . | Y | unattributed |
| 45 | neg-eig | . | . | . | Y | unattributed |
| 46 | back-xform,neg-eig,severe-dq | . | . | . | Y | unattributed |
| 47 | neg-eig | . | . | . | Y | unattributed |

### baker_shajan_2023/caffeine (crashed at step 75)

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 27 | neg-eig | . | Y | . | Y | planar-center |
| 31 | neg-eig | . | Y | . | Y | planar-center |
| 33 | neg-eig | . | Y | . | Y | planar-center |
| 37 | neg-eig | . | Y | . | Y | planar-center |
| 40 | neg-eig | . | Y | . | Y | planar-center |
| 43 | neg-eig | . | Y | . | Y | planar-center |
| 45 | neg-eig | . | Y | . | Y | planar-center |
| 46 | neg-eig | . | Y | . | Y | planar-center |
| 50 | neg-eig | . | Y | . | Y | planar-center |
| 51 | neg-eig | . | Y | . | Y | planar-center |
| 55 | neg-eig | . | Y | . | Y | planar-center |
| 58 | neg-eig | . | Y | . | Y | planar-center |
| 59 | neg-eig | . | Y | . | Y | planar-center |
| 60 | neg-eig | . | Y | . | Y | planar-center |
| 62 | neg-eig | . | Y | . | Y | planar-center |
| 63 | neg-eig | . | Y | . | Y | planar-center |
| 66 | neg-eig | . | Y | . | Y | planar-center |
| 67 | neg-eig | . | Y | . | Y | planar-center |
| 70 | neg-eig | . | Y | . | Y | planar-center |

### birkholz_schlegel/maltose

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 27 | back-xform,neg-eig,severe-dq | . | . | . | Y | unattributed |
| 30 | back-xform,neg-eig,severe-dq | . | . | . | Y | unattributed |
| 31 | neg-eig | . | . | . | Y | unattributed |

### birkholz_schlegel/inosine_cation

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 11 | back-xform,severe-dq | . | Y | . | Y | planar-center |

### birkholz_schlegel/ochratoxin_a

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 41 | neg-eig | . | Y | . | Y | planar-center |
| 44 | neg-eig | . | Y | . | Y | planar-center |
| 47 | neg-eig | . | Y | . | Y | planar-center |
| 51 | neg-eig | . | Y | . | Y | planar-center |
| 54 | neg-eig | . | Y | . | Y | planar-center |

### birkholz_schlegel/bisphenol_a

| step | warnings | angle>=175 | planar-3coord | sp3-inverted | pinv gap < 3N-6 | mechanism |
|---:|---|---|---|---|---|---|
| 13 | neg-eig | . | Y | . | Y | planar-center |
| 15 | neg-eig | . | Y | . | Y | planar-center |

## Mechanism summary across molecules

| mechanism | warning steps with this mechanism | share |
|---|---:|---:|
| linear-angle-dihedral | 16 / 53 | 30% |
| planar-center | 28 / 53 | 53% |
| sp3-inversion | 0 / 53 | 0% |
| bond-stretch | 0 / 53 | 0% |
| unattributed | 9 / 53 | 17% |

## Classification rules

- `linear-angle-dihedral`: any `Angle` in `InternalCoords` >= 175.0 deg. Reproduces the estradiol H-C-O mechanism: 1/||a1|| in `Dihedral.eval` diverges as the containing angle approaches 180 deg.
- `planar-center`: any 3-bond-neighbour atom whose three neighbour-pair angles sum to >= 355.0 deg (sp2-like flattening). **Caveat**: this fires permanently for every aromatic-ring carbon (benzene, all six C atoms always sum to ~360 deg) and is therefore a *false positive* in the mechanism column for any molecule whose backbone contains an aromatic ring (caffeine, ochratoxin_a, bisphenol_a, inosine_cation, estradiol). Use the *transition* in `max_3coord_anglesum_deg` across the warning step (not its absolute value) when interpreting those rows; a step where a previously non-planar centre suddenly flattens is the genuine signal.
- `sp3-inversion`: any 4-bond-neighbour atom whose minimum out-of-plane angle (defined as the smallest of the 4 centre->m to plane-of-the-other-three angles) drops below 5.0 deg (sp3 inverted through planar).
- `bond-stretch`: any `Bond` whose length exceeds 1.5 x the sum of its atoms' covalent radii.
- `unattributed`: no geometric flag triggered at this step. These are the cases that *do not* match the estradiol-style singular-coordinate story and need a separate explanation (candidate: BFGS Hessian flips / RFO saddle-mode descent on a flat torsional manifold; not investigated here).


## Estradiol cross-validation

The bespoke estradiol driver perturbs the starting geometry to reach the basin where the pinv-at-final catastrophe fires. The new driver runs the *published* estradiol.xyz unperturbed, so the new run is expected to behave like a non-pathological trajectory: the pinv gap should stay at the "natural" 3N - 6 chemical-DOF position and never drop to index 0. Confirmation of the bespoke run's findings on the perturbed seeds (basin 4 and basin 3) is reproduced below:

| case | step | pinv gap idx | gap | top contributor |
|---|---:|---:|---:|---|
| basin 4 (high) | 7 | 125 | 5.98e+11 | Dihedral(37-36-40-41) |
| basin 4 (high) | 8 | 0 | 1.93e+03 | Dihedral(6-2-3-7) |
| basin 3 | 8 | 125 | 2.99e+10 | Bond(10-12) |
| basin 3 | 9 | 0 | 1.55e+05 | Dihedral(6-2-3-7) |
| basin 0 (deepest, sanity) | 24 | 125 | 9.66e+12 | Angle(3-2-6) |
| basin 0 (deepest, sanity) | 25 | 125 | 1.04e+13 | Angle(2-3-7) |

Re-running the new driver on `estradiol` (published .xyz) is a sanity check: it should converge in 11 steps with the pinv-at-final signature flagged at the final step, but with the *same* H37-C36-O40 H-C-O angle as the trigger (see the per-molecule co-occurrence table above).

