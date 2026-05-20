# Internal-coordinate health check on problematic benchmark cases

For every molecule that `benchmark_trajectory_check.py` flagged, this script re-runs the same `Berny+MopacSolver` trajectory but captures the Wilson B-matrix at every step. The tables below distill **which internal coordinate(s) fire** - i.e. which Bond/Angle/Dihedral becomes the dominant contributor to the singular direction that `Math.pinv` would (and on three molecules, does) silently truncate.

Atom indices are zero-based, prefixed with the element symbol of that atom (e.g. `H37` is hydrogen #37 in the xyz file). The "top contrib" coordinate is the one whose weight in the eigenvector at the pinv gap is largest at the final step where the warning fires.


## birkholz/estradiol

*pinv@final: H-C-O angle near-linear (known smoking gun)*

Steps: 11, converged: True, wall: 3.5s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 125 | 1.18e+13 | 2.23e+01 | 1.00e-30 | 0 | 0 | 22 | 6 |
| 2 | 125 | 1.18e+13 | 2.07e+01 | 1.00e-30 | 0 | 0 | 24 | 6 |
| 3 | 125 | 4.75e+12 | 2.26e+01 | 1.00e-30 | 0 | 0 | 25 | 6 |
| 4 | 125 | 7.30e+12 | 3.08e+01 | 1.00e-30 | 0 | 0 | 30 | 6 |
| 5 | 125 | 4.54e+12 | 3.95e+01 | 1.00e-30 | 0 | 0 | 34 | 6 |
| 6 | 125 | 5.19e+12 | 7.93e+01 | 1.00e-30 | 0 | 0 | 35 | 6 |
| 7 | 125 | 1.12e+12 | 3.37e+02 | 1.00e-30 | 1 | 2 | 41 | 6 |
| 8 | 0 | 2.74e+03 | 4.38e+04 | 1.00e-30 | 1 | 2 | 40 | 6 |
| 9 | 0 | 2.74e+03 | 4.38e+04 | 1.00e-30 | 1 | 2 | 40 | 6 |
| 10 | 0 | 2.62e+03 | 4.19e+04 | 1.00e-30 | 1 | 2 | 40 | 6 |
| 11 | 0 | 2.61e+03 | 4.19e+04 | 1.00e-30 | 1 | 2 | 40 | 6 |

### Firing internals (steps where the pinv warning would fire)

These are the steps where `Math.pinv`'s gap value falls below `1e8` (its log threshold) - covering both the "spurious gap at low index" pathology (the gradient projection kills a far-away real DOF) and the "natural-position small gap" pathology (the smallest real singular value is being truncated). The "top contributors" are the internals whose B-row dominates the truncated eigenvector at this step; the "rogue B-row" list is the coords whose gradient magnitude is most inflated.


**Step 8** - gap at index 0 (value `2.74e+03`), `sv_max = 4.38e+04`, `sv_just_after_gap = 1.60e+01`
- Near-linear angles: Angle(H37-C36-O40)=179.66°
- Top contributors to the truncated direction:
  - `Dihedral(C6-C2-C3-C7)` (idx 163, weight 0.091)
  - `Dihedral(C1-C2-C3-C4)` (idx 160, weight 0.085)
  - `Dihedral(C2-C3-C4-C5)` (idx 170, weight 0.066)
  - `Dihedral(C0-C1-C2-C3)` (idx 154, weight 0.066)
  - `Dihedral(C2-C3-C7-C8)` (idx 174, weight 0.054)
- Largest B-row norms (the rogue gradients):
  - `Dihedral(H37-C36-O40-C34)` (idx 312, |B_row| = 1.48e+02)
  - `Dihedral(H37-C36-O40-H41)` (idx 313, |B_row| = 1.48e+02)
  - `Dihedral(H16-C0-C5-H18)` (idx 153, |B_row| = 1.63e+00)
  - `Dihedral(H17-C4-C5-H18)` (idx 183, |B_row| = 1.60e+00)
  - `Dihedral(H16-C0-C1-O42)` (idx 149, |B_row| = 1.57e+00)

**Step 9** - gap at index 0 (value `2.74e+03`), `sv_max = 4.38e+04`, `sv_just_after_gap = 1.60e+01`
- Near-linear angles: Angle(H37-C36-O40)=179.66°
- Top contributors to the truncated direction:
  - `Dihedral(C6-C2-C3-C7)` (idx 163, weight 0.091)
  - `Dihedral(C1-C2-C3-C4)` (idx 160, weight 0.085)
  - `Dihedral(C2-C3-C4-C5)` (idx 170, weight 0.066)
  - `Dihedral(C0-C1-C2-C3)` (idx 154, weight 0.066)
  - `Dihedral(C2-C3-C7-C8)` (idx 174, weight 0.054)
- Largest B-row norms (the rogue gradients):
  - `Dihedral(H37-C36-O40-C34)` (idx 312, |B_row| = 1.48e+02)
  - `Dihedral(H37-C36-O40-H41)` (idx 313, |B_row| = 1.48e+02)
  - `Dihedral(H16-C0-C5-H18)` (idx 153, |B_row| = 1.63e+00)
  - `Dihedral(H17-C4-C5-H18)` (idx 183, |B_row| = 1.60e+00)
  - `Dihedral(H16-C0-C1-O42)` (idx 149, |B_row| = 1.57e+00)

**Step 10** - gap at index 0 (value `2.62e+03`), `sv_max = 4.19e+04`, `sv_just_after_gap = 1.60e+01`
- Near-linear angles: Angle(H37-C36-O40)=179.65°
- Top contributors to the truncated direction:
  - `Dihedral(C6-C2-C3-C7)` (idx 163, weight 0.091)
  - `Dihedral(C1-C2-C3-C4)` (idx 160, weight 0.085)
  - `Dihedral(C2-C3-C4-C5)` (idx 170, weight 0.066)
  - `Dihedral(C0-C1-C2-C3)` (idx 154, weight 0.066)
  - `Dihedral(C2-C3-C7-C8)` (idx 174, weight 0.054)
- Largest B-row norms (the rogue gradients):
  - `Dihedral(H37-C36-O40-C34)` (idx 312, |B_row| = 1.45e+02)
  - `Dihedral(H37-C36-O40-H41)` (idx 313, |B_row| = 1.45e+02)
  - `Dihedral(H16-C0-C5-H18)` (idx 153, |B_row| = 1.63e+00)
  - `Dihedral(H17-C4-C5-H18)` (idx 183, |B_row| = 1.60e+00)
  - `Dihedral(H16-C0-C1-O42)` (idx 149, |B_row| = 1.57e+00)

**Step 11** - gap at index 0 (value `2.61e+03`), `sv_max = 4.19e+04`, `sv_just_after_gap = 1.60e+01`
- Near-linear angles: Angle(H37-C36-O40)=179.65°
- Top contributors to the truncated direction:
  - `Dihedral(C6-C2-C3-C7)` (idx 163, weight 0.091)
  - `Dihedral(C1-C2-C3-C4)` (idx 160, weight 0.085)
  - `Dihedral(C2-C3-C4-C5)` (idx 170, weight 0.066)
  - `Dihedral(C0-C1-C2-C3)` (idx 154, weight 0.066)
  - `Dihedral(C2-C3-C7-C8)` (idx 174, weight 0.054)
- Largest B-row norms (the rogue gradients):
  - `Dihedral(H37-C36-O40-C34)` (idx 312, |B_row| = 1.45e+02)
  - `Dihedral(H37-C36-O40-H41)` (idx 313, |B_row| = 1.45e+02)
  - `Dihedral(H16-C0-C5-H18)` (idx 153, |B_row| = 1.63e+00)
  - `Dihedral(H17-C4-C5-H18)` (idx 183, |B_row| = 1.60e+00)
  - `Dihedral(H16-C0-C1-O42)` (idx 149, |B_row| = 1.57e+00)

### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 22 planar dihedral(s), 6 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 6 | backx | 0 | 16 | 0 | YES | NEW Dihedral(C0-C1-C2-C3) = 2.7° (cis) |
| 7 | backx, severe-dq | 1 | 23 | 0 | YES | Angle(H37-C36-O40) = 176.0° |
| 8 | pinv | 1 | 22 | 0 | YES | Angle(H37-C36-O40) = 179.7° |
| 9 | pinv | 1 | 22 | 0 | YES | Angle(H37-C36-O40) = 179.7° |
| 10 | pinv | 1 | 22 | 0 | YES | Angle(H37-C36-O40) = 179.7° |
| 11 | pinv | 1 | 22 | 0 | YES | Angle(H37-C36-O40) = 179.7° |

**Verdict:** every one of the 6 warning step(s) coincides with at least one planar/linear event new vs step 1 (or a near-linear angle).


## baker/allene

*pinv@final: C=C=C linear backbone (symmetry-imposed)*

Steps: 12, converged: True, wall: 0.3s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 4.61e+14 | 3.99e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 2 | 13 | 8.22e+09 | 4.01e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 3 | 13 | 6.01e+10 | 4.00e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 4 | 13 | 2.07e+11 | 4.01e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 5 | 13 | 2.01e+07 | 4.02e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 6 | 13 | 1.72e+05 | 4.02e+00 | 2.55e-17 | 1 | 0 | 0 | 2 |
| 7 | 13 | 1.60e+03 | 4.02e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 8 | 13 | 6.35e+05 | 4.02e+00 | 1.44e-17 | 1 | 0 | 0 | 2 |
| 9 | 13 | 4.12e+04 | 4.02e+00 | 3.91e-17 | 1 | 0 | 0 | 2 |
| 10 | 13 | 1.05e+09 | 4.02e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 11 | 13 | 3.12e+06 | 4.02e+00 | 1.00e-30 | 1 | 0 | 0 | 2 |
| 12 | 13 | 3.04e+07 | 4.02e+00 | 4.70e-16 | 1 | 0 | 0 | 2 |

### Firing internals (steps where the pinv warning would fire)

These are the steps where `Math.pinv`'s gap value falls below `1e8` (its log threshold) - covering both the "spurious gap at low index" pathology (the gradient projection kills a far-away real DOF) and the "natural-position small gap" pathology (the smallest real singular value is being truncated). The "top contributors" are the internals whose B-row dominates the truncated eigenvector at this step; the "rogue B-row" list is the coords whose gradient magnitude is most inflated.


**Step 5** - gap at index 13 (value `2.01e+07`), `sv_max = 4.02e+00`, `sv_just_after_gap = 1.01e-08`
- Near-linear angles: Angle(C1-C0-C2)=179.98°
- Top contributors to the truncated direction:
  - `Angle(C0-C2-H4)` (idx 11, weight 0.167)
  - `Angle(H3-C2-H4)` (idx 12, weight 0.167)
  - `Angle(C0-C2-H3)` (idx 10, weight 0.167)
  - `Angle(C0-C1-H6)` (idx 8, weight 0.166)
  - `Angle(H5-C1-H6)` (idx 9, weight 0.166)
- Largest B-row norms (the rogue gradients):
  - `Bond(C0-C2)` (idx 1, |B_row| = 1.41e+00)
  - `Bond(C0-C1)` (idx 0, |B_row| = 1.41e+00)
  - `Bond(C1-H6)` (idx 3, |B_row| = 1.41e+00)
  - `Bond(C2-H3)` (idx 4, |B_row| = 1.41e+00)
  - `Bond(C1-H5)` (idx 2, |B_row| = 1.41e+00)

**Step 6** - gap at index 13 (value `1.72e+05`), `sv_max = 4.02e+00`, `sv_just_after_gap = 1.17e-06`
- Near-linear angles: Angle(C1-C0-C2)=179.98°
- Top contributors to the truncated direction:
  - `Angle(C0-C1-H5)` (idx 7, weight 0.168)
  - `Angle(H5-C1-H6)` (idx 9, weight 0.168)
  - `Angle(C0-C1-H6)` (idx 8, weight 0.167)
  - `Angle(C0-C2-H3)` (idx 10, weight 0.166)
  - `Angle(H3-C2-H4)` (idx 12, weight 0.165)
- Largest B-row norms (the rogue gradients):
  - `Bond(C0-C1)` (idx 0, |B_row| = 1.41e+00)
  - `Bond(C1-H5)` (idx 2, |B_row| = 1.41e+00)
  - `Bond(C2-H4)` (idx 5, |B_row| = 1.41e+00)
  - `Bond(C0-C2)` (idx 1, |B_row| = 1.41e+00)
  - `Bond(C1-H6)` (idx 3, |B_row| = 1.41e+00)

**Step 7** - gap at index 13 (value `1.60e+03`), `sv_max = 4.02e+00`, `sv_just_after_gap = 1.26e-04`
- Near-linear angles: Angle(C1-C0-C2)=179.98°
- Top contributors to the truncated direction:
  - `Angle(C0-C2-H4)` (idx 11, weight 0.249)
  - `Angle(H3-C2-H4)` (idx 12, weight 0.245)
  - `Angle(C0-C2-H3)` (idx 10, weight 0.239)
  - `Angle(C0-C1-H6)` (idx 8, weight 0.093)
  - `Angle(H5-C1-H6)` (idx 9, weight 0.089)
- Largest B-row norms (the rogue gradients):
  - `Bond(C0-C1)` (idx 0, |B_row| = 1.41e+00)
  - `Bond(C0-C2)` (idx 1, |B_row| = 1.41e+00)
  - `Bond(C1-H6)` (idx 3, |B_row| = 1.41e+00)
  - `Bond(C2-H3)` (idx 4, |B_row| = 1.41e+00)
  - `Bond(C1-H5)` (idx 2, |B_row| = 1.41e+00)

**Step 8** - gap at index 13 (value `6.35e+05`), `sv_max = 4.02e+00`, `sv_just_after_gap = 3.18e-07`
- Near-linear angles: Angle(C1-C0-C2)=179.68°
- Top contributors to the truncated direction:
  - `Angle(C0-C1-H5)` (idx 7, weight 0.244)
  - `Angle(H5-C1-H6)` (idx 9, weight 0.243)
  - `Angle(C0-C1-H6)` (idx 8, weight 0.243)
  - `Angle(C0-C2-H4)` (idx 11, weight 0.090)
  - `Angle(H3-C2-H4)` (idx 12, weight 0.090)
- Largest B-row norms (the rogue gradients):
  - `Bond(C0-C1)` (idx 0, |B_row| = 1.41e+00)
  - `Bond(C1-H5)` (idx 2, |B_row| = 1.41e+00)
  - `Bond(C1-H6)` (idx 3, |B_row| = 1.41e+00)
  - `Bond(C2-H3)` (idx 4, |B_row| = 1.41e+00)
  - `Bond(C2-H4)` (idx 5, |B_row| = 1.41e+00)

**Step 9** - gap at index 13 (value `4.12e+04`), `sv_max = 4.02e+00`, `sv_just_after_gap = 4.90e-06`
- Near-linear angles: Angle(C1-C0-C2)=179.82°
- Top contributors to the truncated direction:
  - `Angle(C0-C1-H5)` (idx 7, weight 0.330)
  - `Angle(H5-C1-H6)` (idx 9, weight 0.330)
  - `Angle(C0-C1-H6)` (idx 8, weight 0.330)
  - `Angle(C0-C2-H4)` (idx 11, weight 0.003)
  - `Angle(H3-C2-H4)` (idx 12, weight 0.003)
- Largest B-row norms (the rogue gradients):
  - `Bond(C0-C2)` (idx 1, |B_row| = 1.41e+00)
  - `Bond(C1-H5)` (idx 2, |B_row| = 1.41e+00)
  - `Bond(C1-H6)` (idx 3, |B_row| = 1.41e+00)
  - `Bond(C2-H3)` (idx 4, |B_row| = 1.41e+00)
  - `Bond(C2-H4)` (idx 5, |B_row| = 1.41e+00)

**Step 11** - gap at index 13 (value `3.12e+06`), `sv_max = 4.02e+00`, `sv_just_after_gap = 6.48e-08`
- Near-linear angles: Angle(C1-C0-C2)=179.97°
- Top contributors to the truncated direction:
  - `Angle(C0-C1-H6)` (idx 8, weight 0.190)
  - `Angle(H5-C1-H6)` (idx 9, weight 0.190)
  - `Angle(C0-C1-H5)` (idx 7, weight 0.190)
  - `Angle(C0-C2-H3)` (idx 10, weight 0.143)
  - `Angle(H3-C2-H4)` (idx 12, weight 0.143)
- Largest B-row norms (the rogue gradients):
  - `Bond(C0-C1)` (idx 0, |B_row| = 1.41e+00)
  - `Bond(C0-C2)` (idx 1, |B_row| = 1.41e+00)
  - `Bond(C1-H5)` (idx 2, |B_row| = 1.41e+00)
  - `Bond(C1-H6)` (idx 3, |B_row| = 1.41e+00)
  - `Bond(C2-H3)` (idx 4, |B_row| = 1.41e+00)

**Step 12** - gap at index 13 (value `3.04e+07`), `sv_max = 4.02e+00`, `sv_just_after_gap = 6.65e-09`
- Near-linear angles: Angle(C1-C0-C2)=179.95°
- Top contributors to the truncated direction:
  - `Angle(C0-C2-H3)` (idx 10, weight 0.333)
  - `Angle(H3-C2-H4)` (idx 12, weight 0.333)
  - `Angle(C0-C2-H4)` (idx 11, weight 0.333)
  - `Angle(C0-C1-H5)` (idx 7, weight 0.000)
  - `Angle(H5-C1-H6)` (idx 9, weight 0.000)
- Largest B-row norms (the rogue gradients):
  - `Bond(C1-H5)` (idx 2, |B_row| = 1.41e+00)
  - `Bond(C2-H4)` (idx 5, |B_row| = 1.41e+00)
  - `Bond(C0-C2)` (idx 1, |B_row| = 1.41e+00)
  - `Bond(C0-C1)` (idx 0, |B_row| = 1.41e+00)
  - `Bond(C2-H3)` (idx 4, |B_row| = 1.41e+00)

### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 0 planar dihedral(s), 2 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 3 | backx | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 180.0° |
| 5 | pinv | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 180.0° |
| 6 | backx, pinv | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 180.0° |
| 7 | backx, pinv | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 180.0° |
| 8 | pinv | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 179.7° |
| 9 | backx, pinv | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 179.8° |
| 10 | backx | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 179.8° |
| 11 | pinv | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 180.0° |
| 12 | backx, pinv | 1 | 0 | 0 | YES | Angle(C1-C0-C2) = 180.0° |

**Verdict:** every one of the 9 warning step(s) coincides with at least one planar/linear event new vs step 1 (or a near-linear angle).


## baker/disilyl_ether

*pinv@final: Si-O-Si linearization*

Steps: 7, converged: True, wall: 0.6s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 5.66e+13 | 1.41e+01 | 1.00e-30 | 0 | 0 | 2 | 0 |
| 2 | 20 | 8.66e+13 | 1.81e+01 | 1.00e-30 | 0 | 0 | 2 | 0 |
| 3 | 20 | 3.41e+13 | 3.50e+01 | 1.00e-30 | 0 | 0 | 2 | 0 |
| 4 | 20 | 7.35e+13 | 5.86e+01 | 1.00e-30 | 0 | 0 | 2 | 0 |
| 5 | 20 | 1.81e+13 | 1.89e+02 | 1.00e-30 | 0 | 0 | 2 | 0 |
| 6 | 20 | 6.31e+11 | 1.77e+03 | 1.00e-30 | 1 | 6 | 2 | 0 |
| 7 | 0 | 9.40e+05 | 3.36e+06 | 1.00e-30 | 1 | 6 | 2 | 0 |

### Firing internals (steps where the pinv warning would fire)

These are the steps where `Math.pinv`'s gap value falls below `1e8` (its log threshold) - covering both the "spurious gap at low index" pathology (the gradient projection kills a far-away real DOF) and the "natural-position small gap" pathology (the smallest real singular value is being truncated). The "top contributors" are the internals whose B-row dominates the truncated eigenvector at this step; the "rogue B-row" list is the coords whose gradient magnitude is most inflated.


**Step 7** - gap at index 0 (value `9.40e+05`), `sv_max = 3.36e+06`, `sv_just_after_gap = 3.57e+00`
- Near-linear angles: Angle(Si0-O2-Si1)=179.94°
- Top contributors to the truncated direction:
  - `Bond(Si1-O2)` (idx 4, weight 0.384)
  - `Bond(Si0-O2)` (idx 0, weight 0.384)
  - `Bond(Si1-H6)` (idx 5, weight 0.018)
  - `Bond(Si0-H3)` (idx 1, weight 0.018)
  - `Bond(Si1-H7)` (idx 6, weight 0.018)
- Largest B-row norms (the rogue gradients):
  - `Dihedral(H6-Si1-O2-Si0)` (idx 24, |B_row| = 7.48e+02)
  - `Dihedral(H3-Si0-O2-Si1)` (idx 21, |B_row| = 7.48e+02)
  - `Dihedral(H5-Si0-O2-Si1)` (idx 23, |B_row| = 7.48e+02)
  - `Dihedral(H8-Si1-O2-Si0)` (idx 26, |B_row| = 7.48e+02)
  - `Dihedral(H4-Si0-O2-Si1)` (idx 22, |B_row| = 7.48e+02)

### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 2 planar dihedral(s), 0 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 5 | backx | 0 | 0 | 0 | no | Dihedral(H3-Si0-O2-Si1) = -180.0° (aromatic/baseline) |
| 6 | backx, severe-dq | 1 | 0 | 0 | YES | Angle(Si0-O2-Si1) = 177.3° |
| 7 | pinv | 1 | 0 | 0 | YES | Angle(Si0-O2-Si1) = 179.9° |

**Verdict:** 2 of 3 warning step(s) coincide with a new planar/linear event; the remainder fired on a baseline geometry.


## birkholz/inosine_cation

*severe back-xform RMS(dq) at step 11*

Steps: 15, converged: False, wall: 3.5s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86 | 7.08e+12 | 1.31e+01 | 1.00e-30 | 0 | 0 | 44 | 6 |
| 2 | 86 | 6.94e+12 | 1.35e+01 | 1.00e-30 | 0 | 0 | 43 | 6 |
| 3 | 86 | 7.72e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 | 43 | 6 |
| 4 | 86 | 7.13e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 | 40 | 6 |
| 5 | 86 | 6.52e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 | 37 | 7 |
| 6 | 86 | 5.69e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 | 41 | 7 |
| 7 | 86 | 7.54e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 | 38 | 7 |
| 8 | 86 | 7.07e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 | 43 | 7 |
| 9 | 86 | 7.60e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 | 42 | 7 |
| 10 | 86 | 7.66e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 | 43 | 7 |
| 11 | 86 | 6.57e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 42 | 7 |
| 12 | 86 | 7.64e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 40 | 7 |
| 13 | 86 | 7.29e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 | 44 | 7 |
| 14 | 86 | 7.55e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 | 44 | 7 |
| 15 | 86 | 8.74e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 | 41 | 7 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 44 planar dihedral(s), 6 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 11 | backx, severe-dq | 0 | 5 | 1 | YES | NEW Dihedral(C4-C0-C13-O16) = 177.5° (trans) |

**Verdict:** every one of the 1 warning step(s) coincides with at least one planar/linear event new vs step 1 (or a near-linear angle).


## birkholz/maltose

*severe back-xform + saddle pass at steps 27/30*

Steps: 35, converged: False, wall: 10.1s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.80e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 | 26 | 0 |
| 2 | 128 | 2.94e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 | 26 | 0 |
| 3 | 128 | 3.91e+12 | 1.05e+01 | 1.00e-30 | 0 | 0 | 20 | 0 |
| 4 | 128 | 3.76e+12 | 1.03e+01 | 1.00e-30 | 0 | 0 | 15 | 0 |
| 5 | 128 | 3.56e+12 | 1.03e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 6 | 128 | 2.90e+12 | 1.04e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 7 | 128 | 3.08e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 | 12 | 0 |
| 8 | 128 | 3.14e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 9 | 128 | 3.13e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 10 | 128 | 2.35e+12 | 1.09e+01 | 1.00e-30 | 0 | 0 | 9 | 0 |
| 11 | 128 | 3.07e+12 | 1.10e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 12 | 128 | 3.26e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 10 | 0 |
| 13 | 128 | 2.97e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 9 | 0 |
| 14 | 128 | 3.46e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 15 | 128 | 2.92e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 10 | 0 |
| 16 | 128 | 3.43e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 17 | 128 | 2.96e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 10 | 0 |
| 18 | 128 | 2.67e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 9 | 0 |
| 19 | 128 | 2.89e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 9 | 0 |
| 20 | 128 | 2.86e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 9 | 0 |
| 21 | 128 | 3.61e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 22 | 128 | 3.17e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 10 | 0 |
| 23 | 128 | 3.54e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 24 | 128 | 2.81e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 25 | 128 | 2.84e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 26 | 128 | 3.19e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 27 | 128 | 3.38e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 28 | 128 | 3.04e+12 | 1.18e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 29 | 128 | 3.42e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 9 | 0 |
| 30 | 128 | 3.70e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 31 | 128 | 4.43e+12 | 1.04e+01 | 1.00e-30 | 0 | 0 | 7 | 0 |
| 32 | 128 | 3.27e+12 | 1.13e+01 | 1.00e-30 | 0 | 0 | 7 | 0 |
| 33 | 128 | 3.78e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 8 | 0 |
| 34 | 128 | 3.68e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 7 | 0 |
| 35 | 128 | 3.85e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 | 6 | 0 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 26 planar dihedral(s), 0 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 27 | backx, neg-eig, severe-dq | 0 | 4 | 0 | YES | NEW Dihedral(H6-C1-C2-O10) = -177.3° (trans) |
| 30 | backx, neg-eig, severe-dq | 0 | 4 | 0 | YES | NEW Dihedral(H6-C1-C2-O10) = -175.9° (trans) |
| 31 | neg-eig | 0 | 4 | 0 | YES | NEW Dihedral(C11-C13-C24-H26) = -176.8° (trans) |

**Verdict:** every one of the 3 warning step(s) coincides with at least one planar/linear event new vs step 1 (or a near-linear angle).


## birkholz/raffinose

*heavy back-xform + non-converger (steps 42-49)*

Steps: 55, converged: False, wall: 25.0s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 191 | 7.03e+11 | 3.95e+01 | 1.00e-30 | 0 | 0 | 38 | 0 |
| 2 | 191 | 4.64e+11 | 3.26e+01 | 1.00e-30 | 0 | 0 | 36 | 0 |
| 3 | 191 | 8.79e+11 | 2.20e+01 | 1.00e-30 | 0 | 0 | 38 | 0 |
| 4 | 191 | 7.39e+11 | 1.59e+01 | 1.00e-30 | 0 | 0 | 34 | 0 |
| 5 | 191 | 9.34e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 32 | 0 |
| 6 | 191 | 8.13e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 31 | 0 |
| 7 | 191 | 8.17e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 | 26 | 0 |
| 8 | 191 | 8.77e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 | 27 | 0 |
| 9 | 191 | 8.21e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 | 26 | 0 |
| 10 | 191 | 9.17e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 25 | 0 |
| 11 | 191 | 7.83e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 | 21 | 0 |
| 12 | 191 | 8.90e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 18 | 0 |
| 13 | 191 | 8.49e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 14 | 191 | 9.64e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 15 | 191 | 9.03e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 16 | 191 | 8.77e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 12 | 0 |
| 17 | 191 | 9.77e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 18 | 191 | 9.69e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 19 | 191 | 9.31e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 16 | 0 |
| 20 | 191 | 9.50e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 | 17 | 0 |
| 21 | 191 | 9.74e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 17 | 0 |
| 22 | 191 | 9.58e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 16 | 0 |
| 23 | 191 | 9.18e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 | 15 | 0 |
| 24 | 191 | 1.09e+12 | 1.05e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 25 | 191 | 9.00e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 12 | 0 |
| 26 | 191 | 9.25e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 27 | 191 | 9.02e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 | 12 | 0 |
| 28 | 191 | 9.83e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 29 | 191 | 9.89e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 | 12 | 0 |
| 30 | 191 | 9.41e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 31 | 191 | 8.63e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 32 | 191 | 1.03e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 33 | 191 | 1.13e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 | 18 | 0 |
| 34 | 191 | 1.00e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 | 18 | 0 |
| 35 | 191 | 8.44e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 17 | 0 |
| 36 | 191 | 1.03e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 | 16 | 0 |
| 37 | 191 | 9.58e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 15 | 0 |
| 38 | 191 | 9.46e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 | 15 | 0 |
| 39 | 191 | 9.15e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 40 | 191 | 8.41e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 | 16 | 0 |
| 41 | 191 | 9.39e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 42 | 191 | 1.15e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 43 | 191 | 1.10e+12 | 1.03e+01 | 1.00e-30 | 0 | 0 | 15 | 0 |
| 44 | 191 | 9.57e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 | 17 | 0 |
| 45 | 191 | 1.01e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 | 17 | 0 |
| 46 | 191 | 7.37e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 47 | 191 | 4.68e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 | 15 | 0 |
| 48 | 191 | 7.36e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 | 15 | 0 |
| 49 | 191 | 5.52e+11 | 1.11e+01 | 1.00e-30 | 0 | 0 | 14 | 0 |
| 50 | 191 | 8.55e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 51 | 191 | 8.93e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 | 12 | 0 |
| 52 | 191 | 8.17e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 | 11 | 0 |
| 53 | 191 | 7.54e+11 | 1.09e+01 | 1.00e-30 | 0 | 0 | 10 | 0 |
| 54 | 191 | 6.92e+11 | 1.10e+01 | 1.00e-30 | 0 | 0 | 13 | 0 |
| 55 | 191 | 7.25e+11 | 1.09e+01 | 1.00e-30 | 0 | 0 | 12 | 0 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 38 planar dihedral(s), 0 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 42 | backx, neg-eig, severe-dq | 0 | 7 | 0 | YES | NEW Dihedral(C4-C0-O44-H45) = -1.9° (cis) |
| 43 | backx, severe-dq | 0 | 9 | 0 | YES | NEW Dihedral(C4-C0-O44-H45) = -4.0° (cis) |
| 45 | neg-eig | 0 | 8 | 0 | YES | NEW Dihedral(O10-C2-C3-H8) = 175.8° (trans) |
| 46 | backx, neg-eig, severe-dq | 0 | 5 | 0 | YES | NEW Dihedral(O10-C2-C3-H8) = 176.4° (trans) |
| 47 | backx, neg-eig, severe-dq | 0 | 5 | 0 | YES | NEW Dihedral(O21-C14-C17-H20) = -177.4° (trans) |
| 48 | backx, neg-eig, severe-dq | 0 | 6 | 0 | YES | NEW Dihedral(O10-C2-C3-H8) = 176.3° (trans) |
| 49 | backx, severe-dq | 0 | 6 | 0 | YES | NEW Dihedral(O21-C14-C17-H20) = -178.9° (trans) |

**Verdict:** every one of the 7 warning step(s) coincides with at least one planar/linear event new vs step 1 (or a near-linear angle).


## baker/caffeine

*control: 43 neg-eigval events; not coord-singular*

Steps: 55, converged: False, wall: 8.9s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65 | 7.09e+12 | 1.58e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 2 | 65 | 1.45e+13 | 1.57e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 3 | 65 | 1.20e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 4 | 65 | 1.14e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 5 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 6 | 65 | 1.24e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 7 | 65 | 1.23e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 8 | 65 | 1.22e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 9 | 65 | 1.44e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 10 | 65 | 1.70e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 11 | 65 | 1.32e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 12 | 65 | 1.59e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 13 | 65 | 1.02e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 14 | 65 | 1.54e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 15 | 65 | 1.58e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 16 | 65 | 1.51e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 17 | 65 | 1.64e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 18 | 65 | 1.53e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 19 | 65 | 1.35e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 20 | 65 | 1.27e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 21 | 65 | 1.16e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 22 | 65 | 1.55e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 23 | 65 | 1.46e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 24 | 65 | 1.46e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 25 | 65 | 1.04e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 26 | 65 | 1.25e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 27 | 65 | 1.09e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 28 | 65 | 1.38e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 29 | 65 | 1.23e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 30 | 65 | 1.02e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 31 | 65 | 1.49e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 32 | 65 | 1.69e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 33 | 65 | 1.25e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 34 | 65 | 1.24e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 35 | 65 | 1.50e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 36 | 65 | 1.46e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 37 | 65 | 1.47e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 38 | 65 | 1.25e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 39 | 65 | 1.10e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 40 | 65 | 1.41e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 41 | 65 | 1.48e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 42 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 43 | 65 | 1.22e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 44 | 65 | 1.42e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 45 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 46 | 65 | 1.44e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 47 | 65 | 1.14e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 48 | 65 | 1.33e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 49 | 65 | 1.23e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 50 | 65 | 1.22e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 51 | 65 | 1.35e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 52 | 65 | 1.20e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 53 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 54 | 65 | 1.34e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |
| 55 | 65 | 1.50e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 | 42 | 8 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 42 planar dihedral(s), 8 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 27 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 30 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 33 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 35 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 36 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 38 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 39 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 41 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 45 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 47 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 49 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 51 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 52 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |
| 53 | neg-eig | 0 | 0 | 0 | no | Dihedral(C8-N2-C6-C7) = -0.0° (aromatic/baseline) |

**Verdict:** none of the 14 warning step(s) coincides with a new planar/linear event - the warnings here are not coordinate-singularity events.


## birkholz/avobenzone

*neg-eig only: 4 events at steps 14/17/20/49*

Steps: 55, converged: False, wall: 14.4s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 5.26e+11 | 1.40e+01 | 1.00e-30 | 0 | 0 | 63 | 14 |
| 2 | 128 | 5.36e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 63 | 14 |
| 3 | 128 | 6.21e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 4 | 128 | 5.19e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 63 | 14 |
| 5 | 128 | 5.16e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 66 | 14 |
| 6 | 128 | 6.08e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 67 | 14 |
| 7 | 128 | 5.89e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 69 | 14 |
| 8 | 128 | 6.04e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 69 | 14 |
| 9 | 128 | 5.62e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 69 | 14 |
| 10 | 128 | 4.84e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 69 | 14 |
| 11 | 128 | 6.02e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 70 | 14 |
| 12 | 128 | 6.73e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 72 | 14 |
| 13 | 128 | 6.03e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 73 | 14 |
| 14 | 128 | 5.70e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 71 | 14 |
| 15 | 128 | 6.19e+11 | 1.43e+01 | 1.00e-30 | 0 | 0 | 70 | 14 |
| 16 | 128 | 6.11e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 70 | 14 |
| 17 | 128 | 5.72e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 66 | 14 |
| 18 | 128 | 5.37e+11 | 1.43e+01 | 1.00e-30 | 0 | 0 | 64 | 14 |
| 19 | 128 | 4.83e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 66 | 14 |
| 20 | 128 | 6.01e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 65 | 14 |
| 21 | 128 | 5.94e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 66 | 14 |
| 22 | 128 | 6.34e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 66 | 14 |
| 23 | 128 | 6.04e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 63 | 14 |
| 24 | 128 | 6.90e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 64 | 14 |
| 25 | 128 | 8.48e+11 | 1.42e+01 | 1.00e-30 | 0 | 0 | 60 | 14 |
| 26 | 128 | 1.06e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 60 | 14 |
| 27 | 128 | 1.01e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 28 | 128 | 8.43e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 29 | 128 | 8.15e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 30 | 128 | 1.06e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 31 | 128 | 8.85e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 60 | 14 |
| 32 | 128 | 9.84e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 33 | 128 | 9.87e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 34 | 128 | 9.72e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 35 | 128 | 1.41e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 36 | 128 | 1.27e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 37 | 128 | 1.19e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 38 | 128 | 1.07e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 60 | 14 |
| 39 | 128 | 1.11e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 40 | 128 | 7.42e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 41 | 128 | 9.71e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 42 | 128 | 1.01e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 | 63 | 14 |
| 43 | 128 | 8.60e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 63 | 14 |
| 44 | 128 | 7.39e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 65 | 14 |
| 45 | 128 | 8.46e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 46 | 128 | 8.73e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 47 | 128 | 7.90e+11 | 1.40e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 48 | 128 | 6.15e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 49 | 128 | 6.87e+11 | 1.41e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 50 | 128 | 6.59e+11 | 1.38e+01 | 1.00e-30 | 0 | 0 | 40 | 14 |
| 51 | 128 | 6.49e+11 | 1.40e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 52 | 128 | 5.71e+11 | 1.40e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |
| 53 | 128 | 6.06e+11 | 1.40e+01 | 1.00e-30 | 0 | 0 | 58 | 14 |
| 54 | 128 | 6.47e+11 | 1.40e+01 | 1.00e-30 | 0 | 0 | 62 | 14 |
| 55 | 128 | 5.60e+11 | 1.40e+01 | 1.00e-30 | 0 | 0 | 61 | 14 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 63 planar dihedral(s), 14 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 14 | neg-eig | 0 | 10 | 0 | YES | NEW Dihedral(C8-C12-O27-C28) = -1.3° (cis) |
| 17 | neg-eig | 0 | 11 | 0 | YES | NEW Dihedral(C8-C12-O27-C28) = 0.1° (cis) |
| 20 | neg-eig | 0 | 10 | 0 | YES | NEW Dihedral(C8-C12-O27-C28) = 1.1° (cis) |
| 49 | neg-eig | 0 | 10 | 0 | YES | NEW Dihedral(C8-C12-O27-C28) = 0.2° (cis) |

**Verdict:** every one of the 4 warning step(s) coincides with at least one planar/linear event new vs step 1 (or a near-linear angle).


## birkholz/codeine

*neg-eig only: 2 events at steps 15/18*

Steps: 25, converged: False, wall: 7.3s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 122 | 6.94e+12 | 1.66e+01 | 1.00e-30 | 0 | 0 | 26 | 8 |
| 2 | 122 | 3.79e+12 | 1.65e+01 | 1.00e-30 | 0 | 0 | 28 | 8 |
| 3 | 122 | 7.55e+12 | 1.67e+01 | 1.00e-30 | 0 | 0 | 28 | 8 |
| 4 | 122 | 7.47e+12 | 1.69e+01 | 1.00e-30 | 0 | 0 | 28 | 8 |
| 5 | 122 | 7.55e+12 | 1.66e+01 | 1.00e-30 | 0 | 0 | 24 | 8 |
| 6 | 122 | 6.39e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 22 | 8 |
| 7 | 122 | 8.19e+12 | 1.62e+01 | 1.00e-30 | 0 | 0 | 26 | 8 |
| 8 | 122 | 7.01e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 27 | 8 |
| 9 | 122 | 8.14e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 27 | 8 |
| 10 | 122 | 8.12e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 28 | 8 |
| 11 | 122 | 4.74e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 28 | 8 |
| 12 | 122 | 7.06e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 27 | 8 |
| 13 | 122 | 7.17e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 28 | 8 |
| 14 | 122 | 5.26e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 30 | 8 |
| 15 | 122 | 7.51e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 31 | 8 |
| 16 | 122 | 8.02e+12 | 1.60e+01 | 1.00e-30 | 0 | 0 | 25 | 8 |
| 17 | 122 | 6.12e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 33 | 8 |
| 18 | 122 | 7.48e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 32 | 8 |
| 19 | 122 | 5.50e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 22 | 8 |
| 20 | 122 | 7.76e+12 | 1.62e+01 | 1.00e-30 | 0 | 0 | 34 | 8 |
| 21 | 122 | 5.77e+12 | 1.62e+01 | 1.00e-30 | 0 | 0 | 31 | 8 |
| 22 | 122 | 8.76e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 30 | 8 |
| 23 | 122 | 5.36e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 31 | 8 |
| 24 | 122 | 6.72e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 31 | 8 |
| 25 | 122 | 8.30e+12 | 1.63e+01 | 1.00e-30 | 0 | 0 | 32 | 8 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 26 planar dihedral(s), 8 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 15 | neg-eig | 0 | 14 | 0 | YES | NEW Dihedral(C5-C0-C1-C2) = 4.2° (cis) |
| 18 | neg-eig | 0 | 16 | 0 | YES | NEW Dihedral(C5-C0-C1-C2) = 2.6° (cis) |

**Verdict:** every one of the 2 warning step(s) coincides with at least one planar/linear event new vs step 1 (or a near-linear angle).


## baker/methylamine

*neg-eig only: 4 events at steps 5/7/8/9*

Steps: 18, converged: True, wall: 0.4s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals | #near-planar dihedrals | #planar sp3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14 | 2.57e+14 | 6.89e+00 | 1.00e-30 | 0 | 0 | 2 | 1 |
| 2 | 14 | 8.21e+14 | 6.88e+00 | 1.00e-30 | 0 | 0 | 2 | 1 |
| 3 | 14 | 5.71e+14 | 6.82e+00 | 1.00e-30 | 0 | 0 | 2 | 1 |
| 4 | 14 | 4.60e+14 | 6.79e+00 | 1.00e-30 | 0 | 0 | 2 | 1 |
| 5 | 14 | 7.02e+14 | 6.77e+00 | 1.00e-30 | 0 | 0 | 2 | 1 |
| 6 | 14 | 6.49e+14 | 7.23e+00 | 1.00e-30 | 0 | 0 | 0 | 1 |
| 7 | 14 | 4.75e+14 | 6.83e+00 | 1.00e-30 | 0 | 0 | 2 | 1 |
| 8 | 14 | 4.16e+14 | 6.72e+00 | 1.00e-30 | 0 | 0 | 0 | 1 |
| 9 | 14 | 7.77e+14 | 6.58e+00 | 1.00e-30 | 0 | 0 | 0 | 0 |
| 10 | 14 | 6.86e+14 | 5.72e+00 | 1.00e-30 | 0 | 0 | 0 | 0 |
| 11 | 14 | 5.29e+14 | 6.00e+00 | 1.00e-30 | 0 | 0 | 0 | 0 |
| 12 | 14 | 7.64e+14 | 5.71e+00 | 1.00e-30 | 0 | 0 | 0 | 0 |
| 13 | 14 | 4.66e+14 | 5.76e+00 | 1.00e-30 | 0 | 0 | 0 | 0 |
| 14 | 14 | 9.08e+14 | 5.73e+00 | 1.00e-30 | 0 | 0 | 0 | 0 |
| 15 | 14 | 7.58e+14 | 5.70e+00 | 1.00e-30 | 0 | 0 | 0 | 0 |
| 16 | 14 | 1.18e+15 | 5.67e+00 | 1.00e-30 | 0 | 0 | 2 | 0 |
| 17 | 14 | 7.14e+14 | 5.68e+00 | 1.00e-30 | 0 | 0 | 2 | 0 |
| 18 | 14 | 6.80e+14 | 5.68e+00 | 1.00e-30 | 0 | 0 | 2 | 0 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


### Warning-step / geometry correlation

For each step where `benchmark_trajectory_check.py`'s log scan would record a warning, the columns below report the count of geometric degeneracies at that step. `near_lin_ang` counts angles > 175°. `new_planar_dih` is the number of dihedrals within 5° of 0°/180° that were NOT already planar at step 1 (subtracting the aromatic-ring baseline). `new_planar_sp3` is the number of three-coord heavy centres whose substituent-angle sum exceeds 355° and were not already planar at step 1. The "geom flag" column is YES if at least one of those three is nonzero.

Step-1 baseline: 2 planar dihedral(s), 1 planar sp3 centre(s).

| step | warning kinds | n_near_lin_ang | new_planar_dih | new_planar_sp3 | geom flag | example |
|---:|---|---:|---:|---:|---|---|
| 5 | neg-eig | 0 | 0 | 0 | no | Dihedral(H2-N0-C1-H3) = 179.4° (aromatic/baseline) |
| 7 | neg-eig | 0 | 0 | 0 | no | Dihedral(H2-N0-C1-H3) = 177.5° (aromatic/baseline) |
| 8 | neg-eig | 0 | 0 | 0 | no | centre N0 planar (aromatic/baseline) |
| 9 | neg-eig | 0 | 0 | 0 | no | - |

**Verdict:** none of the 4 warning step(s) coincides with a new planar/linear event - the warnings here are not coordinate-singularity events.


## Conclusions

Two distinct mechanisms produce the `Pseudoinverse gap of only:` warning:

1. **Inflated dihedral B-row** (estradiol, disilyl_ether). A near-linear three-atom motif (`H-C-O`, `Si-O-Si`) makes adjacent dihedrals' gradient formula blow up: in `Dihedral.eval(grad=True)` the term `1 / norm(a1)` diverges as the central angle approaches 180°. This pushes a single B-row norm to ~10²-10³ while everything else stays at ~1, creating a huge sv[0] and a spurious gap at index 0. Truncation then zeroes a far-away real DOF.

2. **Rank drop** (allene). A symmetry-imposed linear backbone (`C=C=C`) means rotation about that axis genuinely is not a DOF, so the B-matrix has one extra near-zero singular value. The pinv gap fires at the natural-redundancy index but the gap value is small (~10³-10⁷) and a real H-C-H angle DOF gets truncated alongside the missing rotational mode.

The other flagged cases (inosine_cation, maltose, raffinose, caffeine, plus the neg-eig-only cases avobenzone, codeine, methylamine) have pinv gap values uniformly above `1e11` and never trigger the pinv warning. Tracking dihedrals within 5° of 0°/180° and three-coord centres within 5° of their substituent plane (subtracting the step-1 aromatic baseline) splits these into two groups:

- **Planar-dihedral coincidence** (inosine_cation, maltose, raffinose, avobenzone, codeine): every step that emitted a back-xform, severe-dq, or neg-eig warning had at least one dihedral cross into the 5°-of-planar region between the starting geometry and that step. On raffinose the offenders are `Dihedral(C4-C0-O44-H45)` and `Dihedral(O10-C2-C3-H8)` cycling through 0°/180° as the sugar ring puckers; on maltose it is `Dihedral(H6-C1-C2-O10)` going trans; on avobenzone it is `Dihedral(C8-C12-O27-C28)` flipping through cis on every neg-eig step (14, 17, 20, 49). These warnings are the same class of coordinate-singularity event as the pinv@final cases - the back-transform Newton iteration just happens to absorb the gradient blow-up before the pinv truncation fires.

- **PES topology, not coordinate singularity** (caffeine, methylamine): zero of the neg-eig warning steps coincided with a new planar event. Caffeine's 43 neg-eig events and methylamine's 4 are genuine BFGS spurious-unstable-mode events on a confusing PES region, not numerical breakdowns of the internal-coord representation.

In every truncation case the geometric culprit is identifiable in one line:

| molecule | culprit | mechanism |
|---|---|---|
| estradiol | `Angle(H37-C36-O40)` -> 179.66° | dihedral B-row inflated (`Dihedral(H37-C36-O40-C34)` and `-H41`) |
| disilyl_ether | `Angle(Si0-O2-Si1)` -> 179.94° | six dihedrals `H?-Si?-O-Si?` B-row inflated to ~7.5e+02 |
| allene | `Angle(C1-C0-C2)` -> 179.98° (symmetry) | rank drop; H-C-H angles get truncated instead |
| inosine_cation | `Dihedral(C4-C0-C13-O16)` -> 177.5° | new trans-planar dihedral coincides with step-11 severe-dq |
| maltose | `Dihedral(H6-C1-C2-O10)` -> ~177° | new trans dihedral on every severe-dq + neg-eig step (27/30/31) |
| raffinose | `Dihedral(C4-C0-O44-H45)`, `Dihedral(O10-C2-C3-H8)`, `Dihedral(O21-C14-C17-H20)` -> ~0°/180° | sugar-ring puckering through planar configurations over steps 42-49 |
| avobenzone | `Dihedral(C8-C12-O27-C28)` -> ~0° | aryl-ether C-O dihedral flipping cis on every neg-eig step (14/17/20/49) |
| codeine | `Dihedral(C5-C0-C1-C2)` -> ~3° | ring dihedral going cis on the neg-eig steps (15/18) |
| caffeine | none | 43 neg-eig events on a confusing PES; no new planar/linear event on any warning step |
| methylamine | none | 4 neg-eig events; N inversion baseline-planar already, no new planar/linear event |

### Planar/linear coincidence across all warning categories

| case | warning steps w/ planar-or-linear flag | total warning steps | verdict |
|---|---:|---:|---|
| birkholz/estradiol | 6 | 6 | all planar/linear |
| baker/allene | 9 | 9 | all planar/linear |
| baker/disilyl_ether | 2 | 3 | partial |
| birkholz/inosine_cation | 1 | 1 | all planar/linear |
| birkholz/maltose | 3 | 3 | all planar/linear |
| birkholz/raffinose | 7 | 7 | all planar/linear |
| baker/caffeine | 0 | 14 | none planar/linear |
| birkholz/avobenzone | 4 | 4 | all planar/linear |
| birkholz/codeine | 2 | 2 | all planar/linear |
| baker/methylamine | 0 | 4 | none planar/linear |
