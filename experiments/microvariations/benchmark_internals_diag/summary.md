# Internal-coordinate health check on problematic benchmark cases

For every molecule that `benchmark_trajectory_check.py` flagged, this script re-runs the same `Berny+MopacSolver` trajectory but captures the Wilson B-matrix at every step. The tables below distill **which internal coordinate(s) fire** - i.e. which Bond/Angle/Dihedral becomes the dominant contributor to the singular direction that `Math.pinv` would (and on three molecules, does) silently truncate.

Atom indices are zero-based, prefixed with the element symbol of that atom (e.g. `H37` is hydrogen #37 in the xyz file). The "top contrib" coordinate is the one whose weight in the eigenvector at the pinv gap is largest at the final step where the warning fires.


## birkholz/estradiol

*pinv@final: H-C-O angle near-linear (known smoking gun)*

Steps: 11, converged: True, wall: 3.0s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 125 | 1.18e+13 | 2.23e+01 | 1.00e-30 | 0 | 0 |
| 2 | 125 | 1.18e+13 | 2.07e+01 | 1.00e-30 | 0 | 0 |
| 3 | 125 | 4.75e+12 | 2.26e+01 | 1.00e-30 | 0 | 0 |
| 4 | 125 | 7.30e+12 | 3.08e+01 | 1.00e-30 | 0 | 0 |
| 5 | 125 | 4.54e+12 | 3.95e+01 | 1.00e-30 | 0 | 0 |
| 6 | 125 | 5.19e+12 | 7.93e+01 | 1.00e-30 | 0 | 0 |
| 7 | 125 | 1.12e+12 | 3.37e+02 | 1.00e-30 | 1 | 2 |
| 8 | 0 | 2.74e+03 | 4.38e+04 | 1.00e-30 | 1 | 2 |
| 9 | 0 | 2.74e+03 | 4.38e+04 | 1.00e-30 | 1 | 2 |
| 10 | 0 | 2.62e+03 | 4.19e+04 | 1.00e-30 | 1 | 2 |
| 11 | 0 | 2.61e+03 | 4.19e+04 | 1.00e-30 | 1 | 2 |

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

## baker/allene

*pinv@final: C=C=C linear backbone (symmetry-imposed)*

Steps: 12, converged: True, wall: 0.3s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 4.61e+14 | 3.99e+00 | 1.00e-30 | 1 | 0 |
| 2 | 13 | 8.22e+09 | 4.01e+00 | 1.00e-30 | 1 | 0 |
| 3 | 13 | 6.01e+10 | 4.00e+00 | 1.00e-30 | 1 | 0 |
| 4 | 13 | 2.07e+11 | 4.01e+00 | 1.00e-30 | 1 | 0 |
| 5 | 13 | 2.01e+07 | 4.02e+00 | 1.00e-30 | 1 | 0 |
| 6 | 13 | 1.72e+05 | 4.02e+00 | 2.55e-17 | 1 | 0 |
| 7 | 13 | 1.60e+03 | 4.02e+00 | 1.00e-30 | 1 | 0 |
| 8 | 13 | 6.35e+05 | 4.02e+00 | 1.44e-17 | 1 | 0 |
| 9 | 13 | 4.12e+04 | 4.02e+00 | 3.91e-17 | 1 | 0 |
| 10 | 13 | 1.05e+09 | 4.02e+00 | 1.00e-30 | 1 | 0 |
| 11 | 13 | 3.12e+06 | 4.02e+00 | 1.00e-30 | 1 | 0 |
| 12 | 13 | 3.04e+07 | 4.02e+00 | 4.70e-16 | 1 | 0 |

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

## baker/disilyl_ether

*pinv@final: Si-O-Si linearization*

Steps: 7, converged: True, wall: 0.6s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 5.66e+13 | 1.41e+01 | 1.00e-30 | 0 | 0 |
| 2 | 20 | 8.66e+13 | 1.81e+01 | 1.00e-30 | 0 | 0 |
| 3 | 20 | 3.41e+13 | 3.50e+01 | 1.00e-30 | 0 | 0 |
| 4 | 20 | 7.35e+13 | 5.86e+01 | 1.00e-30 | 0 | 0 |
| 5 | 20 | 1.81e+13 | 1.89e+02 | 1.00e-30 | 0 | 0 |
| 6 | 20 | 6.31e+11 | 1.77e+03 | 1.00e-30 | 1 | 6 |
| 7 | 0 | 9.40e+05 | 3.36e+06 | 1.00e-30 | 1 | 6 |

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

## birkholz/inosine_cation

*severe back-xform RMS(dq) at step 11*

Steps: 15, converged: False, wall: 3.1s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86 | 7.08e+12 | 1.31e+01 | 1.00e-30 | 0 | 0 |
| 2 | 86 | 6.94e+12 | 1.35e+01 | 1.00e-30 | 0 | 0 |
| 3 | 86 | 7.72e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 |
| 4 | 86 | 7.13e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 |
| 5 | 86 | 6.52e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 |
| 6 | 86 | 5.69e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 |
| 7 | 86 | 7.54e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 |
| 8 | 86 | 7.07e+12 | 1.39e+01 | 1.00e-30 | 0 | 0 |
| 9 | 86 | 7.60e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 |
| 10 | 86 | 7.66e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 |
| 11 | 86 | 6.57e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 |
| 12 | 86 | 7.64e+12 | 1.41e+01 | 1.00e-30 | 0 | 0 |
| 13 | 86 | 7.29e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 |
| 14 | 86 | 7.55e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 |
| 15 | 86 | 8.74e+12 | 1.40e+01 | 1.00e-30 | 0 | 0 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


## birkholz/maltose

*severe back-xform + saddle pass at steps 27/30*

Steps: 35, converged: False, wall: 8.8s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.80e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 2 | 128 | 2.94e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 3 | 128 | 3.91e+12 | 1.05e+01 | 1.00e-30 | 0 | 0 |
| 4 | 128 | 3.76e+12 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 5 | 128 | 3.56e+12 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 6 | 128 | 2.90e+12 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 7 | 128 | 3.08e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 8 | 128 | 3.14e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 9 | 128 | 3.13e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 10 | 128 | 2.35e+12 | 1.09e+01 | 1.00e-30 | 0 | 0 |
| 11 | 128 | 3.07e+12 | 1.10e+01 | 1.00e-30 | 0 | 0 |
| 12 | 128 | 3.26e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 13 | 128 | 2.97e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 14 | 128 | 3.46e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 15 | 128 | 2.92e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 16 | 128 | 3.43e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 17 | 128 | 2.96e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 18 | 128 | 2.67e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 19 | 128 | 2.89e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 20 | 128 | 2.86e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 21 | 128 | 3.61e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 22 | 128 | 3.17e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 23 | 128 | 3.54e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 24 | 128 | 2.81e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 25 | 128 | 2.84e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 26 | 128 | 3.19e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 27 | 128 | 3.38e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 28 | 128 | 3.04e+12 | 1.18e+01 | 1.00e-30 | 0 | 0 |
| 29 | 128 | 3.42e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 30 | 128 | 3.70e+12 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 31 | 128 | 4.43e+12 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 32 | 128 | 3.27e+12 | 1.13e+01 | 1.00e-30 | 0 | 0 |
| 33 | 128 | 3.78e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 34 | 128 | 3.68e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |
| 35 | 128 | 3.85e+12 | 1.12e+01 | 1.00e-30 | 0 | 0 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


## birkholz/raffinose

*heavy back-xform + non-converger (steps 42-49)*

Steps: 55, converged: False, wall: 20.5s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 191 | 7.03e+11 | 3.95e+01 | 1.00e-30 | 0 | 0 |
| 2 | 191 | 4.64e+11 | 3.26e+01 | 1.00e-30 | 0 | 0 |
| 3 | 191 | 8.79e+11 | 2.20e+01 | 1.00e-30 | 0 | 0 |
| 4 | 191 | 7.39e+11 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 5 | 191 | 9.34e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 6 | 191 | 8.13e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 7 | 191 | 8.17e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 8 | 191 | 8.77e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 9 | 191 | 8.21e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 10 | 191 | 9.17e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 11 | 191 | 7.83e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 |
| 12 | 191 | 8.90e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 13 | 191 | 8.49e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 14 | 191 | 9.64e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 15 | 191 | 9.03e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 16 | 191 | 8.77e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 17 | 191 | 9.77e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 18 | 191 | 9.69e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 |
| 19 | 191 | 9.31e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 20 | 191 | 9.50e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 |
| 21 | 191 | 9.74e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 22 | 191 | 9.58e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 23 | 191 | 9.18e+11 | 1.05e+01 | 1.00e-30 | 0 | 0 |
| 24 | 191 | 1.09e+12 | 1.05e+01 | 1.00e-30 | 0 | 0 |
| 25 | 191 | 9.00e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 26 | 191 | 9.25e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 27 | 191 | 9.02e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 28 | 191 | 9.83e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 |
| 29 | 191 | 9.89e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 30 | 191 | 9.41e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 31 | 191 | 8.63e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 32 | 191 | 1.03e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 33 | 191 | 1.13e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 34 | 191 | 1.00e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 35 | 191 | 8.44e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 36 | 191 | 1.03e+12 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 37 | 191 | 9.58e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 38 | 191 | 9.46e+11 | 1.06e+01 | 1.00e-30 | 0 | 0 |
| 39 | 191 | 9.15e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 40 | 191 | 8.41e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 41 | 191 | 9.39e+11 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 42 | 191 | 1.15e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 43 | 191 | 1.10e+12 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 44 | 191 | 9.57e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 45 | 191 | 1.01e+12 | 1.07e+01 | 1.00e-30 | 0 | 0 |
| 46 | 191 | 7.37e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 |
| 47 | 191 | 4.68e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 48 | 191 | 7.36e+11 | 1.03e+01 | 1.00e-30 | 0 | 0 |
| 49 | 191 | 5.52e+11 | 1.11e+01 | 1.00e-30 | 0 | 0 |
| 50 | 191 | 8.55e+11 | 1.04e+01 | 1.00e-30 | 0 | 0 |
| 51 | 191 | 8.93e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 |
| 52 | 191 | 8.17e+11 | 1.08e+01 | 1.00e-30 | 0 | 0 |
| 53 | 191 | 7.54e+11 | 1.09e+01 | 1.00e-30 | 0 | 0 |
| 54 | 191 | 6.92e+11 | 1.10e+01 | 1.00e-30 | 0 | 0 |
| 55 | 191 | 7.25e+11 | 1.09e+01 | 1.00e-30 | 0 | 0 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


## baker/caffeine

*control: 43 neg-eigval events; not coord-singular*

Steps: 55, converged: False, wall: 9.3s.

### Per-step pinv-gap summary

| step | pinv_gap_idx | gap | sv_max | sv_min | #near-lin angles | #near-lin dihedrals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65 | 7.09e+12 | 1.58e+01 | 1.00e-30 | 0 | 0 |
| 2 | 65 | 1.45e+13 | 1.57e+01 | 1.00e-30 | 0 | 0 |
| 3 | 65 | 1.20e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 4 | 65 | 1.14e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 5 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 6 | 65 | 1.24e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 7 | 65 | 1.23e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 8 | 65 | 1.22e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 9 | 65 | 1.44e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 10 | 65 | 1.70e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 11 | 65 | 1.32e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 12 | 65 | 1.59e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 13 | 65 | 1.02e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 14 | 65 | 1.54e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 15 | 65 | 1.58e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 16 | 65 | 1.51e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 17 | 65 | 1.64e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 18 | 65 | 1.53e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 19 | 65 | 1.35e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 20 | 65 | 1.27e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 21 | 65 | 1.16e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 22 | 65 | 1.55e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 23 | 65 | 1.46e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 24 | 65 | 1.46e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 25 | 65 | 1.04e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 26 | 65 | 1.25e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 27 | 65 | 1.09e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 28 | 65 | 1.38e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 29 | 65 | 1.23e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 30 | 65 | 1.02e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 31 | 65 | 1.49e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 32 | 65 | 1.69e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 33 | 65 | 1.25e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 34 | 65 | 1.24e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 35 | 65 | 1.50e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 36 | 65 | 1.46e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 37 | 65 | 1.47e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 38 | 65 | 1.25e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 39 | 65 | 1.10e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 40 | 65 | 1.41e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 41 | 65 | 1.48e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 42 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 43 | 65 | 1.22e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 44 | 65 | 1.42e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 45 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 46 | 65 | 1.44e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 47 | 65 | 1.14e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 48 | 65 | 1.33e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 49 | 65 | 1.23e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 50 | 65 | 1.22e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 51 | 65 | 1.35e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 52 | 65 | 1.20e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 53 | 65 | 1.30e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 54 | 65 | 1.34e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |
| 55 | 65 | 1.50e+13 | 1.59e+01 | 1.00e-30 | 0 | 0 |

No step recorded a pinv gap value below `1e8` - the natural redundancy boundary is well-separated throughout the trajectory, and `Math.pinv` would never log a warning for this molecule.


## Conclusions

Two distinct mechanisms produce the `Pseudoinverse gap of only:` warning:

1. **Inflated dihedral B-row** (estradiol, disilyl_ether). A near-linear three-atom motif (`H-C-O`, `Si-O-Si`) makes adjacent dihedrals' gradient formula blow up: in `Dihedral.eval(grad=True)` the term `1 / norm(a1)` diverges as the central angle approaches 180°. This pushes a single B-row norm to ~10²-10³ while everything else stays at ~1, creating a huge sv[0] and a spurious gap at index 0. Truncation then zeroes a far-away real DOF.

2. **Rank drop** (allene). A symmetry-imposed linear backbone (`C=C=C`) means rotation about that axis genuinely is not a DOF, so the B-matrix has one extra near-zero singular value. The pinv gap fires at the natural-redundancy index but the gap value is small (~10³-10⁷) and a real H-C-H angle DOF gets truncated alongside the missing rotational mode.

The other flagged cases (inosine_cation, maltose, raffinose, caffeine) have pinv gap values uniformly above `1e11` and never trigger the warning. Their pathologies are different: saddle-pass attempts (negative eigenvalues + huge predicted energy change) for maltose/raffinose, multivalued Cartesian↔internal map at one step for inosine_cation, and a confusing PES for caffeine. None of these implicates `Math.pinv` truncation.

In every truncation case the geometric culprit is identifiable in one line:

| molecule | culprit | mechanism |
|---|---|---|
| estradiol | `Angle(H37-C36-O40)` -> 179.66° | dihedral B-row inflated (`Dihedral(H37-C36-O40-C34)` and `-H41`) |
| disilyl_ether | `Angle(Si0-O2-Si1)` -> 179.94° | six dihedrals `H?-Si?-O-Si?` B-row inflated to ~7.5e+02 |
| allene | `Angle(C1-C0-C2)` -> 179.98° (symmetry) | rank drop; H-C-H angles get truncated instead |
