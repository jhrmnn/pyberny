#!/usr/bin/env python3
"""Why pyberny's approximate (BFGS) Hessian never catches the saddle.

Background (issue #148): the no-noise methylamine run converges to a planar
first-order saddle. Its *true* Hessian has one imaginary mode (the nitrogen
umbrella inversion), yet pyberny reports convergence to a minimum. This script
shows why, entirely from the optimizer's own state, and shows that the blind
direction is automatically detectable.

It runs the clean GFN2-xTB optimization from the Baker start with ``debug=True``
(so every step exposes the internal-coordinate BFGS Hessian and the running
geometry), then reports three things:

1. **The BFGS spectrum.** Eigenvalues of the converged approximate Hessian
   (restricted to the non-redundant internal subspace). All are positive: the
   optimizer "sees" a minimum. The eigenvalue best aligned with the true
   umbrella mode is compared against the true curvature and the initial model
   guess -- it is essentially the untouched guess value, with the wrong sign.

2. **The unsampled subspace.** BFGS only learns curvature along directions it
   actually stepped in (the secant condition H.dq = dg). The rank of the
   accumulated steps {dq} is compared with the active dimension; the orthogonal
   complement is the set of directions that received *no* gradient information.

3. **The umbrella mode was never revised.** Its BFGS curvature stays at the
   model-guess value (ratio ~1), confirming it is one of the unsampled
   directions -- with the wrong (positive) sign relative to the true Hessian.

Run ``bfgs_blindness.py``; requires the ``[benchmark]`` extra (``tblite``).
"""

from __future__ import annotations

import numpy as np
from numpy import dot

from berny import Berny, Math
from berny.benchmarks import iter_molecules
from berny.coords import angstrom
from berny.solvers import XTBSolver
from berny.species_data import get_property


def _cartesian_hessian(geom, h=2e-3):
    """Finite-difference GFN2-xTB Cartesian Hessian at ``geom`` (a.u.)."""
    from tblite.interface import Calculator

    nums = np.array([int(get_property(sp, "number")) for sp in geom.species])
    pos = np.array(geom.coords) * angstrom

    def grad(p):
        calc = Calculator("GFN2-xTB", nums, p)
        calc.set("verbosity", 0)
        return np.array(calc.singlepoint().get("gradient")).reshape(-1)

    n = pos.size
    flat = pos.reshape(-1)
    H = np.zeros((n, n))
    for i in range(n):
        fp, fm = flat.copy(), flat.copy()
        fp[i] += h
        fm[i] -= h
        H[i] = (grad(fp.reshape(-1, 3)) - grad(fm.reshape(-1, 3))) / (2 * h)
    return (H + H.T) / 2


def main():
    geom, ref = list(iter_molecules("baker", ["methylamine"]))[0][1:]

    opt = Berny(geom.copy(), debug=True, maxsteps=150)
    solver = XTBSolver(charge=ref["charge"], mult=ref["mult"])
    next(solver)
    steps, prev_q, state = [], None, None
    for g in opt:
        energy, gradients = solver.send((list(g), g.lattice))
        state = opt.send((energy, gradients))
        q = state["coords"].eval_geom(state["geom"])
        if prev_q is not None:
            steps.append(q - prev_q)
        prev_q = q

    H = state["H"]
    coords = state["coords"]
    ggeom = state["geom"]
    B = coords.B_matrix(ggeom)
    B_inv = B.T.dot(Math.pinv(np.dot(B, B.T)))
    proj = dot(B, B_inv)
    wp, Vp = np.linalg.eigh((proj + proj.T) / 2)
    active = Vp[:, wp > 0.5]  # basis of the non-redundant internal subspace
    nact = active.shape[1]

    # (1) BFGS spectrum and the umbrella-aligned eigenvalue
    Hsub = active.T.dot(proj.dot(H).dot(proj)).dot(active)
    ev, Vc = np.linalg.eigh((Hsub + Hsub.T) / 2)
    intvecs = active.dot(Vc)
    H0 = coords.hessian_guess(ggeom)
    ev_guess = np.linalg.eigvalsh(
        active.T.dot(proj.dot(H0).dot(proj)).dot(active)
    )

    Hcart = _cartesian_hessian(ggeom)
    wcart, Vcart = np.linalg.eigh(Hcart)
    neg = Vcart[:, 0]
    neg = neg / np.linalg.norm(neg)

    overlaps = []
    for k in range(intvecs.shape[1]):
        dx = B_inv.dot(intvecs[:, k])
        nrm = np.linalg.norm(dx)
        overlaps.append(abs(dot(dx / nrm, neg)) if nrm > 1e-12 else 0.0)
    k_umb = int(np.argmax(overlaps))

    print("(1) Converged BFGS Hessian — what the optimizer believes")
    print(f"    eigenvalues (a.u.): {np.round(ev, 4)}")
    print(
        f"    negative eigenvalues: {int((ev < -1e-6).sum())}  (=> sees a minimum)"
    )
    print("    curvature along the umbrella mode (a.u.):")
    print(f"      true Hessian : {wcart[0]:+.4f}   (imaginary mode)")
    print(f"      model guess  : {ev_guess.min():+.4f}")
    print(
        f"      BFGS         : {ev[k_umb]:+.4f}   (overlap {overlaps[k_umb]:.2f})"
    )

    # (2) unsampled subspace from the rank of the accumulated steps. With only
    # `rank` independent displacements in `nact` dimensions, at least
    # nact - rank directions received no secant information at all.
    Qa = active.T.dot(np.array(steps).T)
    sv = np.linalg.svd(Qa, compute_uv=False)
    rank = int((sv > 1e-6 * sv.max()).sum())

    # (3) the umbrella mode is one of them: its BFGS curvature never moved off
    # the model guess (ratio ~1), whereas a sampled stiff mode does move. This
    # ratio is deterministic; an empirical "is the imaginary mode inside the
    # unsampled span" overlap is not -- tblite gradients are not bitwise
    # reproducible, so the tiny asymmetry leaks the umbrella mode in and out of
    # the sampled steps run to run.
    umb_ratio = ev[k_umb] / ev_guess.min()

    print("\n(2) What BFGS was actually told (secant condition H.dq = dg)")
    print(
        f"    steps taken: {len(steps)}   active DOF: {nact}   sampled rank: {rank}"
    )
    print(f"    directions with NO gradient information: {nact - rank}")
    print("\n(3) Was the umbrella mode's curvature ever revised?")
    print(
        f"    BFGS / guess curvature along umbrella: {umb_ratio:.2f}"
        "  (~1 => untouched)"
    )


if __name__ == "__main__":
    main()
