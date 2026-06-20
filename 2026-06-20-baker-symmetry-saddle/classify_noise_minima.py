#!/usr/bin/env python3
"""Classify the stationary points pyberny reaches from the Baker start geometries.

Background (issue #148): for a handful of Baker molecules, adding tiny Gaussian
noise to the start geometry makes pyberny reach a *lower* minimum than the
unperturbed run, by a constant, seed-independent amount. This script settles
*why* by characterising what the no-noise run actually converges to.

For each molecule it

1. optimizes from the bundled Baker start geometry with GFN2-xTB (the clean,
   no-noise run);
2. builds the Cartesian Hessian by finite differences of the analytic gradient
   and counts its imaginary (negative-curvature) vibrational modes -- i.e. the
   *order* of the stationary point (0 = minimum, 1 = transition state, >=2 =
   higher saddle);
3. re-optimizes from a symmetry-broken (slightly perturbed) copy of the same
   start and characterises that stationary point too.

The verdict per molecule is then one of

* ``saddle`` -- the clean run sits at a true saddle point (>=1 imaginary mode);
  the start geometry is symmetric and the optimizer, which cannot break exact
  symmetry, descended within the symmetric subspace to a constrained
  stationary point. The perturbed run finds the real minimum below it.
* ``conformer`` -- the clean run is a genuine minimum (0 imaginary modes) but a
  higher-energy conformer; the perturbed start simply lands in a lower basin.

Run ``classify_noise_minima.py`` for the default affected set, or pass molecule
names. Requires the ``[benchmark]`` extra (``tblite``).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from berny import Berny, geomlib
from berny.benchmarks import iter_molecules
from berny.coords import angstrom
from berny.species_data import get_property

HARTREE_KCAL = 627.5094740631
# Hessian eigenvalue (Hartree / bohr**2 / amu_au) -> wavenumber (cm**-1).
AU_PER_AMU = 1822.888486
HARTREE_CM = 219474.63
# After projecting out the 6 translation/rotation modes, residual finite-
# difference noise on the soft torsions still leaves a few cm**-1 of scatter
# around zero; only negative modes below this count as genuine negative
# curvature (a saddle direction).
ZERO_MODE_CM = 10.0

# The molecules flagged in issue #148, plus benzene as a control (its planar
# reference *is* the minimum, so it should classify as a 0-imaginary-mode
# "conformer"/minimum and stay put under a small perturbation).
DEFAULT_MOLECULES = [
    "methylamine",
    "ethanol",
    "mesityl_oxide",
    "acanil01",
    "caffeine",
    "benzidine",
    "histidine",
]


def _eval(numbers, pos_bohr, charge, uhf):
    """GFN2-xTB energy (Ha) and gradient (Ha/bohr) at ``pos_bohr``."""
    from tblite.interface import Calculator

    calc = Calculator(
        "GFN2-xTB", numbers, pos_bohr, charge=float(charge), uhf=uhf
    )
    calc.set("verbosity", 0)
    res = calc.singlepoint()
    return float(res.get("energy")), np.array(res.get("gradient"))


def _optimize(geom, ref, maxsteps=150):
    """Run one clean GFN2-xTB optimization, returning the final Geometry."""
    from berny.solvers import XTBSolver

    opt = Berny(geom, maxsteps=maxsteps)
    solver = XTBSolver(charge=ref["charge"], mult=ref["mult"])
    next(solver)
    energy = None
    final = geom
    for g in opt:
        energy, gradients = solver.send((list(g), g.lattice))
        opt.send((energy, gradients))
        final = g  # Berny yields a fresh Geometry each step; keep the last
    return final, energy, opt.converged


def _arrays(geom):
    numbers = np.array([int(get_property(sp, "number")) for sp in geom.species])
    pos = np.array(geom.coords) * angstrom
    masses = np.array([float(get_property(sp, "mass")) for sp in geom.species])
    return numbers, pos, masses


def _transrot_projector(pos, masses):
    """Mass-weighted projector that removes overall translation and rotation.

    Returns ``P = I - B B^T`` where the columns of ``B`` are the orthonormal
    mass-weighted translation/rotation vectors; projecting the mass-weighted
    Hessian with ``P`` collapses the 6 external modes onto exactly zero so the
    remaining spectrum is the internal vibrations.
    """
    nat = len(masses)
    sm = np.sqrt(np.repeat(masses, 3))
    com = (masses[:, None] * pos).sum(0) / masses.sum()
    r = pos - com
    vecs = np.zeros((6, 3 * nat))
    for k in range(3):  # translations
        t = np.zeros((nat, 3))
        t[:, k] = 1.0
        vecs[k] = (t.reshape(-1)) * sm
    for k, (a, b) in enumerate(
        ((1, 2), (2, 0), (0, 1))
    ):  # infinitesimal rotations
        rot = np.zeros((nat, 3))
        rot[:, a] = r[:, b]
        rot[:, b] = -r[:, a]
        vecs[3 + k] = rot.reshape(-1) * sm
    q, _ = np.linalg.qr(vecs.T)  # orthonormal basis of the external subspace
    return np.eye(3 * nat) - q @ q.T


def _imaginary_modes(geom, charge, uhf, h=2e-3):
    """Return (gmax, sorted wavenumbers cm**-1, n_imaginary) at ``geom``."""
    numbers, pos, masses = _arrays(geom)
    _, g = _eval(numbers, pos, charge, uhf)
    gmax = float(np.abs(g).max())
    n = pos.size
    flat = pos.reshape(-1)
    H = np.zeros((n, n))
    for i in range(n):
        fp, fm = flat.copy(), flat.copy()
        fp[i] += h
        fm[i] -= h
        _, gp = _eval(numbers, fp.reshape(-1, 3), charge, uhf)
        _, gm = _eval(numbers, fm.reshape(-1, 3), charge, uhf)
        H[i] = (gp.reshape(-1) - gm.reshape(-1)) / (2 * h)
    H = (H + H.T) / 2
    m = np.repeat(masses, 3)
    Hmw = H / np.sqrt(np.outer(m, m))
    P = _transrot_projector(pos, masses)
    Hmw = P @ Hmw @ P
    w = np.linalg.eigvalsh(Hmw)
    freqs = np.array(
        [
            (
                -np.sqrt(-ev / AU_PER_AMU) * HARTREE_CM
                if ev < 0
                else np.sqrt(ev / AU_PER_AMU) * HARTREE_CM
            )
            for ev in w
        ]
    )
    n_imag = int((freqs < -ZERO_MODE_CM).sum())
    return gmax, freqs, n_imag


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("molecules", nargs="*", default=None, help="molecule names")
    p.add_argument(
        "--sigma", type=float, default=0.05, help="perturbation RMS (A)"
    )
    p.add_argument(
        "--seeds", type=int, default=3, help="perturbation seeds to try"
    )
    args = p.parse_args(argv)

    names = args.molecules or DEFAULT_MOLECULES
    refs = {
        name: (geom, ref) for name, geom, ref in iter_molecules("baker", names)
    }

    print(
        f'{"molecule":14s} {"clean E":>12s} {"order":>5s} {"lowest modes (cm-1)":>22s}'
        f'  {"best dE":>9s} {"order":>5s}  verdict'
    )
    for name in names:
        geom, ref = refs[name]
        charge, uhf = ref["charge"], ref["mult"] - 1

        clean, e_clean, _ = _optimize(geom.copy(), ref)
        _, freqs_c, order_c = _imaginary_modes(clean, charge, uhf)

        # The symmetry-breaking direction is random, so a single perturbation
        # may relax straight back; try a few seeds and keep the lowest minimum.
        best_e, best_order = e_clean, order_c
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            coords = np.array(geom.coords) + rng.normal(
                0, args.sigma, (len(geom), 3)
            )
            pert, e_pert, _ = _optimize(
                geomlib.Geometry(geom.species, coords), ref
            )
            if e_pert < best_e:
                _, _, best_order = _imaginary_modes(pert, charge, uhf)
                best_e = e_pert

        dE = (best_e - e_clean) * HARTREE_KCAL
        if order_c >= 1:
            verdict = "saddle"
        elif dE < -0.05:
            verdict = "conformer"
        else:
            verdict = "stable-min"
        low = np.round(freqs_c[: max(order_c, 1) + 2], 1)
        print(
            f"{name:14s} {e_clean:12.6f} {order_c:5d} {low!s:>22s}"
            f"  {dE:+8.2f}  {best_order:5d}  {verdict}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
