#!/usr/bin/env python3
"""Render the two figures for this report (GFN2-xTB via ``tblite``).

* ``inversion_doublewell.png`` -- the methylamine umbrella-inversion energy
  profile. Starting from the planar saddle (the no-noise reference minimum),
  we displace along the pyramidalization vector ``lower_min - reference_min``
  in both directions; the energy is a symmetric double well whose barrier top
  is exactly the symmetric structure the clean optimization converges to.
* ``energy_gaps.png`` -- per-molecule energy drop from the no-noise reference
  to the true minimum found by symmetry-breaking noise, coloured by the order
  of the stationary point the clean run reached (0 = real minimum / higher
  conformer, 1 = transition state, 2 = second-order saddle).

Run from this folder: ``./make_figure.py``. The methylamine geometries live in
``data/``; the per-molecule numbers are the verified results tabulated in
``README.md`` (reproduce them with ``classify_noise_minima.py``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from berny import geomlib
from berny.coords import angstrom
from berny.species_data import get_property

HARTREE_KCAL = 627.5094740631
HERE = Path(__file__).resolve().parent


def _energy(geom):
    from tblite.interface import Calculator

    numbers = np.array([int(get_property(sp, "number")) for sp in geom.species])
    pos = np.array(geom.coords) * angstrom
    calc = Calculator("GFN2-xTB", numbers, pos)
    calc.set("verbosity", 0)
    return float(calc.singlepoint().get("energy"))


def _grad(numbers, pos_bohr):
    from tblite.interface import Calculator

    calc = Calculator("GFN2-xTB", numbers, pos_bohr)
    calc.set("verbosity", 0)
    return np.array(calc.singlepoint().get("gradient"))


def _imaginary_eigenvector(geom, h=2e-3):
    """Cartesian unit displacement along the saddle's single imaginary mode.

    Builds the mass-weighted Hessian by finite differences, projects out
    translation/rotation, takes the eigenvector of the (one) negative
    eigenvalue, and returns it un-mass-weighted and normalised in Angstrom
    space. Being the antisymmetric umbrella coordinate, displacing along
    +/- it traces a symmetric double well.
    """
    numbers = np.array([int(get_property(sp, "number")) for sp in geom.species])
    masses = np.array([float(get_property(sp, "mass")) for sp in geom.species])
    pos = np.array(geom.coords) * angstrom
    n = pos.size
    flat = pos.reshape(-1)
    H = np.zeros((n, n))
    for i in range(n):
        fp, fm = flat.copy(), flat.copy()
        fp[i] += h
        fm[i] -= h
        H[i] = (
            _grad(numbers, fp.reshape(-1, 3)).reshape(-1)
            - _grad(numbers, fm.reshape(-1, 3)).reshape(-1)
        ) / (2 * h)
    H = (H + H.T) / 2
    m = np.repeat(masses, 3)
    Hmw = H / np.sqrt(np.outer(m, m))
    # project out the 6 external (trans/rot) modes
    com = (masses[:, None] * pos.reshape(-1, 3)).sum(0) / masses.sum()
    r = pos.reshape(-1, 3) - com
    nat = len(masses)
    sm = np.sqrt(m)
    ext = np.zeros((6, n))
    for k in range(3):
        t = np.zeros((nat, 3))
        t[:, k] = 1.0
        ext[k] = t.reshape(-1) * sm
    for k, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        rot = np.zeros((nat, 3))
        rot[:, a] = r[:, b]
        rot[:, b] = -r[:, a]
        ext[3 + k] = rot.reshape(-1) * sm
    q, _ = np.linalg.qr(ext.T)
    P = np.eye(n) - q @ q.T
    w, V = np.linalg.eigh(P @ Hmw @ P)
    vec_mw = V[:, int(np.argmin(w))]  # lowest (negative) mode, mass-weighted
    vec = vec_mw / sm  # back to Cartesian (bohr)
    vec = vec / np.linalg.norm(vec)
    return vec.reshape(-1, 3) / angstrom  # unit displacement, Angstrom


def inversion_doublewell():
    planar = geomlib.load(
        open(HERE / "data/methylamine_reference_min.xyz"), "xyz"
    )
    p0 = np.array(planar.coords)
    mode = _imaginary_eigenvector(planar)  # antisymmetric umbrella coordinate
    amps = np.linspace(-0.9, 0.9, 41)  # Angstrom along the mode
    e0 = _energy(planar)
    energies = np.array(
        [
            (_energy(geomlib.Geometry(planar.species, p0 + a * mode)) - e0)
            * HARTREE_KCAL
            for a in amps
        ]
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(amps, energies, "-", color="#2b6cb0", lw=2)
    for sgn in (-1, 1):
        side = energies[amps * sgn > 0]
        a_side = amps[amps * sgn > 0]
        j = int(np.argmin(side))
        ax.plot(a_side[j], side[j], "o", color="#c53030", ms=7, zorder=5)
    ax.plot(
        [],
        [],
        "o",
        color="#c53030",
        ms=7,
        label="rigid-scan turning points (relax → −6.24)",
    )
    ax.plot(
        0,
        0,
        "*",
        color="gold",
        ms=18,
        mec="k",
        mew=0.6,
        zorder=6,
        label="no-noise reference (planar saddle)",
    )
    ax.axhline(0, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("displacement along umbrella mode (Å)  ·  planar = 0")
    ax.set_ylabel("GFN2-xTB energy  (kcal/mol)")
    ax.set_title("methylamine: the clean run halts at the inversion saddle")
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(HERE / "inversion_doublewell.png", dpi=150)
    print("wrote inversion_doublewell.png")


def energy_gaps():
    # molecule, dE to true min (kcal/mol), stationary-point order of clean run
    rows = [
        ("methylamine", -6.24, 1),
        ("histidine", -1.98, 0),
        ("ethanol", -1.55, 0),
        ("mesityl_oxide", -1.30, 2),
        ("benzidine", -1.28, 2),
        ("acanil01", -0.75, 1),
        ("caffeine", -0.36, 2),
    ]
    names = [r[0] for r in rows]
    gaps = [r[1] for r in rows]
    orders = [r[2] for r in rows]
    cmap = {0: "#4a5568", 1: "#dd6b20", 2: "#c53030"}
    colors = [cmap[o] for o in orders]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(names))
    ax.barh(y, gaps, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("energy of true minimum relative to no-noise run  (kcal/mol)")
    ax.set_title("How far below the clean run the true minimum sits")
    for yi, (g, o) in enumerate(zip(gaps, orders)):
        label = {0: "minimum", 1: "saddle (1)", 2: "saddle (2)"}[o]
        ax.text(
            g - 0.05,
            yi,
            label,
            va="center",
            ha="right",
            fontsize=7,
            color="white",
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[o]) for o in (0, 1, 2)]
    ax.legend(
        handles,
        ["real minimum (conformer)", "1st-order saddle", "2nd-order saddle"],
        fontsize=8,
        loc="lower left",
    )
    fig.tight_layout()
    fig.savefig(HERE / "energy_gaps.png", dpi=150)
    print("wrote energy_gaps.png")


if __name__ == "__main__":
    inversion_doublewell()
    energy_gaps()
