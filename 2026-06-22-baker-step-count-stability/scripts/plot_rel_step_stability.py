#!/usr/bin/env python3
"""Plot per-system (RMSD-relative) step-count stability.

Left:  per-molecule step-count inflation (median steps / clean steps) under
       fixed sigma = 0.05 A vs per-system noise = 20 % of the start->minimum
       RMSD, sorted by the relative-noise value. Molecules that stay inflated
       under relative noise (the planar conjugated / pi systems) are coloured.
Right: per-molecule coefficient of variation of step count under 20 % relative
       noise.

Usage: plot_rel_step_stability.py rel_step_stability.json step_stability.json out.png
"""

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Molecules whose step count stays inflated (>1.5x) under 20% relative noise --
# the planar conjugated / pi (aromatic + cumulene) systems whose soft
# out-of-plane modes take a roughly amplitude-independent number of steps to
# damp. Used only for colouring.
CONJUGATED = {
    "benzene",
    "difluorobenzene_13",
    "difluoronaphthalene_15",
    "naphthalene",
    "difuropyrazine",
    "furan",
    "benzaldehyde",
    "trifluorobenzene_135",
    "pterin",
    "allene",
}


def main(rel_file, fix_file, outfile="relative_noise_step_stability.png"):
    rel = json.loads(Path(rel_file).read_text())
    fix = json.loads(Path(fix_file).read_text())

    def infl_rel(n):
        return st.median(rel[n]["by_frac"]["0.2"]["steps"]) / rel[n]["clean_steps"]

    def infl_fix(n):
        return st.median(fix[n]["by_sigma"]["0.05"]["steps"]) / fix[n]["clean_steps"]

    mols = sorted(rel, key=infl_rel)
    y = range(len(mols))
    colors = ["tab:red" if n in CONJUGATED else "tab:blue" for n in mols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Panel A: inflation, fixed vs relative.
    for i, n in enumerate(mols):
        ax1.plot([infl_fix(n), infl_rel(n)], [i, i], color="lightgray", zorder=1)
    ax1.scatter(
        [infl_fix(n) for n in mols],
        list(y),
        marker="x",
        color="gray",
        s=35,
        label="fixed σ = 0.05 Å",
        zorder=2,
    )
    ax1.scatter(
        [infl_rel(n) for n in mols],
        list(y),
        color=colors,
        s=55,
        label="per-system 20% of R$_{sm}$",
        zorder=3,
    )
    ax1.axvline(1.0, color="k", lw=0.8, ls=":")
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(mols, fontsize=8)
    ax1.set_xlabel("step-count inflation (median steps / clean steps)")
    ax1.set_title(
        "Inflation: fixed vs per-system noise\n(red = planar conjugated/π systems)"
    )
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # Panel B: CV under 20% relative noise.
    cvs = []
    for n in mols:
        s = rel[n]["by_frac"]["0.2"]["steps"]
        cvs.append(100 * st.pstdev(s) / st.mean(s) if len(s) > 1 else 0)
    ax2.barh(list(y), cvs, color=colors)
    ax2.axvline(25, color="k", lw=0.8, ls=":")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(mols, fontsize=8)
    ax2.set_xlabel("step-count CV under 20% relative noise (%)")
    ax2.set_title("Dispersion under per-system 20% noise\n(dotted line = 25%)")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Per-system noise (RMSD = 20% of start→minimum distance): step-count stability\n"
        "(Baker set, GFN2-xTB, frustrated-reference molecules excluded)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outfile, dpi=130)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main(*sys.argv[1:])
