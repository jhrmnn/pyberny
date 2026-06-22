#!/usr/bin/env python3
"""Compare symmetry=None vs symmetry='break' across the xtb benchmarks."""
import json
import sys
import warnings

from berny import Berny, geomlib
from berny.benchmarks import load_reference, require_geometries
from berny.symmetry import detect_point_group
from berny.solvers import XTBSolver

HARTREE_KCAL = 627.50947


def optimize(geom, charge, mult, symmetry):
    berny = Berny(geom, symmetry=symmetry)
    solver = XTBSolver(charge=charge, mult=mult)
    next(solver)
    energies = []
    for g in berny:
        e, grad = solver.send((list(g), g.lattice))
        energies.append(e)
        berny.send((e, grad))
    return berny.converged, berny._n, energies[-1] if energies else None


def main():
    benches = sys.argv[1:] or ["baker", "birkholz", "oligomers"]
    out = {}
    for bench in benches:
        data_dir = require_geometries(bench)
        reference = load_reference(bench)
        names = sorted(reference)
        rows = []
        for name in names:
            ref = reference[name]
            fname = ref.get("file", f"{name}.xyz")
            geom = geomlib.readfile(str(data_dir / fname))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pg = detect_point_group(geom)[0]
            row = {
                "name": name,
                "atoms": ref["atoms"],
                "pg": pg,
                "ref_steps": ref.get("xtb_gfn2_steps"),
            }
            for sym, key in [(None, "none"), ("break", "break")]:
                try:
                    conv, n, e = optimize(
                        geomlib.readfile(str(data_dir / fname)),
                        ref["charge"],
                        ref["mult"],
                        sym,
                    )
                    row[f"{key}_conv"] = conv
                    row[f"{key}_steps"] = n
                    row[f"{key}_E"] = e
                except Exception as ex:
                    row[f"{key}_conv"] = False
                    row[f"{key}_steps"] = None
                    row[f"{key}_E"] = None
                    row[f"{key}_err"] = f"{type(ex).__name__}: {ex}"
            de = None
            if row.get("none_E") is not None and row.get("break_E") is not None:
                de = (row["break_E"] - row["none_E"]) * HARTREE_KCAL
            row["dE_kcal"] = de
            rows.append(row)
            print(
                f"{bench:9s} {name:28s} {pg:5s} "
                f"none={row.get('none_steps')}/{row.get('none_conv')} "
                f"break={row.get('break_steps')}/{row.get('break_conv')} "
                f"dE={de:+.3f}" if de is not None else
                f"{bench:9s} {name:28s} {pg:5s} "
                f"none={row.get('none_steps')}/{row.get('none_conv')} "
                f"break={row.get('break_steps')}/{row.get('break_conv')} dE=NA",
                flush=True,
            )
        out[bench] = rows
    with open("/tmp/sym_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
