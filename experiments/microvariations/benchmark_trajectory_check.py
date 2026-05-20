#!/usr/bin/env python3
"""Run pyberny + MopacSolver on every molecule in the Birkholz-Schlegel and
Baker-Shajan benchmark sets with INFO-level logging captured to file, then
scan the logs for the two warning signatures that diagnosed the estradiol
pathology in ``summary.md``:

- ``Pseudoinverse gap of only: ...`` from ``src/berny/Math.py``. When this
  fires at the *final* step (the same step that declares convergence) the
  optimizer halted on a rank-deficient internal-gradient projection - the
  exact failure mode that lets pyberny report ``converged=True`` on
  estradiol ~5 kcal/mol above the true PM7 minimum.

- ``Transformation did not converge in N iterations`` from
  ``src/berny/coords.py``. This precedes the pinv warning on estradiol;
  worth flagging on its own as the Cartesian<->internal back-transform
  struggling on an ill-conditioned geometry.

Outputs (under ``--out``, default ``experiments/microvariations/benchmark_diag/``):

- ``<set>/<molecule>.log`` per molecule (raw INFO log).
- ``warnings_full.json`` - structured per-molecule record (filename
  configurable via ``--out-json``; the default avoids overwriting the
  curated, committed ``warnings.json`` snapshot).
- ``warnings.md`` - per-set Markdown summary table + a "Findings" section
  listing molecules with pinv warnings at the convergence-declaring step.
"""

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

from berny import Berny, geomlib
from berny.solvers import MopacSolver


@contextlib.contextmanager
def silence_fd(fd):
    """Redirect OS file descriptor ``fd`` to /dev/null for the duration of the block.

    Needed because ``berny.solvers.MopacSolver`` runs MOPAC with inherited
    stdout/stderr; the child writes one banner per call and we run hundreds.
    Python-level ``redirect_stdout`` doesn't help: the subprocess writes to
    the OS fd directly.
    """
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'tests' / 'data'
BENCHMARKS = {
    'birkholz': 'birkholz_schlegel',
    'baker': 'baker_shajan_2023',
}

# Pattern emitted by Math.pinv when the singular-value gap is small enough
# to log but a truncation still happened. Step prefix added by BernyAdapter.
PINV_RE = re.compile(
    r'^(\d+)\s+Pseudoinverse gap of only:\s+([0-9.eE+-]+)\s*$'
)
# Pattern from coords.py back-transform when its inner loop hit max_iter.
BACKXFORM_RE = re.compile(
    r'^(\d+)\s+Transformation did not converge in (\d+) iterations\s*$'
)
# Negative eigenvalues in the BFGS Hessian: a saddle-pass or an ill-conditioned
# Hessian. Healthy minima trajectories have all-positive eigenvalues at every
# step; nonzero counts mean the optimizer was using sphere-minimization to
# descend along an unstable mode (or that the BFGS update produced a spurious
# negative direction).
NEGEIG_RE = re.compile(
    r'^(\d+)\s+\* Number of negative eigenvalues:\s+([1-9]\d*)\s*$'
)
# RMS(dq) line of the back-transform. We flag entries where dq is large enough
# to indicate the internal-coord step couldn't faithfully represent the
# attempted Cartesian step (the disilyl_ether step-6 / maltose step-27 mode).
BACKXFORM_DQ_RE = re.compile(
    r'^(\d+)\s+\* RMS\(dcart\):\s+([0-9.eE+-]+),\s+RMS\(dq\):\s+([0-9.eE+-]+)\s*$'
)
SEVERE_BACKXFORM_DQ_THRESHOLD = 0.05  # well above the ~1e-6 a healthy back-xform produces
MAXSTEPS_RE = re.compile(r'^(\d+)\s+Maximum number of steps reached\s*$')
ALL_CRITERIA_RE = re.compile(r'^(\d+)\s+\* All criteria matched\s*$')


def _relpath(p):
    """Return ``p`` relative to the repo root if possible, else the absolute path."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def run_one(name, ref, data_dir, log_path, maxsteps):
    """Run Berny+MopacSolver on one molecule with INFO logging to ``log_path``.

    Mirrors ``scripts/benchmark.py:run_mopac`` (same ``maxsteps=110`` ceiling)
    but installs a per-molecule ``FileHandler`` on the ``berny`` logger so
    the verbose trajectory lands on disk for post-hoc scanning.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    berny_logger = logging.getLogger('berny')
    prev_level = berny_logger.level
    berny_logger.setLevel(logging.INFO)
    berny_logger.addHandler(handler)
    t0 = time.perf_counter()
    error = None
    energy = None
    converged = False
    n = 0
    try:
        with silence_fd(1), silence_fd(2):
            geom = geomlib.readfile(str(data_dir / f'{name}.xyz'))
            berny = Berny(geom, maxsteps=maxsteps)
            solver = MopacSolver(charge=ref['charge'], mult=ref['mult'])
            next(solver)
            for current in berny:
                e, g = solver.send((list(current), current.lattice))
                energy = e
                berny.send((e, g))
            converged = berny.converged
            n = berny._n
    except Exception as e:  # noqa: BLE001
        error = f'{type(e).__name__}: {e}'
    finally:
        handler.close()
        berny_logger.removeHandler(handler)
        berny_logger.setLevel(prev_level)
    return {
        'converged': converged,
        'steps': n,
        'energy': energy,
        'wall': time.perf_counter() - t0,
        'error': error,
    }


def scan_log(log_path):
    """Extract warning flags and the convergence-declaring step from a log file.

    Returns a dict with:

    - ``pinv_warnings``: list of ``{step, gap}`` for every Math.pinv warning.
    - ``backtransform_warnings``: list of ``{step, iterations}`` for every
      back-transform failure.
    - ``maxsteps_reached``: bool.
    - ``convergence_step``: int or None - the step number of the line
      ``* All criteria matched``. ``None`` if the optimizer never converged.
    """
    pinv = []
    backx = []
    negeig = []
    severe_backx_dq = []
    maxs = False
    final_step = None
    with open(log_path) as fh:
        for line in fh:
            m = PINV_RE.match(line)
            if m:
                pinv.append({'step': int(m.group(1)), 'gap': float(m.group(2))})
                continue
            m = BACKXFORM_RE.match(line)
            if m:
                backx.append({'step': int(m.group(1)), 'iterations': int(m.group(2))})
                continue
            m = NEGEIG_RE.match(line)
            if m:
                negeig.append({'step': int(m.group(1)), 'count': int(m.group(2))})
                continue
            m = BACKXFORM_DQ_RE.match(line)
            if m:
                dq = float(m.group(3))
                if dq >= SEVERE_BACKXFORM_DQ_THRESHOLD:
                    severe_backx_dq.append(
                        {
                            'step': int(m.group(1)),
                            'dcart': float(m.group(2)),
                            'dq': dq,
                        }
                    )
                continue
            if MAXSTEPS_RE.match(line):
                maxs = True
                continue
            m = ALL_CRITERIA_RE.match(line)
            if m:
                final_step = int(m.group(1))
    return {
        'pinv_warnings': pinv,
        'backtransform_warnings': backx,
        'negative_eigenvalue_events': negeig,
        'severe_backxform_dq': severe_backx_dq,
        'maxsteps_reached': maxs,
        'convergence_step': final_step,
    }


def render_markdown(records, ref_steps):
    """Render the per-set table and Findings section."""
    out = []
    out.append('# Benchmark trajectory warning sweep\n')
    out.append(
        'For every molecule in the two MOPAC benchmark sets, '
        '`Berny(geom, maxsteps=110)` + `MopacSolver(charge, mult)` ran '
        'with INFO-level logging captured to `<set>/<molecule>.log`. The '
        'columns below summarise the warning lines that fired during the '
        'trajectory. `pinv@final` is the dangerous case discovered on '
        'estradiol: a `Pseudoinverse gap of only:` warning at the same '
        'step that declared convergence means pyberny halted on a '
        'rank-deficient internal-gradient projection.\n'
    )

    findings_pinv_at_final = []
    findings_pinv_anywhere = []
    findings_backx_heavy = []
    findings_severe_dq = []
    findings_negeig = []
    findings_maxsteps = []

    for set_name, set_dir in BENCHMARKS.items():
        out.append(f'\n## {set_dir}\n')
        out.append(
            '| molecule | ref | steps | conv | pinv | pinv@final '
            '| back-xform | neg-eig steps | severe dq | maxsteps |'
        )
        out.append('|---|---:|---:|---|---:|---|---:|---:|---:|---|')
        rows = [r for r in records if r['set'] == set_name]
        for r in sorted(rows, key=lambda x: x['molecule']):
            mol = r['molecule']
            ref = ref_steps.get((set_name, mol))
            ref_s = '-' if ref is None else str(ref)
            scan = r['scan']
            pinv = scan['pinv_warnings']
            backx = scan['backtransform_warnings']
            negeig = scan['negative_eigenvalue_events']
            severe_dq = scan['severe_backxform_dq']
            steps_s = '-' if r['error'] else str(r['steps'])
            conv_s = (
                f'ERR ({r["error"]})'
                if r['error']
                else ('yes' if r['converged'] else 'no')
            )
            final = scan['convergence_step']
            pinv_at_final = bool(pinv) and final is not None and pinv[-1]['step'] == final
            pinv_at_final_s = 'YES' if pinv_at_final else ('-' if not pinv else 'no')
            maxs_s = 'yes' if scan['maxsteps_reached'] else '-'
            out.append(
                f'| {mol} | {ref_s} | {steps_s} | {conv_s} | {len(pinv)} | '
                f'{pinv_at_final_s} | {len(backx)} | {len(negeig)} | '
                f'{len(severe_dq)} | {maxs_s} |'
            )
            if pinv_at_final:
                findings_pinv_at_final.append((set_name, mol, pinv[-1]['gap']))
            elif pinv:
                findings_pinv_anywhere.append((set_name, mol, len(pinv)))
            if len(backx) >= 3:
                findings_backx_heavy.append((set_name, mol, len(backx)))
            if severe_dq:
                findings_severe_dq.append((set_name, mol, severe_dq))
            if negeig:
                findings_negeig.append((set_name, mol, negeig))
            if scan['maxsteps_reached']:
                findings_maxsteps.append((set_name, mol))

    out.append('\n## Findings\n')
    if findings_pinv_at_final:
        out.append(
            '### Pseudoinverse warning at the convergence-declaring step '
            '(estradiol-style)\n'
        )
        out.append(
            'These are the cases where `converged=True` was returned even '
            'though `Math.pinv` had silently zeroed a singular direction at '
            'the final step:\n'
        )
        for set_name, mol, gap in findings_pinv_at_final:
            out.append(f'- **{set_name}/{mol}** - final-step pinv gap `{gap:.2e}`')
        out.append('')
    else:
        out.append(
            'No molecule outside estradiol\'s perturbed variants triggered '
            'a final-step `Pseudoinverse gap of only:` warning on its '
            'published starting geometry.\n'
        )

    if findings_pinv_anywhere:
        out.append('### Pseudoinverse warning during the trajectory but not at the final step\n')
        for set_name, mol, n in findings_pinv_anywhere:
            out.append(f'- {set_name}/{mol}: {n} pinv warning(s) at intermediate steps')
        out.append('')

    if findings_backx_heavy:
        out.append('### Heavy back-transform struggle (>=3 warnings)\n')
        out.append(
            'On estradiol the precursor to the pinv catastrophe was three '
            'consecutive `Transformation did not converge in 20 iterations` '
            'lines. Molecules below show the same precursor signature:\n'
        )
        for set_name, mol, n in findings_backx_heavy:
            out.append(f'- {set_name}/{mol}: {n} back-transform warning(s)')
        out.append('')

    if findings_severe_dq:
        out.append('### Severe back-transform step blow-up (RMS(dq) >= 0.05)\n')
        out.append(
            'These steps reported a back-transform with internal-coord '
            'RMS displacement at least an order of magnitude above what a '
            'well-behaved Cartesian<->internal mapping produces. They are '
            'often saddle-pass attempts on rigid macrocycles or near-linear '
            'angles; if they coincide with `pinv@final = YES` they are the '
            'mechanism by which the pinv pathology fires.\n'
        )
        for set_name, mol, events in findings_severe_dq:
            steps = ','.join(
                f'{e["step"]} (dq={e["dq"]:.2g})' for e in events
            )
            out.append(f'- {set_name}/{mol}: {steps}')
        out.append('')

    if findings_negeig:
        out.append('### Negative-eigenvalue events (sphere-minimization saddle passes)\n')
        out.append(
            'Healthy minimum-finding trajectories report all-positive BFGS '
            'Hessian eigenvalues at every step. Negative-eigenvalue events '
            'happen when the BFGS update produces a spurious unstable mode '
            '(or when the geometry really is near a saddle). Pyberny then '
            'switches to RFO sphere-minimization to descend along that '
            'unstable direction. Occasional events are normal; a sustained '
            'count is a sign of a confusing PES region:\n'
        )
        for set_name, mol, events in findings_negeig:
            steps = ','.join(str(e['step']) for e in events)
            out.append(f'- {set_name}/{mol}: {len(events)} event(s) at step(s) {steps}')
        out.append('')

    if findings_maxsteps:
        out.append('### Hit maxsteps ceiling\n')
        for set_name, mol in findings_maxsteps:
            out.append(f'- {set_name}/{mol}')
        out.append('')

    return '\n'.join(out) + '\n'


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        '--out',
        type=Path,
        default=REPO_ROOT / 'experiments' / 'microvariations' / 'benchmark_diag',
    )
    ap.add_argument(
        '--benchmarks',
        nargs='+',
        choices=sorted(BENCHMARKS),
        default=sorted(BENCHMARKS),
    )
    ap.add_argument('--molecules', nargs='+', default=None)
    ap.add_argument('--maxsteps', type=int, default=110)
    ap.add_argument(
        '--out-json',
        default='warnings_full.json',
        help=(
            'filename (relative to --out) for the JSON records file. '
            "Defaults to 'warnings_full.json' so the curated, committed "
            "'warnings.json' snapshot is not overwritten by a fresh run."
        ),
    )
    ap.add_argument(
        '--out-md',
        default='warnings.md',
        help='filename (relative to --out) for the rendered Markdown report.',
    )
    ap.add_argument(
        '--resume',
        action='store_true',
        help='reuse <molecule>.log files that already exist',
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not shutil.which('mopac'):
        raise SystemExit('mopac not on PATH')
    args.out.mkdir(parents=True, exist_ok=True)

    plan = []
    ref_steps = {}
    for set_name in args.benchmarks:
        set_dir = BENCHMARKS[set_name]
        data_dir = DATA_ROOT / set_dir
        reference = json.loads((data_dir / 'reference.json').read_text())
        for name in sorted(reference):
            if args.molecules and name not in args.molecules:
                continue
            plan.append((set_name, set_dir, data_dir, name, reference[name]))
            ref_steps[(set_name, name)] = reference[name].get('mopac_pm7_steps')

    records = []
    t_start = time.perf_counter()
    for i, (set_name, set_dir, data_dir, name, ref) in enumerate(plan, 1):
        log_path = args.out / set_dir / f'{name}.log'
        if args.resume and log_path.exists() and log_path.stat().st_size > 0:
            # Recover step/convergence info purely from the log
            scan = scan_log(log_path)
            with open(log_path) as fh:
                lines = fh.readlines()
            # Step prefix of the last numbered line is the step count;
            # converged iff `* All criteria matched` appeared.
            last_step = 0
            for line in lines:
                m = re.match(r'^(\d+)\s+', line)
                if m:
                    last_step = max(last_step, int(m.group(1)))
            rec_run = {
                'converged': scan['convergence_step'] is not None,
                'steps': last_step,
                'energy': None,  # not parsed from log
                'wall': None,
                'error': None,
                'cached': True,
            }
            elapsed = time.perf_counter() - t_start
            print(
                f'[{i}/{len(plan)}] {set_name}/{name}: cached '
                f'(steps={last_step}, conv={rec_run["converged"]}) '
                f'(total {elapsed:.0f}s)',
                flush=True,
            )
        else:
            rec_run = run_one(name, ref, data_dir, log_path, args.maxsteps)
            scan = scan_log(log_path)
            rec_run['cached'] = False
            elapsed = time.perf_counter() - t_start
            print(
                f'[{i}/{len(plan)}] {set_name}/{name}: '
                f'converged={rec_run["converged"]} steps={rec_run["steps"]} '
                f'pinv={len(scan["pinv_warnings"])} '
                f'backx={len(scan["backtransform_warnings"])} '
                f'wall={rec_run["wall"]:.1f}s (total {elapsed:.0f}s)',
                flush=True,
            )

        records.append(
            {
                'set': set_name,
                'set_dir': set_dir,
                'molecule': name,
                **rec_run,
                'scan': scan,
                'log_path': _relpath(log_path),
            }
        )

        # Incremental persistence so a crash doesn't lose finished cells
        (args.out / args.out_json).write_text(
            json.dumps({'records': records, 'ref_steps': {
                f'{s}/{m}': v for (s, m), v in ref_steps.items()
            }}, indent=2)
        )

    markdown = render_markdown(records, ref_steps)
    (args.out / args.out_md).write_text(markdown)
    print('\n' + markdown)
    return 0


if __name__ == '__main__':
    sys.exit(main())
