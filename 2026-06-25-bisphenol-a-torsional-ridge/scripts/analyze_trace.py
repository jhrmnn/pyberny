#!/usr/bin/env python3
"""Tabulate and summarize a Berny ``trace`` JSON: where are the steps spent?

Prints a per-step table (energy above final, actual vs predicted dE, trust
radius, step size, number/lowest Hessian eigenvalue, line-search t, gradient
norms) and a one-line summary of the thrashing signature: how many steps go
uphill, how many enter negative curvature, how many are taken on the
trust-sphere, and how far the trust radius collapses.

    python analyze_trace.py ../data/xtb_trace_clean.json ../data/xtb_trace_noisy.json
"""
import json
import sys

import numpy as np

HARTREE_KCAL = 627.5094740631


def table(trace, label):
    print(f'\n===== {label} ({len(trace)} steps) =====')
    e = [r['energy'] for r in trace]
    ef = e[-1]
    hdr = ('st', 'E_rel', 'dE_act', 'dE_pred', 'ratio', 'trust', 'stepMax',
           'nNeg', 'loEv', 'sphere', 't_ls', 'gMax')
    print(('{:>3} {:>9} {:>8} {:>9} {:>6} {:>7} {:>8} {:>4} {:>9} {:>6} '
           '{:>6} {:>8}').format(*hdr))
    for i, r in enumerate(trace):
        qs = r.get('quadratic_step', {})
        tu = r.get('trust_update', {})
        ls = r.get('linear_search', {})
        crit = {c['name']: c for c in r.get('convergence', {}).get('criteria', [])}
        gmax = crit.get('Gradient maximum', {}).get('value', float('nan'))
        dep = qs.get('predicted_energy_change')
        print(('{:>3} {:>9.4f} {:>8.4f} {:>9.4f} {:>6.2f} {:>7.4f} {:>8.5f} '
               '{:>4} {:>9.4f} {:>6} {:>6.2f} {:>8.6f}').format(
            r['step'], (e[i] - ef) * HARTREE_KCAL,
            (e[i] - e[i - 1]) * HARTREE_KCAL if i else 0.0,
            dep * HARTREE_KCAL if dep is not None else float('nan'),
            tu.get('fletcher', float('nan')) if tu.get('fletcher') is not None else float('nan'),
            qs.get('trust_radius', float('nan')), qs.get('step_max', float('nan')),
            qs.get('n_negative_eigenvalues', -1), qs.get('lowest_eigenvalue', float('nan')),
            str(qs.get('on_sphere', '')), ls.get('t', float('nan')) if ls.get('t') is not None else float('nan'),
            gmax))


def summary(trace, label):
    e = np.array([r['energy'] for r in trace])
    dE = np.diff(e) * HARTREE_KCAL
    nneg = sum(1 for r in trace if r.get('quadratic_step', {}).get('n_negative_eigenvalues', 0) > 0)
    onsph = sum(1 for r in trace if r.get('quadratic_step', {}).get('on_sphere'))
    trusts = [r.get('quadratic_step', {}).get('trust_radius') for r in trace]
    last_up = max([i for i in range(len(dE)) if dE[i] > 0.1], default=-1) + 2
    print(f'{label}: {len(trace)} steps | uphill(>0.1 kcal): {(dE > 0.1).sum()} '
          f'| neg-eig steps: {nneg} | on-sphere: {onsph} '
          f'| last big-uphill @ step {last_up} | min trust: {min(trusts):.4f}')


if __name__ == '__main__':
    traces = [(p, json.load(open(p))) for p in sys.argv[1:]]
    for p, tr in traces:
        table(tr, p)
    print()
    for p, tr in traces:
        summary(tr, p)
