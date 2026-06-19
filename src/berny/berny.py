# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator, Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy import dot, eye
from numpy.linalg import norm
from numpy.typing import NDArray

from . import Math
from .coords import InternalCoords
from .geomlib import Geometry

__all__ = ['Berny', 'BernyParams']

log = logging.getLogger(__name__)

FloatArray = NDArray[np.floating[Any]]


@dataclass
class BernyParams:
    """Tunable parameters for :class:`Berny`.

    Attributes:
        gradientmax, gradientrms, stepmax, steprms: convergence criteria in
            atomic units (``step`` refers to the step in internal coordinates,
            assuming radian units for angles).
        trust: initial trust radius in atomic units; the maximum RMS of the
            quadratic step.
        energy_noise: estimated absolute precision (a.u.) of one energy
            evaluation; used to suppress trust-region updates from noisy
            ``dE/dE_predicted`` ratios.
        dihedral: whether to form dihedral angles.
        superweakdih: whether to form dihedral angles containing two or more
            noncovalent bonds.
    """

    gradientmax: float = 0.45e-3
    gradientrms: float = 0.15e-3
    stepmax: float = 1.8e-3
    steprms: float = 1.2e-3
    trust: float = 0.3
    energy_noise: float = 2e-8
    dihedral: bool = True
    superweakdih: bool = False


class OptPoint(NamedTuple):
    # E and g are None for ``future``/``predicted`` points whose energy or
    # gradient haven't been computed yet, and a float/ndarray otherwise.
    q: FloatArray
    E: float | None
    g: FloatArray | None


@dataclass
class BernyState:
    """Mutable optimizer state. Captured/restored via the ``debug``/``restart`` API."""

    geom: Geometry
    params: BernyParams
    trust: float
    coords: InternalCoords
    H: FloatArray
    weights: FloatArray
    future: OptPoint
    first: bool = True
    interpolated: OptPoint | None = None
    predicted: OptPoint | None = None
    previous: OptPoint | None = None
    best: OptPoint | None = None


class BernyAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger, {})
        self.step: int = 0

    def process(self, msg: Any, kwargs: Any) -> tuple[str, Any]:
        return f'{self.step} {msg}', kwargs


class Berny(Generator):  # type: ignore[type-arg]
    """Generator that receives energy and gradients and yields the next geometry.

    Args:
        geom: geometry to start with
        debug: if :data:`True`, the generator yields debug info on receiving
            the energy and gradients, otherwise it yields :data:`None`
        restart: state captured from a previous run with ``debug=True``
        maxsteps: abort after maximum number of steps
        logger: alternative logger to use
        trace: optional path to a JSON file. When given, a structured
            dict-like record is captured for every optimization step
            (mirroring the textual log output) and the full list of
            per-step records is written to the file after each step, so
            partial progress survives an interrupted run.
        params: parameter overrides — see :class:`BernyParams`

    The Berny object is to be used as follows::

        optimizer = Berny(geom)
        for geom in optimizer:
            # calculate energy and gradients (as N-by-3 matrix)
            debug = optimizer.send((energy, gradients))
    """

    def __init__(
        self,
        geom: Geometry,
        debug: bool = False,
        restart: dict[str, Any] | None = None,
        maxsteps: int = 100,
        logger: logging.Logger | None = None,
        trace: str | os.PathLike[str] | None = None,
        **params: Any,
    ) -> None:
        self._debug = debug
        self._maxsteps = maxsteps
        self._converged = False
        self._n = 0
        self._log = BernyAdapter(logger or log)
        self._trace_path: Path | None = Path(trace) if trace is not None else None
        self._trace: list[dict[str, Any]] = []
        if restart:
            self._state = BernyState(**restart)
            return
        bparams = BernyParams(**params)
        coords, H, weights, future = self._build_coord_state(geom, bparams)
        self._state = BernyState(
            geom=geom,
            params=bparams,
            trust=bparams.trust,
            coords=coords,
            H=H,
            weights=weights,
            future=future,
        )

    def _build_coord_state(
        self, geom: Geometry, params: BernyParams
    ) -> tuple[InternalCoords, FloatArray, FloatArray, OptPoint]:
        """Build ``InternalCoords`` and the coord-derived state for ``geom``.

        Returns ``(coords, H, weights, future)`` and logs ``str(coords)``.
        """
        coords = InternalCoords(
            geom, dihedral=params.dihedral, superweakdih=params.superweakdih
        )
        for line in str(coords).split('\n'):
            self._log.info(line)
        return (
            coords,
            coords.hessian_guess(geom),
            coords.weights(geom),
            OptPoint(coords.eval_geom(geom), None, None),
        )

    def __next__(self) -> Geometry:
        assert self._n <= self._maxsteps
        if self._n >= self._maxsteps or self._converged:
            raise StopIteration
        self._n += 1
        return self._state.geom

    @property
    def trust(self) -> float:
        """Current trust radius."""
        return self._state.trust

    @property
    def converged(self) -> bool:
        """Whether the optimized has converged."""
        return self._converged

    def send(  # type: ignore[override]
        self, energy_and_gradients: tuple[float, Any]
    ) -> dict[str, Any] | None:
        self._log.step = self._n
        log, s = self._log.info, self._state
        record: dict[str, Any] | None
        if self._trace_path is not None:
            record = {'step': self._n}
        else:
            record = None
        energy, gradients = energy_and_gradients
        gradients = np.array(gradients)
        log(f'Energy: {energy:.12}')
        if record is not None:
            record['energy'] = float(energy)
        # C2: adaptive coordinate rebuild. If an sp-like triple has crossed
        # the linear-bend threshold (175° / 170° hysteresis) since the coord
        # set was last built, rebuild now — *before* computing B and the
        # current q — so that this iteration runs entirely in the new
        # q-space. We carry over the old Hessian block for coordinates that
        # survive the rebuild (same coord type + atom indices) and keep the
        # diagonal guess for genuinely new coordinates. ``first=True`` still
        # short-circuits the next iteration's
        # update_hessian/linear_search/update_trust calls, and the q-space
        # history (best/previous/predicted/interpolated) is dropped because
        # those points live in the old coordinate space. We skip the check on
        # the first iteration since coords were just built from this geometry.
        coord_rebuild = False
        if not s.first and s.coords.needs_rebuild(s.geom):
            log('Linear-bend topology changed; rebuilding internal coordinates')
            old_coords, old_H = s.coords, s.H
            s.coords, s.H, s.weights, s.future = self._build_coord_state(
                s.geom, s.params
            )
            s.H = _carry_over_hessian(old_coords, old_H, s.coords, s.H)
            s.first = True
            s.interpolated = None
            s.predicted = None
            s.previous = None
            s.best = None
            coord_rebuild = True
        if record is not None:
            record['coord_rebuild'] = coord_rebuild
        B = s.coords.B_matrix(s.geom)
        B_inv = B.T.dot(Math.pinv(np.dot(B, B.T), log=log))
        current = OptPoint(s.future.q, energy, dot(B_inv.T, gradients.reshape(-1)))
        assert current.E is not None
        assert current.g is not None
        if not s.first:
            assert s.best is not None
            assert s.best.E is not None
            assert s.best.g is not None
            assert s.previous is not None
            assert s.previous.E is not None
            assert s.predicted is not None
            assert s.predicted.E is not None
            assert s.interpolated is not None
            assert s.interpolated.E is not None
            s.H = update_hessian(
                s.H, current.q - s.best.q, current.g - s.best.g, log=log, record=record
            )
            s.trust = update_trust(
                s.trust,
                current.E - s.previous.E,  # or should it be s.interpolated.E?
                s.predicted.E - s.interpolated.E,
                s.predicted.q - s.interpolated.q,
                log=log,
                energy_noise=s.params.energy_noise,
                record=record,
            )
            dq: FloatArray = s.best.q - current.q
            t, E = linear_search(
                current.E,
                s.best.E,
                float(dot(current.g, dq)),
                float(dot(s.best.g, dq)),
                log=log,
                record=record,
            )
            s.interpolated = OptPoint(
                current.q + t * dq, E, current.g + t * (s.best.g - current.g)
            )
        else:
            s.interpolated = current
        if s.trust < 1e-6:
            raise RuntimeError('The trust radius got too small, check forces?')
        proj = dot(B, B_inv)
        H_proj = proj.dot(s.H).dot(proj) + 1000 * (eye(len(s.coords)) - proj)
        assert s.interpolated.g is not None
        # ``on_sphere`` (whether the step hit the trust boundary) is recorded
        # in the trace by ``quadratic_step`` itself; it no longer gates
        # convergence, so it is not needed here.
        dq, dE, _on_sphere = quadratic_step(
            dot(proj, s.interpolated.g),
            H_proj,
            s.weights,
            s.trust,
            log=log,
            record=record,
        )
        assert s.interpolated.E is not None
        s.predicted = OptPoint(s.interpolated.q + dq, s.interpolated.E + dE, None)
        dq = s.predicted.q - current.q
        rms_dq = Math.rms(dq)
        log(f'Total step: RMS: {rms_dq:.3}, max: {max(abs(dq)):.3}')
        if record is not None:
            record['total_step'] = {
                'rms': float(rms_dq) if rms_dq is not None else None,
                'max': float(max(abs(dq))),
            }
        q, s.geom = s.coords.update_geom(
            s.geom, current.q, s.predicted.q - current.q, B_inv, log=log
        )
        s.future = OptPoint(q, None, None)
        s.previous = current
        if s.first or (
            s.best is not None and s.best.E is not None and current.E < s.best.E
        ):
            s.best = current
        s.first = False
        self._converged = is_converged(
            current.g,
            s.future.q - current.q,
            s.params,
            log=log,
            record=record,
        )
        max_steps_reached = self._n == self._maxsteps
        if max_steps_reached:
            log('Maximum number of steps reached')
        if record is not None:
            record['converged'] = bool(self._converged)
            record['max_steps_reached'] = bool(max_steps_reached)
            self._trace.append(record)
            self._dump_trace()
        if self._debug:
            return {f.name: getattr(s, f.name) for f in fields(s)}
        return None

    def _dump_trace(self) -> None:
        """Atomically write the accumulated trace list to ``self._trace_path``.

        Writes to a sibling ``*.tmp`` file then ``os.replace``s into place,
        so a crash mid-write can't leave a half-written / unparseable JSON
        file in the artifact.
        """
        assert self._trace_path is not None
        tmp = self._trace_path.with_suffix(self._trace_path.suffix + '.tmp')
        tmp.write_text(json.dumps(self._trace, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp, self._trace_path)

    def throw(self, *args: Any, **kwargs: Any) -> Any:
        return Generator.throw(self, *args, **kwargs)


def no_log(msg: str, **kwargs: Any) -> None:
    pass


def update_hessian(
    H: FloatArray,
    dq: FloatArray,
    dg: FloatArray,
    log: Any = no_log,
    *,
    record: dict[str, Any] | None = None,
) -> FloatArray:
    dH1 = dg[None, :] * dg[:, None] / dot(dq, dg)
    dH2 = H.dot(dq[None, :] * dq[:, None]).dot(H) / dq.dot(H).dot(dq)
    dH = dH1 - dH2  # BFGS update
    log('Hessian update information:')
    rms_dH = Math.rms(dH)
    log(f'* Change: RMS: {rms_dH:.3}, max: {abs(dH).max():.3}')
    if record is not None:
        record['hessian_update'] = {
            'rms_change': float(rms_dH) if rms_dH is not None else None,
            'max_change': float(abs(dH).max()),
        }
    result: FloatArray = H + dH
    return result


def _carry_over_hessian(
    old_coords: Iterable[object],
    old_H: FloatArray,
    new_coords: Iterable[object],
    guess_H: FloatArray,
) -> FloatArray:
    """Seed a rebuilt Hessian from the previous one."""
    old_pos = {coord: i for i, coord in enumerate(old_coords)}
    pairs = [
        (new_i, old_pos[coord])
        for new_i, coord in enumerate(new_coords)
        if coord in old_pos
    ]
    H = guess_H.copy()
    if pairs:
        new_idx, old_idx = zip(*pairs)
        new_idx_arr = np.array(new_idx, dtype=np.int64)
        old_idx_arr = np.array(old_idx, dtype=np.int64)
        H[np.ix_(new_idx_arr, new_idx_arr)] = old_H[np.ix_(old_idx_arr, old_idx_arr)]
    result: FloatArray = H
    return result


def update_trust(
    trust: float,
    dE: float,
    dE_predicted: float,
    dq: FloatArray,
    log: Any = no_log,
    *,
    energy_noise: float = 2e-8,
    record: dict[str, Any] | None = None,
) -> float:
    if abs(dE_predicted) < 10 * energy_noise:
        if abs(norm(dq) - trust) < 1e-10:
            new_trust = 2 * trust
        else:
            new_trust = trust
        if record is not None:
            record['trust_update'] = {
                'fletcher': None,
                'trust': float(new_trust),
                'below_noise': True,
            }
        return new_trust
    if dE != 0:
        r = dE / dE_predicted  # Fletcher's parameter
    else:
        r = 1.0
    log(f"Trust update: Fletcher's parameter: {r:.3}")
    if r < 0.25:
        new_trust = norm(dq) / 4
    elif r > 0.75 and abs(norm(dq) - trust) < 1e-10:
        new_trust = 2 * trust
    else:
        new_trust = trust
    if record is not None:
        record['trust_update'] = {
            'fletcher': float(r),
            'trust': float(new_trust),
            'below_noise': False,
        }
    return new_trust


def linear_search(
    E0: float,
    E1: float,
    g0: float,
    g1: float,
    log: Any = no_log,
    *,
    record: dict[str, Any] | None = None,
) -> tuple[float, float]:
    log('Linear interpolation:')
    log(f'* Energies: {E0:.8}, {E1:.8}')
    log(f'* Derivatives: {g0:.3}, {g1:.3}')
    t, E = Math.fit_quartic(E0, E1, g0, g1)
    if t is None or t < -1 or t > 2:
        t, E = Math.fit_cubic(E0, E1, g0, g1)
        if t is None or t < 0 or t > 1:
            if E0 <= E1:
                log('* No fit succeeded, staying in new point')
                if record is not None:
                    record['linear_search'] = {
                        'E0': float(E0),
                        'E1': float(E1),
                        'g0': float(g0),
                        'g1': float(g1),
                        'method': 'none-new',
                        't': 0.0,
                        'interpolated_energy': float(E0),
                    }
                return 0, E0

            log('* No fit succeeded, returning to best point')
            if record is not None:
                record['linear_search'] = {
                    'E0': float(E0),
                    'E1': float(E1),
                    'g0': float(g0),
                    'g1': float(g1),
                    'method': 'none-best',
                    't': 1.0,
                    'interpolated_energy': float(E1),
                }
            return 1, E1
        msg = 'Cubic interpolation was performed'
        method = 'cubic'
    else:
        msg = 'Quartic interpolation was performed'
        method = 'quartic'
    assert E is not None
    log(f'* {msg}: t = {t:.3}')
    log(f'* Interpolated energy: {E:.8}')
    if record is not None:
        record['linear_search'] = {
            'E0': float(E0),
            'E1': float(E1),
            'g0': float(g0),
            'g1': float(g1),
            'method': method,
            't': float(t),
            'interpolated_energy': float(E),
        }
    return t, E


def quadratic_step(
    g: FloatArray,
    H: FloatArray,
    w: FloatArray,
    trust: float,
    log: Any = no_log,
    *,
    record: dict[str, Any] | None = None,
) -> tuple[FloatArray, float, bool]:
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    rfo = np.vstack((np.hstack((H, g[:, None])), np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo + rfo.T) / 2)
    dq = V[:-1, 0] / V[-1, 0]
    l = D[0]
    if norm(dq) <= trust:
        log('Pure RFO step was performed:')
        on_sphere = False
        step_type = 'rfo'
    else:

        def steplength(l: float) -> float:
            return float(norm(np.linalg.solve(l * eye(H.shape[0]) - H, g)) - trust)

        l = Math.findroot(steplength, ev[0])  # minimization on sphere
        dq = np.linalg.solve(l * eye(H.shape[0]) - H, g)
        on_sphere = True
        step_type = 'sphere'
        log('Minimization on sphere was performed:')
    dE = dot(g, dq) + 0.5 * dq.dot(H).dot(dq)  # predicted energy change
    log(f'* Trust radius: {trust:.2}')
    log(f'* Number of negative eigenvalues: {(ev < 0).sum()}')
    log(f'* Lowest eigenvalue: {ev[0]:.3}')
    log(f'* lambda: {l:.3}')
    rms_dq = Math.rms(dq)
    log(f'Quadratic step: RMS: {rms_dq:.3}, max: {max(abs(dq)):.3}')
    log(f'* Predicted energy change: {dE:.3}')
    if record is not None:
        record['quadratic_step'] = {
            'step_type': step_type,
            'on_sphere': bool(on_sphere),
            'trust_radius': float(trust),
            'n_negative_eigenvalues': int((ev < 0).sum()),
            'lowest_eigenvalue': float(ev[0]),
            'lambda': float(l),
            'step_rms': float(rms_dq) if rms_dq is not None else None,
            'step_max': float(max(abs(dq))),
            'predicted_energy_change': float(dE),
        }
    return dq, float(dE), on_sphere


def is_converged(
    forces: FloatArray,
    step: FloatArray,
    params: BernyParams,
    log: Any = no_log,
    *,
    record: dict[str, Any] | None = None,
) -> bool:
    # The four standard-method (Gaussian-style) criteria; all must hold
    # simultaneously. ``step`` is the actual displacement taken, so when the
    # quadratic step was truncated to the trust sphere the displacement
    # criteria are tested against that trust-limited step rather than being
    # hard-blocked: a sphere-restricted step at a (noisy, flat) minimum still
    # converges once the trust radius has shrunk below the step thresholds.
    # See ``doc/standard_method.rst`` and issue #129.
    criteria: list[tuple[str, Any, float]] = [
        ('Gradient RMS', Math.rms(forces), params.gradientrms),
        ('Gradient maximum', np.max(abs(forces)), params.gradientmax),
        ('Step RMS', Math.rms(step), params.steprms),
        ('Step maximum', np.max(abs(step)), params.stepmax),
    ]
    log('Convergence criteria:')
    all_matched = True
    crit_records: list[dict[str, Any]] = []
    for name, value, threshold in criteria:
        result = value < threshold
        op = '<' if result else '>'
        log(f'* {name}: {value:.3} {op} {threshold:.3} => {"OK" if result else "no"}')
        crit_records.append(
            {
                'name': name,
                'value': float(value),
                'threshold': float(threshold),
                'matched': bool(result),
            }
        )
        if not result:
            all_matched = False
    if all_matched:
        log('* All criteria matched')
    if record is not None:
        record['convergence'] = {
            'criteria': crit_records,
            'all_matched': bool(all_matched),
        }
    return all_matched
