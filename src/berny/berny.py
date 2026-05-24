# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass, fields
from typing import Any, NamedTuple

import numpy as np
from numpy import dot, eye
from numpy.linalg import norm

from . import Math
from .coords import InternalCoords
from .geomlib import Geometry

__all__ = ['Berny', 'BernyParams']

log = logging.getLogger(__name__)


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
    q: np.ndarray
    E: Any
    g: Any


@dataclass
class BernyState:
    """Mutable optimizer state. Captured/restored via the ``debug``/``restart`` API."""

    geom: Geometry
    params: BernyParams
    trust: float
    coords: InternalCoords
    H: np.ndarray
    weights: np.ndarray
    future: OptPoint
    first: bool = True
    interpolated: OptPoint | None = None
    predicted: OptPoint | None = None
    previous: OptPoint | None = None
    best: OptPoint | None = None


class BernyAdapter(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger, {})
        self.step: int = 0

    def process(self, msg, kwargs):
        return f'{self.step} {msg}', kwargs


class Berny(Generator):
    """Generator that receives energy and gradients and yields the next geometry.

    Args:
        geom: geometry to start with
        debug: if :data:`True`, the generator yields debug info on receiving
            the energy and gradients, otherwise it yields :data:`None`
        restart: state captured from a previous run with ``debug=True``
        maxsteps: abort after maximum number of steps
        logger: alternative logger to use
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
        **params: Any,
    ) -> None:
        self._debug = debug
        self._maxsteps = maxsteps
        self._converged = False
        self._n = 0
        self._log = BernyAdapter(logger or log)
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
    ) -> tuple[InternalCoords, np.ndarray, np.ndarray, OptPoint]:
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
    ) -> dict[str, Any] | None:  # noqa: D102
        self._log.step = self._n
        log, s = self._log.info, self._state
        energy, gradients = energy_and_gradients
        gradients = np.array(gradients)
        log(f'Energy: {energy:.12}')
        # C2: adaptive coordinate rebuild. If an sp-like triple has crossed
        # the linear-bend threshold (175° / 170° hysteresis) since the coord
        # set was last built, rebuild now — *before* computing B and the
        # current q — so that this iteration runs entirely in the new
        # q-space. The BFGS history is dropped because the Hessian was
        # accumulated in a different (and possibly differently-sized)
        # coordinate system; ``first=True`` short-circuits the next
        # iteration's update_hessian/linear_search/update_trust calls and
        # restarts them from a guess Hessian. We skip the check on the
        # first iteration since coords were just built from this geometry.
        if not s.first and s.coords.needs_rebuild(s.geom):
            log('Linear-bend topology changed; rebuilding internal coordinates')
            s.coords, s.H, s.weights, s.future = self._build_coord_state(
                s.geom, s.params
            )
            s.first = True
            s.interpolated = None
            s.predicted = None
            s.previous = None
            s.best = None
        B = s.coords.B_matrix(s.geom)
        B_inv = B.T.dot(Math.pinv(np.dot(B, B.T), log=log))
        current = OptPoint(s.future.q, energy, dot(B_inv.T, gradients.reshape(-1)))
        if not s.first:
            assert s.best is not None
            assert s.previous is not None
            assert s.predicted is not None
            assert s.interpolated is not None
            s.H = update_hessian(
                s.H, current.q - s.best.q, current.g - s.best.g, log=log
            )
            s.trust = update_trust(
                s.trust,
                current.E - s.previous.E,  # or should it be s.interpolated.E?
                s.predicted.E - s.interpolated.E,
                s.predicted.q - s.interpolated.q,
                log=log,
                energy_noise=s.params.energy_noise,
            )
            dq = s.best.q - current.q
            t, E = linear_search(
                current.E, s.best.E, dot(current.g, dq), dot(s.best.g, dq), log=log
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
        dq, dE, on_sphere = quadratic_step(
            dot(proj, s.interpolated.g), H_proj, s.weights, s.trust, log=log
        )
        s.predicted = OptPoint(s.interpolated.q + dq, s.interpolated.E + dE, None)
        dq = s.predicted.q - current.q
        log(f'Total step: RMS: {Math.rms(dq):.3}, max: {max(abs(dq)):.3}')
        q, s.geom = s.coords.update_geom(
            s.geom, current.q, s.predicted.q - current.q, B_inv, log=log
        )
        s.future = OptPoint(q, None, None)
        s.previous = current
        if s.first or (s.best is not None and current.E < s.best.E):
            s.best = current
        s.first = False
        self._converged = is_converged(
            current.g, s.future.q - current.q, on_sphere, s.params, log=log
        )
        if self._n == self._maxsteps:
            log('Maximum number of steps reached')
        if self._debug:
            return {f.name: getattr(s, f.name) for f in fields(s)}
        return None

    def throw(self, *args, **kwargs):  # noqa: D102
        return Generator.throw(self, *args, **kwargs)


def no_log(msg, **kwargs):
    pass


def update_hessian(H, dq, dg, log=no_log):
    dH1 = dg[None, :] * dg[:, None] / dot(dq, dg)
    dH2 = H.dot(dq[None, :] * dq[:, None]).dot(H) / dq.dot(H).dot(dq)
    dH = dH1 - dH2  # BFGS update
    log('Hessian update information:')
    log(f'* Change: RMS: {Math.rms(dH):.3}, max: {abs(dH).max():.3}')
    return H + dH


def update_trust(trust, dE, dE_predicted, dq, log=no_log, *, energy_noise=2e-8):
    if abs(dE_predicted) < 10 * energy_noise:
        if abs(norm(dq) - trust) < 1e-10:
            return 2 * trust
        return trust
    if dE != 0:
        r = dE / dE_predicted  # Fletcher's parameter
    else:
        r = 1.0
    log(f"Trust update: Fletcher's parameter: {r:.3}")
    if r < 0.25:
        return norm(dq) / 4
    elif r > 0.75 and abs(norm(dq) - trust) < 1e-10:
        return 2 * trust
    else:
        return trust


def linear_search(E0, E1, g0, g1, log=no_log):
    log('Linear interpolation:')
    log(f'* Energies: {E0:.8}, {E1:.8}')
    log(f'* Derivatives: {g0:.3}, {g1:.3}')
    t, E = Math.fit_quartic(E0, E1, g0, g1)
    if t is None or t < -1 or t > 2:
        t, E = Math.fit_cubic(E0, E1, g0, g1)
        if t is None or t < 0 or t > 1:
            if E0 <= E1:
                log('* No fit succeeded, staying in new point')
                return 0, E0

            else:
                log('* No fit succeeded, returning to best point')
                return 1, E1
        else:
            msg = 'Cubic interpolation was performed'
    else:
        msg = 'Quartic interpolation was performed'
    log(f'* {msg}: t = {t:.3}')
    log(f'* Interpolated energy: {E:.8}')
    return t, E


def quadratic_step(g, H, w, trust, log=no_log):
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    rfo = np.vstack((np.hstack((H, g[:, None])), np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo + rfo.T) / 2)
    dq = V[:-1, 0] / V[-1, 0]
    l = D[0]
    if norm(dq) <= trust:
        log('Pure RFO step was performed:')
        on_sphere = False
    else:

        def steplength(l):
            return norm(np.linalg.solve(l * eye(H.shape[0]) - H, g)) - trust

        l = Math.findroot(steplength, ev[0])  # minimization on sphere
        dq = np.linalg.solve(l * eye(H.shape[0]) - H, g)
        on_sphere = True
        log('Minimization on sphere was performed:')
    dE = dot(g, dq) + 0.5 * dq.dot(H).dot(dq)  # predicted energy change
    log(f'* Trust radius: {trust:.2}')
    log(f'* Number of negative eigenvalues: {(ev < 0).sum()}')
    log(f'* Lowest eigenvalue: {ev[0]:.3}')
    log(f'* lambda: {l:.3}')
    log(f'Quadratic step: RMS: {Math.rms(dq):.3}, max: {max(abs(dq)):.3}')
    log(f'* Predicted energy change: {dE:.3}')
    return dq, dE, on_sphere


def is_converged(forces, step, on_sphere, params: BernyParams, log=no_log) -> bool:
    criteria: list[tuple] = [
        ('Gradient RMS', Math.rms(forces), params.gradientrms),
        ('Gradient maximum', np.max(abs(forces)), params.gradientmax),
    ]
    if on_sphere:
        criteria.append(('Minimization on sphere', False))
    else:
        criteria.extend(
            [
                ('Step RMS', Math.rms(step), params.steprms),
                ('Step maximum', np.max(abs(step)), params.stepmax),
            ]
        )
    log('Convergence criteria:')
    all_matched = True
    for crit in criteria:
        if len(crit) > 2:
            result = crit[1] < crit[2]
            op = '<' if result else '>'
            msg = f'{crit[1]:.3} {op} {crit[2]:.3}'
        else:
            msg, result = crit
        msg = f'{crit[0]}: {msg}' if msg else crit[0]
        verdict = 'OK' if result else 'no'
        msg = f'* {msg} => {verdict}'
        log(msg)
        if not result:
            all_matched = False
    if all_matched:
        log('* All criteria matched')
    return all_matched
