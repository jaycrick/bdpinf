r"""Constrained maximum likelihood on the non-negativity wall of spline rates.

A spline rate (:class:`~bdpinf.models.rate_functions.PiecewiseCubicRate`,
:class:`~bdpinf.models.rate_functions.PiecewiseQuadraticRate`) can dip below
zero between its knots, and the likelihood is $-\infty$ there.
The maximum likelihood estimate often sits on that wall: when the rate value
at $T$ is fixed at zero the end slope must satisfy $w'(T^-) \le 0$, and an
interior segment can have its minimum touch zero (a tangency).
An unconstrained trust-region Newton method cannot converge there, because
every step out of the feasible set is rejected and the gradient is *not* zero
at a constrained optimum.

This module solves the constrained problem
$\max \ell(x)$ subject to $c(x) \ge 0$, with $c$ the scaled
``nonneg_margins`` of the rates, on the free unconstrained coordinates $x$:

1. :func:`project_feasible` (phase 1, only if the start violates $c \ge$ margin);
2. an SLSQP solve (:func:`solve_slsqp`);
3. :func:`kkt_polish`, a Newton method on the KKT system of the active set
   found by non-negative least squares;
4. :func:`verify_kkt`, the convergence definition below.

:func:`refine_constrained` runs these from up to three starts until one is
verified.
The final point is strictly feasible, $c_i \ge 10^{-9}$ (the rows are divided
by the largest knot value of their rate), so that re-evaluating the likelihood
from saved parameters is finite.

**Convergence of a guard-active optimum** (:func:`verify_kkt`).
Let the active set be the constraints with $c_i \le 10^{-6}$ together with the
coordinates whose transform bound is active (a log coordinate at or below the
boundary threshold).
The point is verified when the log-likelihood is finite;
the multipliers $\lambda \ge 0$ of the non-negative least-squares fit of
$\nabla f$ on the active rows leave a residual
$\lVert \nabla f - J_A^\top \lambda \rVert_2 \le$ ``grad_tol``;
$\max_i \lambda_i (c_i - m) \le 10^{-6}$ (complementary slackness, measured
from the margin $m$ that the solver holds the active constraints at,
because $\lambda_i m$ grows with the sample size);
and the Hessian of the Lagrangian reduced to the tangent space of the active
constraints with positive multiplier has smallest eigenvalue above
$-10^{-8} \max(1, \max |\text{eig}|)$.
With no active constraint the ordinary interior test applies
(gradient norm within tolerance, positive-definite Hessian).
"""

import dataclasses
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from bdpinf.inference.constraints import (
    INTERVAL_CODE,
    LOG_CODE,
    CompiledConstraints,
)
from bdpinf.inference.rate_models import BirthDeathModel

__all__ = [
    "ConstrainedOutcome",
    "KKTReport",
    "WallConstraints",
    "kkt_polish",
    "project_feasible",
    "refine_constrained",
    "solve_slsqp",
    "verify_kkt",
    "wall_constraints",
]

ACTIVE_TOL = 1e-6
"""A scaled margin at or below this counts as active."""

SLACK_TOL = 1e-6
"""Largest accepted complementarity product ``lambda_i * c_i``."""

MIN_MARGIN = 1e-9
"""Smallest scaled margin of an accepted final point (strict feasibility)."""

TARGET_MARGIN = 2e-9
"""Scaled margin aimed at by the solvers, so that rounding keeps ``MIN_MARGIN``."""

EIG_TOL = 1e-8
"""Relative tolerance on the smallest reduced-Hessian eigenvalue."""

logger = logging.getLogger(__name__)

_MAX_TRIES = 3
_MAX_ROUNDS = 8
_MAX_CACHED = 64
_PERTURB = (0.05, 0.15)
_POLISH_ITERS = 30
_PINNED_NAME = re.compile(r"^(?P<prefix>birth|death)\.knot(?P<index>\d+)$")

NllGrad = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
Hessian = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclasses.dataclass(frozen=True)
class KKTReport:
    """Outcome of :func:`verify_kkt`.

    Attributes
    ----------
    verified : bool
        All the convergence conditions hold.
    kind : str
        ``"interior"`` (no active constraint) or ``"constrained"``.
    residual : float
        ``||grad - J_A' lambda||`` (the gradient norm when interior).
    min_reduced_eig : float | None
        Smallest eigenvalue of the reduced Lagrangian Hessian (of the Hessian
        when interior); ``None`` when the tangent space is empty.
    complementarity : float
        ``max lambda_i c_i`` over the active constraints.
    active : tuple[int, ...]
        Indices of the active constraint rows.
    bound_active : tuple[int, ...]
        Indices of the active transform-bound rows.
    multipliers : NDArray
        Non-negative multipliers, constraint rows first then bound rows.
    """

    verified: bool
    kind: str
    residual: float
    min_reduced_eig: float | None
    complementarity: float
    active: tuple[int, ...]
    bound_active: tuple[int, ...]
    multipliers: NDArray[np.float64]


def _null_space(rows: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Orthonormal basis (columns) of the null space of ``rows``."""
    if rows.shape[0] == 0:
        return np.eye(n)
    _, s, vt = np.linalg.svd(rows)
    rank = int(np.sum(s > 1e-10 * s[0])) if s.size and s[0] > 0 else 0
    return vt[rank:].T


def verify_kkt(  # noqa: PLR0913  # the problem data and the tolerances
    grad: NDArray[np.float64],
    hess: NDArray[np.float64],
    c: NDArray[np.float64],
    jac: NDArray[np.float64],
    hess_c: NDArray[np.float64] | None = None,
    *,
    grad_tol: float,
    bound_rows: NDArray[np.float64] | None = None,
    finite: bool = True,
    margin: float = 0.0,
    active_tol: float = ACTIVE_TOL,
    slack_tol: float = SLACK_TOL,
    eig_tol: float = EIG_TOL,
) -> KKTReport:
    r"""Check the KKT conditions of $\min f$ subject to $c \ge 0$ at a point.

    Parameters
    ----------
    grad, hess : NDArray
        Gradient and Hessian of the objective $f$ (here the negative
        log-likelihood).
    c, jac, hess_c : NDArray
        Scaled constraint values ``(m,)``, Jacobian ``(m, n)`` and Hessians
        ``(m, n, n)`` (``None`` for linear constraints).
    grad_tol : float
        Tolerance on the stationarity residual.
    bound_rows : NDArray, optional
        Rows ``(nb, n)`` of transform bounds that are active (always
        counted as active, with zero curvature and zero slack).
    finite : bool
        Whether the log-likelihood is finite at the point.
    margin : float
        The margin the solver holds the active constraints at (the problem
        solved is $c \ge$ margin, which keeps the point strictly feasible);
        complementary slackness is measured from it,
        $\lambda_i \max(c_i - \text{margin}, 0)$, because
        $\lambda_i \cdot \text{margin}$ grows with the sample size
        (the multipliers scale with the number of observations) and is only the
        deliberate price of strict feasibility.
        Zero measures $\lambda_i c_i$.
    active_tol, slack_tol, eig_tol : float
        Thresholds on active margins, complementarity and the reduced
        eigenvalue (relative).

    Returns
    -------
    KKTReport
    """
    n = len(grad)
    if not (
        np.all(np.isfinite(grad))
        and np.all(np.isfinite(hess))
        and np.all(np.isfinite(c))
        and np.all(np.isfinite(jac))
    ):
        return KKTReport(
            False, "non-finite", float("inf"), None, 0.0, (), (), np.zeros(0)
        )
    act = np.flatnonzero(np.asarray(c) <= active_tol)
    bounds = np.zeros((0, n)) if bound_rows is None else np.asarray(bound_rows)
    try:
        eig_all = np.linalg.eigvalsh(0.5 * (hess + hess.T))
    except np.linalg.LinAlgError:
        eig_all = np.array([np.nan])
    if act.size + len(bounds) == 0:
        residual = float(np.linalg.norm(grad))
        min_eig = float(eig_all[0])
        verified = bool(
            finite and residual <= grad_tol and np.isfinite(min_eig) and min_eig > 0
        )
        return KKTReport(
            verified, "interior", residual, min_eig, 0.0, (), (), np.zeros(0)
        )
    rows = np.vstack([jac[act], bounds])
    lam, residual = scipy.optimize.nnls(rows.T, grad, maxiter=50 * n)
    n_c = act.size
    slack = float(np.max(lam[:n_c] * np.maximum(c[act] - margin, 0.0))) if n_c else 0.0
    positive = lam > 1e-10 * max(1.0, float(lam.max()))
    z = _null_space(rows[positive], n)
    min_eig: float | None = None
    ok_curvature = True
    if z.shape[1]:
        lagrangian = np.array(hess, dtype=np.float64)
        if hess_c is not None:
            for i, mult in zip(act, lam[:n_c], strict=True):
                lagrangian = lagrangian - mult * hess_c[i]
        reduced = z.T @ (0.5 * (lagrangian + lagrangian.T)) @ z
        try:
            eigs = np.linalg.eigvalsh(reduced)
            min_eig = float(eigs[0])
            ok_curvature = min_eig > -eig_tol * max(1.0, float(np.max(np.abs(eigs))))
        except np.linalg.LinAlgError:
            ok_curvature = False
    verified = bool(
        finite and residual <= grad_tol and slack <= slack_tol and ok_curvature
    )
    return KKTReport(
        verified,
        "constrained",
        float(residual),
        min_eig,
        slack,
        tuple(int(i) for i in act),
        tuple(range(len(bounds))),
        lam,
    )


# ---------------------------------------------------------------------------
# The constraint set of a model
# ---------------------------------------------------------------------------

_WALL_CACHE: dict[Any, tuple[Callable[..., Any], Callable[..., Any]]] = {}


def _pinned_knots(compiled: CompiledConstraints, prefix: str) -> tuple[int, ...]:
    """Knot indices of ``prefix`` rate that a fixed parameter pins to exactly zero."""
    pinned = []
    for i, p in enumerate(compiled.params):
        found = _PINNED_NAME.match(p.name)
        if (
            found is not None
            and found["prefix"] == prefix
            and compiled.is_fixed[i]
            and float(compiled.fixed_natural[i]) == 0.0
        ):
            pinned.append(int(found["index"]))
    return tuple(sorted(pinned))


def _rate_scale(rate: Any) -> float:
    """Largest knot value of ``rate`` (1 if it is not positive)."""
    top = max(
        float(jnp.max(jnp.abs(rate.start_values))),
        float(jnp.max(jnp.abs(rate.end_values))),
    )
    return top if top > 0 and np.isfinite(top) else 1.0


@dataclasses.dataclass(frozen=True)
class WallConstraints:
    r"""Scaled non-negativity constraints $c(x) \ge 0$ on the free coordinates.

    Built by :func:`wall_constraints`; the rows of each rate are divided by
    that rate's largest knot value at the start of the fit.
    """

    compiled: CompiledConstraints
    labels: tuple[str, ...]
    _margins: Callable[..., jax.Array]
    _jacobian: Callable[..., jax.Array]
    _fixed: jax.Array
    _scales: jax.Array
    bound_log: float

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Scaled margins at ``x`` (all $\ge 0$ when the rates are non-negative)."""
        return np.asarray(
            self._margins(jnp.asarray(x), self._fixed, self._scales), dtype=np.float64
        )

    def jac(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the margins, shape ``(m, n_free)``."""
        return np.asarray(
            self._jacobian(jnp.asarray(x), self._fixed, self._scales),
            dtype=np.float64,
        )

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Hessians of the margins ``(m, n, n)``, by central differences of :meth:`jac`.

        Finite differences of the JAX Jacobian were more robust than
        automatic second derivatives, which gave NaNs at some feasible points.
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        out = np.zeros((len(self.labels), n, n))
        for j in range(n):
            step = 1e-6 * max(1.0, abs(x[j]))
            e = np.zeros(n)
            e[j] = step
            out[:, :, j] = (self.jac(x + e) - self.jac(x - e)) / (2.0 * step)
        return 0.5 * (out + out.transpose(0, 2, 1))

    def coordinate_bounds(self) -> list[tuple[float | None, float | None]]:
        """Box for each free coordinate that keeps its transform well defined.

        A log coordinate is kept above ``bound_log - 8`` and an interval
        coordinate within ``-bound_log + 8`` of zero.
        Without it SLSQP drifts a coordinate at its transform bound towards
        minus infinity (a rate of 1e-67 and smaller), where the likelihood
        stays finite but its gradient is NaN.
        """
        codes = self.compiled.codes[self.compiled.free_indices]
        floor, ceiling = self.bound_log - 8.0, -self.bound_log + 8.0
        return [
            (floor, None)
            if code == LOG_CODE
            else (-ceiling, ceiling)
            if code == INTERVAL_CODE
            else (None, None)
            for code in codes
        ]

    def bound_rows(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], tuple[str, ...]]:
        """Rows of the transform bounds that are active at ``x``, with labels."""
        compiled = self.compiled
        codes = compiled.codes[compiled.free_indices]
        names = compiled.free_names
        rows, labels = [], []
        for k, (code, value) in enumerate(zip(codes, x, strict=True)):
            sign = 0.0
            if code == LOG_CODE and value <= self.bound_log:
                sign = 1.0
            elif code == INTERVAL_CODE and abs(value) > -self.bound_log:
                sign = -float(np.sign(value))
            if sign:
                row = np.zeros(len(x))
                row[k] = sign
                rows.append(row)
                labels.append(f"{names[k]}: transform bound")
        return np.array(rows).reshape(len(rows), len(x)), tuple(labels)


def wall_constraints(
    model: BirthDeathModel,
    compiled: CompiledConstraints,
    natural: Any,
    *,
    bound_log: float,
) -> WallConstraints | None:
    """Build the non-negativity constraints, or ``None`` if no rate has any.

    A rate takes part when it exposes ``nonneg_margins`` (the spline rates),
    except ``pchip``, which is non-negative by construction (its end slope of
    zero is not a binding constraint).
    A knot parameter that is fixed at exactly zero is passed as pinned
    (see :meth:`PiecewiseCubicRate.nonneg_margins`).
    The knot-value rows are left out: the log transform keeps them positive,
    and a knot value at its floor is a transform bound, not the wall.

    Parameters
    ----------
    model, compiled : BirthDeathModel, CompiledConstraints
        The family and the compiled constraints of the fit.
    natural : array-like
        Natural parameter vector at which the row scales are taken.
    bound_log : float
        Log-scale value below which a transform bound counts as active.
    """
    natural = jnp.asarray(natural)
    birth, death = model.rates(natural)
    rates = {"birth": birth, "death": death}
    families = {"birth": model.birth, "death": model.death}
    taking = [
        p
        for p, r in rates.items()
        if hasattr(r, "nonneg_margins")
        and getattr(families[p], "kind", None) != "pchip"
    ]
    if not taking:
        return None
    pinned = {p: _pinned_knots(compiled, p) for p in taking}
    scales = jnp.asarray([_rate_scale(rates[p]) for p in taking])
    key = (model, compiled.structure_key, tuple(taking), tuple(pinned.items()))
    funcs = _WALL_CACHE.get(key)
    if funcs is None:

        def margins(x: jax.Array, fixed: jax.Array, scale: jax.Array) -> jax.Array:
            b, d = model.rates(compiled.expand(x, fixed))
            made = {"birth": b, "death": d}
            return jnp.concatenate(
                [
                    made[p].nonneg_margins(pinned[p], include_values=False) / scale[i]
                    for i, p in enumerate(taking)
                ]
            )

        funcs = (jax.jit(margins), jax.jit(jax.jacfwd(margins)))
        if len(_WALL_CACHE) >= _MAX_CACHED:
            _WALL_CACHE.clear()
        _WALL_CACHE[key] = funcs
    labels = tuple(
        f"{p}: {name}"
        for p in taking
        for name in rates[p].nonneg_margin_labels(pinned[p], include_values=False)
    )
    return WallConstraints(
        compiled,
        labels,
        funcs[0],
        funcs[1],
        jnp.asarray(compiled.fixed_natural),
        scales,
        bound_log,
    )


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def _margin_constraint(wall: WallConstraints, margin: float) -> Any:
    """SLSQP inequality constraint ``wall(z) >= margin`` with its Jacobian."""

    def fun(z: NDArray[np.float64]) -> NDArray[np.float64]:
        return wall(z) - margin

    return {"type": "ineq", "fun": fun, "jac": wall.jac}


def project_feasible(
    wall: WallConstraints,
    x: NDArray[np.float64],
    *,
    margin: float = TARGET_MARGIN,
    is_finite: Callable[[NDArray[np.float64]], bool],
) -> NDArray[np.float64] | None:
    """Phase 1: the nearest point with every scaled margin at least ``margin``.

    Returns ``x`` itself when it is already feasible and has a finite
    likelihood, and ``None`` if no feasible point was found.
    """
    if bool(np.all(wall(x) >= margin)) and is_finite(x):
        return x

    def distance(z: NDArray[np.float64]) -> float:
        return 0.5 * float(np.sum((z - x) ** 2))

    def distance_grad(z: NDArray[np.float64]) -> NDArray[np.float64]:
        return z - x

    result = scipy.optimize.minimize(
        distance,
        x,
        jac=distance_grad,
        method="SLSQP",
        constraints=[_margin_constraint(wall, margin)],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    z = np.asarray(result.x, dtype=np.float64)
    if bool(np.all(wall(z) >= margin - 1e-12)) and is_finite(z):
        return z
    return None


def solve_slsqp(
    wall: WallConstraints,
    nll_grad: NllGrad,
    x0: NDArray[np.float64],
    *,
    margin: float = TARGET_MARGIN,
    max_iter: int = 500,
) -> NDArray[np.float64]:
    """Maximise the likelihood subject to the margins staying above ``margin``."""

    def value(z: NDArray[np.float64]) -> float:
        return nll_grad(z)[0]

    def gradient(z: NDArray[np.float64]) -> NDArray[np.float64]:
        return nll_grad(z)[1]

    bounds = wall.coordinate_bounds()
    lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
    upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    result = scipy.optimize.minimize(
        value,
        np.clip(x0, lower, upper),
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[_margin_constraint(wall, margin)],
        options={"ftol": 1e-14, "maxiter": max_iter},
    )
    return np.asarray(result.x, dtype=np.float64)


def _backtrack(
    wall: WallConstraints,
    nll_grad: NllGrad,
    residual: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    *,
    x: NDArray[np.float64],
    lam: NDArray[np.float64],
    step: NDArray[np.float64],
    idx: NDArray[np.intp],
    r_now: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
    """Halve the Newton step until it is feasible and lowers the KKT residual."""
    scale = 1.0
    for _ in range(20):
        trial = x.copy()
        trial[idx] = x[idx] + scale * step[: len(idx)]
        mult = lam + scale * step[len(idx) :]
        if bool(np.all(wall(trial) >= MIN_MARGIN)) and nll_grad(trial)[0] < 1e19:
            r_trial = residual(trial, mult)
            if r_trial < r_now:
                return trial, mult, r_trial
        scale *= 0.5
    return None


def kkt_polish(  # one Newton loop with its line search
    wall: WallConstraints,
    nll_grad: NllGrad,
    hessian: Hessian,
    x: NDArray[np.float64],
    *,
    margin: float = TARGET_MARGIN,
    max_iter: int = _POLISH_ITERS,
) -> NDArray[np.float64]:
    """Newton on the KKT system of the active set, which NNLS chooses.

    The active constraints (scaled margin at most ``ACTIVE_TOL``) with a
    positive multiplier are held at ``margin``; coordinates sitting on a
    transform bound are frozen.
    Returns ``x`` unchanged when there is nothing to polish or Newton cannot
    reduce the KKT residual while staying feasible.
    """
    n = len(x)
    c0, jac0 = wall(x), wall.jac(x)
    act = np.flatnonzero(c0 <= ACTIVE_TOL)
    if act.size == 0:
        return x
    _, grad0 = nll_grad(x)
    if not (np.all(np.isfinite(grad0)) and np.all(np.isfinite(jac0))):
        return x
    lam, _ = scipy.optimize.nnls(jac0[act].T, grad0, maxiter=50 * n)
    act, lam = act[lam > 0], lam[lam > 0]
    if act.size == 0:
        return x
    bound_rows, _ = wall.bound_rows(x)
    free = np.ones(n, dtype=bool)
    for row in bound_rows:
        free[int(np.flatnonzero(row)[0])] = False
    idx = np.flatnonzero(free)

    def residual(z: NDArray[np.float64], mult: NDArray[np.float64]) -> float:
        _, g = nll_grad(z)
        r = g[idx] - wall.jac(z)[act][:, idx].T @ mult
        return float(np.linalg.norm(np.r_[r, wall(z)[act] - margin]))

    r_now = residual(x, lam)
    for _ in range(max_iter):
        if r_now < 1e-12:
            break
        _, g = nll_grad(x)
        lagrangian = hessian(x) - np.tensordot(lam, wall.hess(x)[act], axes=1)
        a = wall.jac(x)[act][:, idx]
        m = act.size
        kkt = np.block([[lagrangian[np.ix_(idx, idx)], -a.T], [a, np.zeros((m, m))]])
        rhs = -np.r_[g[idx] - a.T @ lam, wall(x)[act] - margin]
        try:
            step = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            break
        found = _backtrack(
            wall,
            nll_grad,
            residual,
            x=x,
            lam=lam,
            step=step,
            idx=idx,
            r_now=r_now,
        )
        if found is None:
            break
        x, lam, r_now = found
    return x


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConstrainedOutcome:
    """Result of :func:`refine_constrained`.

    Attributes
    ----------
    x : NDArray
        Final free unconstrained parameters (strictly feasible).
    nll : float
        Negative log-likelihood there.
    report : KKTReport
        The convergence check at ``x``.
    labels : tuple[str, ...]
        Names of the active constraints (walls, then transform bounds).
    at_guard : bool
        Whether some non-negativity constraint is active.
    n_tries : int
        Starts used.
    """

    x: NDArray[np.float64]
    nll: float
    report: KKTReport
    labels: tuple[str, ...]
    at_guard: bool
    n_tries: int

    @property
    def verified(self) -> bool:
        """Whether the point meets the convergence definition."""
        return self.report.verified


def check_point(
    wall: WallConstraints,
    nll_grad: NllGrad,
    hessian: Hessian,
    x: NDArray[np.float64],
    *,
    grad_tol: float,
    penalty: float,
) -> tuple[KKTReport, tuple[str, ...], float, bool]:
    """Verify ``x``; returns the report, active labels, nll and ``at_guard``."""
    value, grad = nll_grad(x)
    c = wall(x)
    bound_rows, bound_labels = wall.bound_rows(x)
    report = verify_kkt(
        grad,
        hessian(x),
        c,
        wall.jac(x),
        wall.hess(x) if bool(np.any(c <= ACTIVE_TOL)) else None,
        grad_tol=grad_tol,
        bound_rows=bound_rows,
        finite=value < penalty,
        margin=TARGET_MARGIN,
    )
    labels = tuple(wall.labels[i] for i in report.active) + bound_labels
    return report, labels, value, bool(report.active)


def _one_try(  # noqa: PLR0917  # one start of the driver
    wall: WallConstraints,
    nll_grad: NllGrad,
    hessian: Hessian,
    start: NDArray[np.float64],
    tries: int,
    grad_tol: float,
    penalty: float,
    finite: Callable[[NDArray[np.float64]], bool],
) -> ConstrainedOutcome | None:
    """Project, then rounds of solve, polish and check (``None`` if infeasible).

    SLSQP often stops early on an ill-conditioned wall problem, so the
    solve and polish are repeated from the previous point (at most
    ``_MAX_ROUNDS``) until the point is verified or stops improving.
    """
    x = project_feasible(wall, start, is_finite=finite)
    if x is None:
        return None
    best: ConstrainedOutcome | None = None
    for _ in range(_MAX_ROUNDS):
        solved = solve_slsqp(wall, nll_grad, x)
        # SLSQP can stop a hair inside the margin; move back onto it.
        moved = project_feasible(wall, solved, is_finite=finite)
        if moved is None:
            break
        x = moved
        for candidate in (kkt_polish(wall, nll_grad, hessian, x), x):
            if bool(np.all(wall(candidate) >= MIN_MARGIN)) and finite(candidate):
                x = candidate
                break
        else:
            break
        report, labels, value, at_guard = check_point(
            wall, nll_grad, hessian, x, grad_tol=grad_tol, penalty=penalty
        )
        outcome = ConstrainedOutcome(x, value, report, labels, at_guard, tries)
        if report.verified:
            return outcome
        improved = best is None or outcome.nll < best.nll - 1e-9
        if best is None or outcome.nll < best.nll:
            best = outcome
        if not improved:
            break
    return best


def refine_constrained(  # solver inputs and tolerances
    wall: WallConstraints,
    nll_grad: NllGrad,
    hessian: Hessian,
    x0: NDArray[np.float64],
    *,
    grad_tol: float,
    penalty: float,
    extra_starts: Sequence[NDArray[np.float64]] = (),
    seed: int | None = None,
    max_tries: int = _MAX_TRIES,
) -> ConstrainedOutcome | None:
    """Constrained fit from ``x0``, retrying until the optimum is verified.

    Each try is project (if needed), SLSQP, KKT polish, verify.
    The first start is ``x0``, then ``extra_starts`` (for example the other
    fit starts), then perturbations of ``x0``; at most ``max_tries`` in all.
    Returns the first verified outcome, else the feasible one with the
    highest likelihood, else ``None``.
    """

    def finite(z: NDArray[np.float64]) -> bool:
        return nll_grad(z)[0] < penalty

    rng = np.random.default_rng(seed)
    starts = [np.asarray(x0, dtype=np.float64)]
    starts.extend(np.asarray(s, dtype=np.float64) for s in extra_starts)
    for std in _PERTURB:
        starts.append(starts[0] + rng.normal(0.0, std, size=starts[0].shape))
    best: ConstrainedOutcome | None = None
    for tries, start in enumerate(starts[:max_tries], start=1):
        try:
            outcome = _one_try(
                wall, nll_grad, hessian, start, tries, grad_tol, penalty, finite
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            logger.warning("constrained try %d failed: %s", tries, exc)
            continue
        if outcome is None:
            continue
        if outcome.verified:
            return outcome
        if best is None or outcome.nll < best.nll:
            best = outcome
    return best
