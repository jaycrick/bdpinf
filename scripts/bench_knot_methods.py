r"""Validate ways to fit rates and knot times together on date-range data.

Every manuscript date is a range of calendar years, and the likelihood
integrates the age density uniformly over it (an exact year ``y`` is the
one-year range ``[y, y + 1)``).
The ranges have different widths, so the likelihood is a mixture of bin masses
with no common grid.

Families (read from the sibling ``nt-manuscripts`` repository):

``jB``
    Kashouh family jB, start 900, linear birth with the knot at 1400 now
    fitted (the second knot stays at 1700 with zero birth after it) and a
    constant death rate.
``sim1000``
    Known-truth simulation (``T = 1000``): linear birth with a free knot at 500
    (0.025, 0.001 at the knot, 0.0001 at 800, zero afterwards), constant death
    0.003.
    Only date-range data are fitted; the range widths are resampled from the
    jB widths stored in ``JB_RANGE_WIDTHS``.
``K-pc`` / ``K-pl``
    Kashouh family K, start 1100, two free intermediate knots, piecewise
    constant or continuous piecewise-linear birth and a constant death rate.

Methods on the range likelihood:

``newton``
    Joint trust-region Newton over rates and knots from several knot starts.
``profile``
    Scan, Nelder-Mead on the logit knots, final fit (the ``fit_knots`` search).
``hybrid``
    Coarse profile, then a joint Newton polish.
``grid-newton``
    Coarse profile scan, then joint Newton from the best placements.

Further methods used by ``sim1000``: ``lbfgs`` (joint L-BFGS-B with JAX
gradients), ``nelder-mead`` (joint, derivative free), ``brent`` (bounded 1-D
profile), ``de-profile`` (differential evolution on the profile) and
``basinhopping`` (joint, Newton local solves).

``exact-midpoint`` (jB and K-pc only) is the current practice: exact ages from
``quantile_resample_dates`` fitted with a profile search on the exact
likelihood.
Its objective differs, so only its estimates are compared.

Sections (``--only``):

``check``
    Closed-form masses against quadrature on the real ranges.
``real``
    One fit per method on the real data, then profile curves for the knots.
``sim``
    Replicates simulated from the real-data optimum, with ranges whose widths
    are resampled from the real ones.

Usage
-----
    uv run python scripts/bench_knot_methods.py --quick --families jB
    uv run python scripts/bench_knot_methods.py --only real
    uv run python scripts/bench_knot_methods.py --only sim1000 --replicates 10
    uv run python scripts/bench_knot_methods.py --only sim1000 --shard 3/40
    uv run python scripts/bench_knot_methods.py --only sim1000 --merge
"""

import argparse
import dataclasses
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats.qmc
from bench_binned_ages import (
    _PENALTY,
    binned_log_likelihood,
    binned_problem,
    solve,
    starting_points,
)
from bench_fitting import _table

from bdpinf.analysis.date_density import quantile_resample_dates
from bdpinf.inference import AgeData, BirthDeathModel, Constraints, FreeKnots, fit
from bdpinf.inference import fitting as fitting_module
from bdpinf.inference.knots import fit_knots
from bdpinf.inference.uncertainty import covariance_from_hessian
from bdpinf.models.math_general import DEFAULT_NODES, terms_for
from bdpinf.paths import find_project_root
from bdpinf.simulation import simulate_ages_from_rates

jax.config.update("jax_enable_x64", True)

SECTIONS = ("check", "real", "sim", "sim1000")
CURRENT_YEAR = 2020
RANGE_METHODS = ("newton", "profile", "hybrid", "grid-newton")
EXTRA_METHODS = ("lbfgs", "nelder-mead", "brent", "de-profile", "basinhopping")
SIM_METHODS = (*RANGE_METHODS, *EXTRA_METHODS)
SIM_KEY = "sim1000"
# Range widths (years) of the 40 Kashouh family jB manuscripts and the
# probability each is chosen, so sim1000 needs no data from nt-manuscripts.
JB_RANGE_WIDTHS = (
    (1, 23 / 40),
    (2, 2 / 40),
    (101, 14 / 40),
    (201, 1 / 40),
)
SIM_TRUTH = {
    "birth.start0": 0.025,
    "birth.end0": 0.001,
    "birth.start1": 0.001,
    "birth.end1": 0.0001,
    "birth.start2": 0.0,
    "birth.end2": 0.0,
    "death": 0.003,
    "tau0": 500.0,
    "tau1": 800.0,
}
HIT_TOL = 1e-3
CHI2_95 = 3.841


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Family:
    """A model, its constraints, and the knots that are being estimated."""

    key: str
    csv: str
    start: int
    model: BirthDeathModel
    equal: list[list[str]]
    fixed: dict[str, float]
    free_knots: tuple[str, ...]
    reported: tuple[str, ...]
    support_end: int | None  # last calendar year with births, if birth stops
    n_cand: int
    n_cand_coarse: int
    n_newton_starts: int

    @property
    def horizon(self) -> float:
        """Observation time ``T`` in years since the start."""
        return float(CURRENT_YEAR - self.start)

    @property
    def names(self) -> list[str]:
        """Natural-scale parameter names (knots last)."""
        return [p.name for p in self.model.params]

    @property
    def knots(self) -> FreeKnots:
        """The model's knots."""
        assert self.model.knots is not None
        return self.model.knots

    @property
    def free_ranges(self) -> list[tuple[float, float]]:
        """The ranges of the free knots."""
        ranges = dict(zip(self.knots.names, self.knots.ranges, strict=True))
        return [ranges[name] for name in self.free_knots]


def make_family(key: str) -> Family:
    """Build one of ``jB``, ``K-pc``, ``K-pl``, ``sim1000``."""
    if key == "jB":
        horizon = float(CURRENT_YEAR - 900)
        knots = FreeKnots(2, horizon, bounds=((0.0, 790.0), (790.0, horizon)))
        model = BirthDeathModel.piecewise_linear(knots, continuous=False)
        return Family(
            key,
            "Kashouh-ALAGM-FamilyJB.csv",
            900,
            model,
            [
                ["birth.end0", "birth.start1"],
                [f"death.{e}{k}" for e in ("start", "end") for k in range(3)],
            ],
            {"birth.start2": 0.0, "birth.end2": 0.0, "tau1": 800.0},
            ("tau0",),
            ("birth.start0", "birth.start1", "birth.end1", "death.start0", "tau0"),
            1700,
            64,
            12,
            3,
        )
    if key == SIM_KEY:
        horizon = 1000.0
        knots = FreeKnots(2, horizon, bounds=((0.0, 790.0), (790.0, horizon)))
        return Family(
            key,
            "",
            CURRENT_YEAR - 1000,
            BirthDeathModel.piecewise_linear(knots, continuous=False),
            [
                ["birth.end0", "birth.start1"],
                [f"death.{e}{k}" for e in ("start", "end") for k in range(3)],
            ],
            {"birth.start2": 0.0, "birth.end2": 0.0, "tau1": 800.0},
            ("tau0",),
            ("birth.start0", "birth.start1", "birth.end1", "death.start0", "tau0"),
            CURRENT_YEAR - 1000 + 800,
            64,
            12,
            3,
        )
    horizon = float(CURRENT_YEAR - 1100)
    knots = FreeKnots(2, horizon, bounds=((0.0, horizon / 2), (horizon / 2, horizon)))
    if key == "K-pc":
        return Family(
            key,
            "Kashouh-ALAGM-FamilyK.csv",
            1100,
            BirthDeathModel.piecewise_constant(knots),
            [["death.v0", "death.v1", "death.v2"]],
            {},
            ("tau0", "tau1"),
            ("birth.v0", "birth.v1", "birth.v2", "death.v0", "tau0", "tau1"),
            None,
            64,
            16,
            6,
        )
    if key == "K-pl":
        return Family(
            key,
            "Kashouh-ALAGM-FamilyK.csv",
            1100,
            BirthDeathModel.piecewise_linear(knots, continuous=True),
            [[f"death.knot{k}" for k in range(4)]],
            {},
            ("tau0", "tau1"),
            (
                "birth.knot0",
                "birth.knot1",
                "birth.knot2",
                "birth.knot3",
                "death.knot0",
                "tau0",
                "tau1",
            ),
            None,
            64,
            16,
            6,
        )
    msg = f"unknown family {key}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Data: calendar-year ranges to age ranges
# ---------------------------------------------------------------------------


def clip_years(
    lower: np.ndarray, upper: np.ndarray, start: int, end: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ranges ``[lower, upper)`` in ``[start, end]``, at least a year wide."""
    lower = np.asarray(lower, dtype=np.int64)
    upper = np.asarray(upper, dtype=np.int64)
    if end is not None:
        upper = np.minimum(upper, end)
        lower = np.minimum(lower, upper - 1)
    lower = np.maximum(lower, start)
    upper = np.maximum(upper, lower + 1)
    return lower, upper


@dataclasses.dataclass(frozen=True)
class RangeData:
    """Age ranges aggregated by distinct ``(lo, hi)``, plus the year ranges."""

    lower: np.ndarray  # calendar years, per observation
    upper: np.ndarray  # exclusive
    horizon: float

    @property
    def n(self) -> int:
        """Number of manuscripts."""
        return len(self.lower)

    @property
    def widths(self) -> np.ndarray:
        """Range widths in years."""
        return self.upper - self.lower

    @property
    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lo, hi, counts)`` of the distinct age ranges."""
        lo_age = (CURRENT_YEAR - self.upper).astype(np.float64)
        hi_age = (CURRENT_YEAR - self.lower).astype(np.float64)
        pairs, counts = np.unique(
            np.stack([lo_age, hi_age], 1), axis=0, return_counts=True
        )
        return pairs[:, 0], pairs[:, 1], counts.astype(np.float64)

    def padded_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``arrays`` padded with zero-weight bins to a power-of-two length.

        A fixed shape lets the jitted objective compile once per bucket
        instead of once per replicate.
        """
        lo, hi, w = self.arrays
        size = 1 << max(3, math.ceil(math.log2(len(w))))
        extra = size - len(w)
        mid = 0.5 * self.horizon
        return (
            np.concatenate([lo, np.full(extra, mid)]),
            np.concatenate([hi, np.full(extra, mid + 1.0)]),
            np.concatenate([w, np.zeros(extra)]),
        )

    def midpoint_data(self) -> AgeData:
        """Midpoint ages as ``AgeData`` (only used for the moment start)."""
        mid = CURRENT_YEAR - 0.5 * (self.lower + self.upper)
        return AgeData(
            non_spine_ages=jnp.asarray(mid, dtype=jnp.float64),
            spine_ages=None,
            ns=jnp.asarray([self.n], dtype=jnp.int32),
            end_time=self.horizon,
            use_unknown_spine=True,
        )

    def resampled_ages(self) -> np.ndarray:
        """Exact ages as in current practice: KDE-quantile resampled years."""
        years = quantile_resample_dates(
            self.lower, self.upper - 1, current_year=CURRENT_YEAR
        )
        return np.clip(CURRENT_YEAR - years.astype(np.float64), 1.0, self.horizon - 1.0)


def load_real(fam: Family, data_dir: Path) -> RangeData:
    """Read the Kashouh family CSV; an exact year ``y`` is the range ``[y, y+1)``."""
    frame = pd.read_csv(data_dir / fam.csv)
    early = pd.to_numeric(frame["Earliest Date"], errors="coerce")
    late = pd.to_numeric(frame["Latest Date"], errors="coerce")
    keep = early.notna() & late.notna()
    lower, upper = clip_years(
        early[keep].to_numpy(), late[keep].to_numpy() + 1, fam.start, fam.support_end
    )
    return RangeData(lower, upper, fam.horizon)


# ---------------------------------------------------------------------------
# The fitting engine
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Result:
    """One method on one dataset."""

    natural: np.ndarray
    log_likelihood: float
    converged: bool
    seconds: float
    work: int  # inner fits for a profile, optimiser iterations for Newton
    se: np.ndarray  # natural-scale standard errors, NaN when unknown
    curve: tuple[np.ndarray, np.ndarray] | None = None  # knot grid, profile ll


class Engine:
    """Range likelihood for one family and dataset, with the shared search tools."""

    def __init__(
        self, fam: Family, data: RangeData, seed: int = 0, *, pad: bool = False
    ) -> None:
        self.fam, self.data, self.seed = fam, data, seed
        self.model = fam.model
        self.arrays = data.padded_arrays() if pad else data.arrays
        self.moments = fitting_module.moment_initial_rates(data.midpoint_data())
        self.names = fam.names
        self.knot_names = list(fam.knots.names)
        self.tau_index = [self.names.index(k) for k in self.knot_names]
        self.free_index = [self.names.index(k) for k in fam.free_knots]
        self.joint = Constraints(fixed=fam.fixed, equal=fam.equal).compile(
            self.model.params
        )
        self.joint_problem = binned_problem(self.arrays, data.n, self.model, self.joint)
        self.fits = 0

    def _compile(self, fixed_knots: dict[str, float]) -> Any:
        return Constraints(
            fixed={**self.fam.fixed, **fixed_knots}, equal=self.fam.equal
        ).compile(self.model.params)

    def _knot_list(self, taus: np.ndarray) -> list[float]:
        values = {**self.fam.fixed, **dict(zip(self.fam.free_knots, taus, strict=True))}
        return [float(values[k]) for k in self.knot_names]

    def fit_at(
        self,
        taus: np.ndarray,
        warm: np.ndarray | None = None,
        *,
        compute_cov: bool = False,
    ) -> Any:
        """Fit the rates with the free knots fixed at ``taus``."""
        compiled = self._compile(dict(zip(self.fam.free_knots, taus, strict=True)))
        problem = binned_problem(self.arrays, self.data.n, self.model, compiled)
        start = None
        if warm is not None:
            nat = np.array(warm)
            nat[self.free_index] = taus
            start = np.asarray(compiled.reduce(jnp.asarray(nat)), dtype=np.float64)
            if float(problem.nll(jnp.asarray(start))) >= _PENALTY:
                start = None
        starts = (
            [start]
            if start is not None
            else starting_points(
                problem, self.model, compiled, self.moments, self._knot_list(taus)
            )
        )
        self.fits += 1
        return solve(problem, compiled, starts, compute_cov=compute_cov)

    def joint_solve(self, naturals: list[np.ndarray]) -> Any:
        """Joint Newton over rates and free knots from full natural-scale starts."""
        starts = []
        for nat in naturals:
            u = np.asarray(self.joint.reduce(jnp.asarray(nat)), dtype=np.float64)
            if float(self.joint_problem.nll(jnp.asarray(u))) < _PENALTY:
                starts.append(u)
        if not starts:
            msg = "No finite joint start"
            raise RuntimeError(msg)
        return solve(self.joint_problem, self.joint, starts)

    def candidates(self, n: int) -> np.ndarray:
        """Knot placements, one row each (free knots only)."""
        ranges = self.fam.free_ranges
        lo = np.array([r[0] for r in ranges])
        hi = np.array([r[1] for r in ranges])
        if len(ranges) == 1:
            return np.linspace(lo[0], hi[0], n + 2)[1:-1, None]
        sobol = scipy.stats.qmc.Sobol(
            d=len(ranges), scramble=True, rng=np.random.default_rng(self.seed)
        )
        unit = sobol.random_base2(max(1, math.ceil(math.log2(n))))
        return lo + (hi - lo) * unit

    def with_se(self, solved: Any, seconds: float, work: int) -> Result:
        """Package a joint fit, taking the standard errors from its covariance."""
        se = np.sqrt(np.clip(np.diag(solved.covariance), 0.0, None))
        return Result(
            solved.natural, solved.log_likelihood, solved.converged, seconds, work, se
        )


def _nan_result(seconds: float, size: int) -> Result:
    return Result(
        np.full(size, np.nan), -np.inf, False, seconds, 0, np.full(size, np.nan)
    )


def run_newton(eng: Engine) -> Result:
    """Joint Newton from several knot starts."""
    began = time.perf_counter()
    starts: list[np.ndarray] = []
    cands = eng.candidates(eng.fam.n_newton_starts)[: eng.fam.n_newton_starts]
    for taus in cands:
        compiled = eng.joint
        knots = eng._knot_list(taus)
        starts += [
            np.asarray(compiled.expand(jnp.asarray(u)), dtype=np.float64)
            for u in starting_points(
                eng.joint_problem, eng.model, compiled, eng.moments, knots
            )
        ]
    solved = eng.joint_solve(starts)
    return eng.with_se(solved, time.perf_counter() - began, solved.n_iter)


def run_profile(
    eng: Engine,
    *,
    n_cand: int,
    n_refine: int,
    maxfev_per_knot: int,
    polish: bool = False,
    newton_top: int = 0,
) -> Result:
    """Profile search over the free knots, optionally finished by Newton.

    With ``newton_top``, skip Nelder-Mead and run joint Newton from the fitted
    rates at the best ``newton_top`` placements.
    With ``polish``, finish with a joint Newton from the profile optimum.
    """
    fam = eng.fam
    ranges = fam.free_ranges
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    tried: dict[tuple[float, ...], tuple[float, np.ndarray]] = {}
    best: dict[str, Any] = {"ll": -np.inf, "natural": None}
    fits_before = eng.fits

    def profile(taus: np.ndarray) -> float:
        key = tuple(float(t) for t in np.round(taus, 9))
        if key in tried:
            return tried[key][0]
        try:
            res = eng.fit_at(taus, warm=best["natural"])
        except RuntimeError:
            tried[key] = (-np.inf, np.array([]))
            return -np.inf
        tried[key] = (res.log_likelihood, res.natural)
        if res.log_likelihood > best["ll"]:
            best["ll"], best["natural"] = res.log_likelihood, res.natural
        return res.log_likelihood

    began = time.perf_counter()
    cands = eng.candidates(n_cand)
    for row in cands:
        profile(row)
    if best["natural"] is None:
        return _nan_result(time.perf_counter() - began, len(eng.names))
    order = np.argsort(
        [-tried[tuple(float(t) for t in np.round(r, 9))][0] for r in cands]
    )

    if newton_top:
        starts = [
            tried[tuple(float(t) for t in np.round(r, 9))][1]
            for r in cands[order[:newton_top]]
        ]
        starts = [s for s in starts if len(s)]
        solved = eng.joint_solve(starts)
        return eng.with_se(
            solved, time.perf_counter() - began, solved.n_iter + eng.fits - fits_before
        )

    def to_knots(u: np.ndarray) -> np.ndarray:
        return lo + (hi - lo) / (1.0 + np.exp(-u))

    for row in cands[order[:n_refine]]:
        frac = np.clip((row - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        scipy.optimize.minimize(
            lambda u: -profile(to_knots(u)),
            np.log(frac / (1 - frac)),
            method="Nelder-Mead",
            options={
                "xatol": 1e-3,
                "fatol": 1e-9,
                "maxfev": maxfev_per_knot * len(ranges),
            },
        )
    taus = np.asarray(best["natural"])[eng.free_index]
    final = eng.fit_at(taus, warm=best["natural"], compute_cov=True)
    work = eng.fits - fits_before
    result = eng.with_se(final, 0.0, work)
    if polish:
        try:
            joint = eng.joint_solve([final.natural])
        except RuntimeError:
            joint = None
        if (
            joint is not None
            and joint.converged
            and joint.log_likelihood >= final.log_likelihood - 1e-9
        ):
            result = eng.with_se(joint, 0.0, work + joint.n_iter)
    result.seconds = time.perf_counter() - began
    grid = sorted((k[0], v[0]) for k, v in tried.items() if np.isfinite(v[0]))
    if len(lo) == 1 and grid:
        result.curve = (np.array([g[0] for g in grid]), np.array([g[1] for g in grid]))
    return result


def _package(
    eng: Engine, x: np.ndarray, converged: bool, seconds: float, work: int
) -> Result:
    """Wrap a joint optimum ``x`` (unconstrained scale) with Hessian standard errors."""
    xj = jnp.asarray(x)
    nll = float(eng.joint_problem.nll(xj))
    natural = np.asarray(eng.joint.expand(xj), dtype=np.float64)
    se = np.full(len(natural), np.nan)
    if nll < _PENALTY:
        try:
            hess = np.asarray(eng.joint_problem.hessian(xj), dtype=np.float64)
            cov_free = np.asarray(covariance_from_hessian(jnp.asarray(hess)))
            jac = np.asarray(jax.jacobian(eng.joint.expand)(xj))
            se = np.sqrt(np.clip(np.diag(jac @ cov_free @ jac.T), 0.0, None))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
    ll = -nll if nll < _PENALTY else float("-inf")
    return Result(natural, ll, converged and nll < _PENALTY, seconds, work, se)


def _joint_starts(eng: Engine, n_knots: int) -> list[np.ndarray]:
    """Moment starts and their birth/death swaps at ``n_knots`` knot placements."""
    cands = eng.candidates(n_knots)[:n_knots]
    out: list[np.ndarray] = []
    for taus in cands:
        out += list(
            starting_points(
                eng.joint_problem,
                eng.model,
                eng.joint,
                eng.moments,
                eng._knot_list(taus),
            )
        )
    return out


def _numpy_objective(eng: Engine) -> Any:
    """Finite numpy ``(value, gradient)`` of the joint negative log-likelihood."""
    vg = eng.joint_problem.value_and_grad

    def fun(u: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = vg(jnp.asarray(u))
        value, grad = float(value), np.asarray(grad, dtype=np.float64)
        if not np.isfinite(value) or value >= _PENALTY or not np.all(np.isfinite(grad)):
            return _PENALTY, np.zeros_like(u)
        return value, grad

    return fun


def run_lbfgs(eng: Engine) -> Result:
    """Joint L-BFGS-B with JAX gradients, from several knot starts."""
    began = time.perf_counter()
    fun = _numpy_objective(eng)
    best, work = None, 0
    for u0 in _joint_starts(eng, eng.fam.n_newton_starts):
        res = scipy.optimize.minimize(
            fun, u0, jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-8}
        )
        work += int(res.nfev)
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return _package(eng, best.x, bool(best.success), time.perf_counter() - began, work)


def run_nelder_mead(eng: Engine) -> Result:
    """Joint derivative-free Nelder-Mead from the middle knot start, restarted once."""
    began = time.perf_counter()
    fun = _numpy_objective(eng)

    def value(u: np.ndarray) -> float:
        return fun(u)[0]

    best, work = None, 0
    for u0 in _joint_starts(eng, 1):
        x, converged = u0, False
        for _ in range(2):  # restart from the optimum to escape a collapsed simplex
            res = scipy.optimize.minimize(
                value,
                x,
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxfev": 3000},
            )
            work += int(res.nfev)
            x, converged = res.x, bool(res.success)
        if best is None or res.fun < best[0]:
            best = (res.fun, x, converged)
    assert best is not None
    return _package(eng, best[1], best[2], time.perf_counter() - began, work)


class _Profiler:
    """Memoised ``tau -> profile log-likelihood`` with warm starts."""

    def __init__(self, eng: Engine) -> None:
        self.eng = eng
        self.best_ll, self.best_nat = -np.inf, None
        self.tried: dict[float, tuple[float, np.ndarray]] = {}

    def __call__(self, tau: float) -> float:
        key = round(float(tau), 9)
        if key in self.tried:
            return self.tried[key][0]
        try:
            res = self.eng.fit_at(np.array([key]), warm=self.best_nat)
        except RuntimeError:
            self.tried[key] = (-np.inf, np.array([]))
            return -np.inf
        self.tried[key] = (res.log_likelihood, res.natural)
        if res.log_likelihood > self.best_ll:
            self.best_ll, self.best_nat = res.log_likelihood, res.natural
        return res.log_likelihood

    def finish(self, began: float, fits_before: int) -> Result:
        """Fit the rates at the best knot and return the result with covariance."""
        eng = self.eng
        if self.best_nat is None:
            return _nan_result(time.perf_counter() - began, len(eng.names))
        taus = np.asarray(self.best_nat)[eng.free_index]
        final = eng.fit_at(taus, warm=self.best_nat, compute_cov=True)
        result = eng.with_se(final, 0.0, eng.fits - fits_before)
        result.seconds = time.perf_counter() - began
        return result


def run_brent(eng: Engine) -> Result:
    """Bounded Brent search of the one-knot profile, with no starting grid."""
    began, fits_before = time.perf_counter(), eng.fits
    (lo, hi), prof = eng.fam.free_ranges[0], _Profiler(eng)
    scipy.optimize.minimize_scalar(
        lambda t: -prof(t), bounds=(lo, hi), method="bounded", options={"xatol": 0.5}
    )
    return prof.finish(began, fits_before)


def run_de_profile(eng: Engine) -> Result:
    """Differential evolution on the one-knot profile, then a final fit."""
    began, fits_before = time.perf_counter(), eng.fits
    prof = _Profiler(eng)
    scipy.optimize.differential_evolution(
        lambda t: -prof(t[0]),
        [eng.fam.free_ranges[0]],
        popsize=8,
        maxiter=15,
        tol=1e-4,
        seed=eng.seed,
        polish=False,
        init="sobol",
    )
    return prof.finish(began, fits_before)


def run_basinhopping(eng: Engine) -> Result:
    """Basin hopping over the joint vector with trust-region Newton local solves."""
    began = time.perf_counter()
    problem = eng.joint_problem

    def value(u: np.ndarray) -> float:
        v = float(problem.nll(jnp.asarray(u)))
        return v if np.isfinite(v) else _PENALTY

    def grad(u: np.ndarray) -> np.ndarray:
        g = np.asarray(problem.value_and_grad(jnp.asarray(u))[1], dtype=np.float64)
        return g if np.all(np.isfinite(g)) else np.zeros_like(u)

    def hess(u: np.ndarray) -> np.ndarray:
        h = np.asarray(problem.hessian(jnp.asarray(u)), dtype=np.float64)
        return h if np.all(np.isfinite(h)) else np.eye(len(u))

    start = _joint_starts(eng, 1)[0]
    res = scipy.optimize.basinhopping(
        value,
        start,
        niter=10,
        stepsize=0.5,
        seed=eng.seed,
        minimizer_kwargs={
            "method": "trust-exact",
            "jac": grad,
            "hess": hess,
            "options": {"gtol": 1e-6 * problem.n_obs, "maxiter": 200},
        },
    )
    return _package(eng, res.x, True, time.perf_counter() - began, int(res.nfev))


def run_method(eng: Engine, method: str) -> Result:
    """Dispatch one range-likelihood method with its settings."""
    fam = eng.fam
    try:
        if method == "newton":
            return run_newton(eng)
        if method == "profile":
            return run_profile(eng, n_cand=fam.n_cand, n_refine=2, maxfev_per_knot=60)
        if method == "hybrid":
            return run_profile(
                eng,
                n_cand=fam.n_cand_coarse,
                n_refine=1,
                maxfev_per_knot=20,
                polish=True,
            )
        if method == "grid-newton":
            return run_profile(
                eng,
                n_cand=fam.n_cand_coarse,
                n_refine=0,
                maxfev_per_knot=0,
                newton_top=3,
            )
        if method == "lbfgs":
            return run_lbfgs(eng)
        if method == "nelder-mead":
            return run_nelder_mead(eng)
        if method == "brent":
            return run_brent(eng)
        if method == "de-profile":
            return run_de_profile(eng)
        if method == "basinhopping":
            return run_basinhopping(eng)
    except (RuntimeError, ValueError):
        return _nan_result(0.0, len(eng.names))
    msg = f"unknown method {method}"
    raise ValueError(msg)


def run_exact_midpoint(fam: Family, data: RangeData) -> Result | None:
    """Fit exact resampled ages with a knot profile search (current practice)."""
    if fam.key == "K-pl":
        return None
    ages = data.resampled_ages()
    exact = AgeData.from_groups([ages], fam.horizon, is_ordered=False)
    began = time.perf_counter()
    if fam.key == "K-pc":
        res = fit_knots(
            exact, fam.model, constraints=Constraints(equal=fam.equal), seed=0
        )
        natural = np.asarray(res.fit.natural, dtype=np.float64)
        return Result(
            natural,
            float("nan"),
            bool(res.fit.converged),
            time.perf_counter() - began,
            len(res.profile_points),
            np.full(len(natural), np.nan),
        )
    # jB: one free knot beside a fixed one, which fit_knots cannot take.
    (lo, hi), tried, best = fam.free_ranges[0], {}, {"ll": -np.inf, "nat": None}

    def profile(tau: float) -> float:
        key = round(float(tau), 6)
        if key not in tried:
            try:
                r = fit(
                    exact,
                    fam.model,
                    constraints=Constraints(
                        fixed={**fam.fixed, "tau0": key}, equal=fam.equal
                    ),
                    init=best["nat"],
                    compute_covariance=False,
                )
                tried[key] = r.log_likelihood
                if r.log_likelihood > best["ll"]:
                    best["ll"], best["nat"] = r.log_likelihood, r.natural
            except ValueError:
                tried[key] = -np.inf
        return tried[key]

    grid = np.linspace(lo, hi, 26)[1:-1]
    for tau in grid:
        profile(tau)
    tau0 = grid[int(np.argmax([tried[round(float(t), 6)] for t in grid]))]
    frac = float(np.clip((tau0 - lo) / (hi - lo), 1e-6, 1 - 1e-6))
    scipy.optimize.minimize(
        lambda u: -profile(lo + (hi - lo) / (1 + np.exp(-u[0]))),
        [np.log(frac / (1 - frac))],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-9, "maxfev": 30},
    )
    natural = np.asarray(best["nat"], dtype=np.float64)
    return Result(
        natural,
        float("nan"),
        True,
        time.perf_counter() - began,
        len(tried),
        np.full(len(natural), np.nan),
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _print(rows: list[dict[str, Any]], title: str) -> None:
    """Print a table whose rows may lack some columns."""
    columns = list(dict.fromkeys(k for row in rows for k in row))
    _table([{c: row.get(c, "") for c in columns} for row in rows], title)


def run_check(families: list[str], data_dir: Path) -> list[dict[str, Any]]:
    """Closed-form masses against quadrature, and finite likelihoods, on real ranges."""
    rows = []
    rng = np.random.default_rng(1)
    for key in families:
        fam = make_family(key)
        data = load_real(fam, data_dir)
        lo, hi, w = (jnp.asarray(a) for a in data.arrays)
        ns = jnp.asarray([data.n], dtype=jnp.int32)
        row: dict[str, Any] = {
            "family": key,
            "n": data.n,
            "distinct_ranges": len(w),
            "min_width": float(np.min(data.widths)),
            "max_width": float(np.max(data.widths)),
        }
        if key == "K-pc":
            worst = 0.0
            for _ in range(5):
                taus = np.sort(rng.uniform(100, 800, size=2))
                taus = [min(taus[0], 450.0), max(taus[1], 470.0)]
                nat = fam.model.constant_natural(
                    rng.uniform(0.003, 0.02), rng.uniform(0.001, 0.01), taus
                )
                birth, death = fam.model.rates(jnp.asarray(nat))
                terms = terms_for(birth, death, n_nodes=DEFAULT_NODES)
                closed = binned_log_likelihood(terms, lo, hi, w, ns)
                quad = binned_log_likelihood(terms, lo, hi, w, ns, quadrature=True)
                worst = max(worst, abs(float(closed - quad)))
            row["max_abs_closed_vs_quadrature"] = worst
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


def _year_params(fam: Family, natural: np.ndarray) -> dict[str, float]:
    out = {}
    for name in fam.reported:
        value = float(natural[fam.names.index(name)])
        out[name] = fam.start + value if name.startswith("tau") else value
    return out


def run_profile_curves(
    eng: Engine, reference: Result, points: int = 40
) -> list[dict[str, Any]]:
    """Profile each free knot on a grid, refitting the rates and the other knot."""
    fam = eng.fam
    rows = []
    for name in fam.free_knots:
        lo, hi = fam.free_ranges[list(fam.free_knots).index(name)]
        idx = fam.names.index(name)
        grid = np.linspace(lo, hi, points + 2)[1:-1]
        centre = float(reference.natural[idx])
        curve: dict[float, float] = {}
        for side in (grid[grid >= centre][::1], grid[grid < centre][::-1]):
            warm = np.array(reference.natural)
            for tau in side:
                compiled = eng._compile({name: float(tau)})
                problem = binned_problem(eng.arrays, eng.data.n, eng.model, compiled)
                nat = np.array(warm)
                nat[idx] = tau
                u = np.asarray(compiled.reduce(jnp.asarray(nat)), dtype=np.float64)
                if float(problem.nll(jnp.asarray(u))) >= _PENALTY:
                    continue
                res = solve(problem, compiled, [u], compute_cov=False)
                curve[float(tau)] = res.log_likelihood
                if np.isfinite(res.log_likelihood):
                    warm = res.natural
        taus = np.array(sorted(curve))
        lls = np.array([curve[t] for t in taus])
        inside = taus[2.0 * (reference.log_likelihood - lls) <= CHI2_95]
        rows.append(
            {
                "knot": name,
                "mle_year": fam.start + centre,
                "ci95_low_year": fam.start + float(inside.min())
                if len(inside)
                else float("nan"),
                "ci95_high_year": fam.start + float(inside.max())
                if len(inside)
                else float("nan"),
                "grid_edge_hit": bool(
                    len(inside)
                    and (inside.min() <= grid[0] or inside.max() >= grid[-1])
                ),
                "curve_years": (fam.start + taus).round(1).tolist(),
                "curve_ll": lls.round(3).tolist(),
            }
        )
    return rows


def run_real(
    families: list[str], data_dir: Path, seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    """One fit per method on each real family, and the reference optimum."""
    tables: dict[str, list[dict[str, Any]]] = {}
    reference: dict[str, np.ndarray] = {}
    for key in families:
        fam = make_family(key)
        data = load_real(fam, data_dir)
        eng = Engine(fam, data, seed)
        results: dict[str, Result] = {}
        for method in RANGE_METHODS:
            run_method(eng, method)  # warm-up: compile
            results[method] = run_method(eng, method)
            print(f"{key}: {method} done", flush=True)
        best_name = max(results, key=lambda m: results[m].log_likelihood)
        best = results[best_name]
        reference[key] = best.natural
        rows = []
        for method, res in results.items():
            rows.append(
                {
                    "method": method,
                    "converged": res.converged,
                    "seconds": res.seconds,
                    "work": res.work,
                    "log_lik": res.log_likelihood,
                    "gap_to_best": res.log_likelihood - best.log_likelihood,
                    "hit": res.log_likelihood >= best.log_likelihood - HIT_TOL,
                    **{
                        k: round(v, 4)
                        for k, v in _year_params(fam, res.natural).items()
                    },
                }
            )
        exact = run_exact_midpoint(fam, data)
        if exact is not None:
            rows.append(
                {
                    "method": "exact-midpoint",
                    "converged": exact.converged,
                    "seconds": exact.seconds,
                    "work": exact.work,
                    "log_lik": float("nan"),
                    "gap_to_best": float("nan"),
                    "hit": "",
                    **{
                        k: round(v, 4)
                        for k, v in _year_params(fam, exact.natural).items()
                    },
                }
            )
        tables[f"{key}: real data, n = {data.n} (knots in calendar years)"] = rows
        curves = run_profile_curves(eng, best)
        tables[f"{key}: knot profile likelihood (rates and other knot refitted)"] = [
            {k: v for k, v in c.items() if not k.startswith("curve")} for c in curves
        ]
        tables[f"{key}: curves"] = curves
    return tables, reference


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_ranges(
    fam: Family,
    truth: np.ndarray,
    real: RangeData,
    rng: np.random.Generator,
) -> RangeData:
    """Simulate one population and give each age a range of a resampled width."""
    birth, death = fam.model.rates(jnp.asarray(truth))
    lo_n, hi_n = max(5, real.n // 2), 2 * real.n
    while True:
        sample = simulate_ages_from_rates(birth, death, rng)
        if lo_n <= sample.population <= hi_n:
            break
    ages = np.asarray(sample.all_ages, dtype=np.float64)
    year = np.floor(CURRENT_YEAR - ages).astype(np.int64)
    width = rng.choice(real.widths, size=len(year))
    offset = np.floor(rng.uniform(size=len(year)) * width).astype(np.int64)
    lower, upper = clip_years(
        year - offset, year - offset + width, fam.start, fam.support_end
    )
    return RangeData(lower, upper, fam.horizon)


def run_sim(
    families: list[str],
    reference: dict[str, np.ndarray],
    data_dir: Path,
    replicates: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Fit every method on simulated range data and summarise."""
    tables: dict[str, list[dict[str, Any]]] = {}
    for key in families:
        fam = make_family(key)
        real = load_real(fam, data_dir)
        truth = np.array(reference[key])
        for name in fam.free_knots:  # whole-year knots
            i = fam.names.index(name)
            truth[i] = round(truth[i])
        rng = np.random.default_rng(seed)
        methods = [*RANGE_METHODS, "exact-midpoint"]
        out: dict[str, list[Result | None]] = {m: [] for m in methods}
        for rep in range(replicates[key]):
            data = simulate_ranges(fam, truth, real, rng)
            eng = Engine(fam, data, seed)
            for method in RANGE_METHODS:
                out[method].append(run_method(eng, method))
            try:
                out["exact-midpoint"].append(run_exact_midpoint(fam, data))
            except (ValueError, RuntimeError):
                out["exact-midpoint"].append(None)
            print(f"{key}: replicate {rep + 1}/{replicates[key]} done", flush=True)
        tables.update(_summarise(fam, truth, out))
    return tables


def _summarise(
    fam: Family, truth: np.ndarray, out: dict[str, list[Result | None]]
) -> dict[str, list[dict[str, Any]]]:
    reps = len(out["newton"])
    best_ll = np.array(
        [max(out[m][r].log_likelihood for m in RANGE_METHODS) for r in range(reps)]  # type: ignore[union-attr]
    )
    speed, answers, wald = [], [], []
    for method, results in out.items():
        have = [r for r in results if r is not None]
        if not have:
            continue
        row: dict[str, Any] = {
            "method": method,
            "converged": f"{sum(r.converged for r in have)}/{reps}",
            "median_seconds": statistics.median(r.seconds for r in have[1:] or have),
            "median_work": statistics.median(r.work for r in have),
        }
        if method in RANGE_METHODS:
            gap = np.array(
                [results[r].log_likelihood - best_ll[r] for r in range(reps)]  # type: ignore[union-attr]
            )
            row["hit_rate"] = float(np.mean(gap >= -HIT_TOL))
            row["median_gap"] = float(np.median(gap))
            row["worst_gap"] = float(gap.min())
        speed.append(row)
        est = np.array([r.natural for r in have])
        for name in fam.reported:
            j = fam.names.index(name)
            err = est[:, j] - truth[j]
            answers.append(
                {
                    "method": method,
                    "param": name,
                    "true": float(truth[j]),
                    "bias": float(np.nanmean(err)),
                    "median_abs_err": float(np.nanmedian(np.abs(err))),
                    "rmse": float(np.sqrt(np.nanmean(err**2))),
                }
            )
        if method in ("newton", "hybrid"):
            for name in fam.free_knots:
                j = fam.names.index(name)
                se = np.array([r.se[j] for r in have])
                err = np.abs(est[:, j] - truth[j])
                ok = np.isfinite(se)
                wald.append(
                    {
                        "method": method,
                        "knot": name,
                        "median_se": float(np.nanmedian(se))
                        if ok.any()
                        else float("nan"),
                        "empirical_sd": float(np.std(est[:, j], ddof=1)),
                        "coverage95": float(np.mean(err[ok] <= 1.96 * se[ok]))
                        if ok.any()
                        else float("nan"),
                        "n_finite_se": int(ok.sum()),
                    }
                )
    return {
        f"{fam.key}: speed and agreement with the best range fit": speed,
        f"{fam.key}: answers vs truth (knots in years since start)": answers,
        f"{fam.key}: knot Wald intervals": wald,
    }


# ---------------------------------------------------------------------------
# Known-truth simulation (sim1000), sharded for an HPC array job
# ---------------------------------------------------------------------------


def sim_truth(fam: Family) -> np.ndarray:
    """Return the natural-scale truth vector of ``sim1000`` in the model order."""
    out = []
    for name in fam.names:
        key = "death" if name.startswith("death.") else name
        out.append(SIM_TRUTH[key])
    return np.array(out)


def simulate_sim1000(
    fam: Family, truth: np.ndarray, rng: np.random.Generator
) -> RangeData:
    """One non-extinct population, each age turned into a range of resampled width."""
    birth, death = fam.model.rates(jnp.asarray(truth))
    sample = simulate_ages_from_rates(birth, death, rng, ensure_non_extinct=True)
    ages = np.asarray(sample.all_ages, dtype=np.float64)
    year = np.floor(CURRENT_YEAR - ages).astype(np.int64)
    width = rng.choice(
        [w for w, _ in JB_RANGE_WIDTHS],
        size=len(year),
        p=[q for _, q in JB_RANGE_WIDTHS],
    )
    offset = np.floor(rng.uniform(size=len(year)) * width).astype(np.int64)
    lower, upper = clip_years(
        year - offset, year - offset + width, fam.start, fam.support_end
    )
    return RangeData(lower, upper, fam.horizon)


def profile_interval(
    curve: tuple[np.ndarray, np.ndarray] | None, level: float = CHI2_95
) -> tuple[float, float]:
    """Likelihood-ratio interval from a profile curve, interpolating crossings."""
    if curve is None or len(curve[0]) < 3:
        return float("nan"), float("nan")
    taus, lls = curve
    cut = lls.max() - 0.5 * level
    inside = np.flatnonzero(lls >= cut)
    first, last = int(inside.min()), int(inside.max())

    def cross(a: int, b: int) -> float:
        return float(
            np.interp(cut, sorted([lls[a], lls[b]]), sorted([taus[a], taus[b]]))
        )

    low = taus[0] if first == 0 else cross(first - 1, first)
    high = taus[-1] if last == len(taus) - 1 else cross(last, last + 1)
    return float(low), float(high)


def _record(res: Result) -> dict[str, Any]:
    interval = profile_interval(res.curve)
    return {
        "natural": res.natural.tolist(),
        "log_likelihood": res.log_likelihood,
        "converged": res.converged,
        "seconds": res.seconds,
        "work": res.work,
        "se": res.se.tolist(),
        "profile_interval": list(interval),
    }


def run_sim1000(
    replicates: list[int], data_dir: Path, seed: int, methods: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Simulate and fit the replicates in ``replicates``; each has its own seed."""
    fam = make_family(SIM_KEY)
    truth = sim_truth(fam)
    records = []
    for position, rep in enumerate(replicates):
        rng = np.random.default_rng([seed, rep])
        data = simulate_sim1000(fam, truth, rng)
        eng = Engine(fam, data, seed + rep, pad=True)
        record: dict[str, Any] = {
            "replicate": rep,
            "n": data.n,
            "distinct_ranges": len(data.arrays[2]),
            "warm": position > 0,  # the first fit in a process carries compile time
            "methods": {},
        }
        for method in methods:
            record["methods"][method] = _record(run_method(eng, method))
            print(
                f"  rep {rep} {method}: {record['methods'][method]['seconds']:.1f}s",
                flush=True,
            )
        records.append(record)
        print(f"{SIM_KEY}: replicate {rep} done (n = {data.n})", flush=True)
    return records


def summarise_sim1000(
    records: list[dict[str, Any]], methods: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """Speed, optimum-hit rate, accuracy and interval coverage over all replicates."""
    fam = make_family(SIM_KEY)
    truth = sim_truth(fam)
    reps = len(records)
    tau = fam.names.index("tau0")
    best_ll = np.array(
        [max(r["methods"][m]["log_likelihood"] for m in methods) for r in records]
    )
    speed, answers, intervals, by_size = [], [], [], []
    sizes = np.array([r["n"] for r in records])
    for method in methods:
        got = [r["methods"][method] for r in records]
        ll = np.array([g["log_likelihood"] for g in got])
        gap = ll - best_ll
        secs = np.array(
            [g["seconds"] for r, g in zip(records, got, strict=True) if r["warm"]]
        )
        speed.append(
            {
                "method": method,
                "converged": f"{sum(g['converged'] for g in got)}/{reps}",
                "median_seconds": float(np.median(secs)) if len(secs) else float("nan"),
                "p90_seconds": float(np.percentile(secs, 90))
                if len(secs)
                else float("nan"),
                "total_seconds": float(sum(g["seconds"] for g in got)),
                "median_work": float(np.median([g["work"] for g in got])),
                "hit_rate": float(np.mean(gap >= -HIT_TOL)),
                "median_gap": float(np.median(gap)),
                "worst_gap": float(np.min(gap)),
            }
        )
        est = np.array([g["natural"] for g in got])
        ok = np.isfinite(ll)
        for name in fam.reported:
            j = fam.names.index(name)
            err = est[ok, j] - truth[j]
            answers.append(
                {
                    "method": method,
                    "param": name,
                    "true": float(truth[j]),
                    "bias": float(np.mean(err)),
                    "median_abs_err": float(np.median(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                }
            )
        se = np.array([g["se"][tau] for g in got], dtype=float)
        err_tau = np.abs(est[:, tau] - truth[tau])
        wald_ok = np.isfinite(se) & ok
        row: dict[str, Any] = {
            "method": method,
            "median_se": float(np.median(se[wald_ok]))
            if wald_ok.any()
            else float("nan"),
            "empirical_sd": float(np.std(est[ok, tau], ddof=1))
            if ok.sum() > 1
            else float("nan"),
            "wald_coverage95": float(np.mean(err_tau[wald_ok] <= 1.96 * se[wald_ok]))
            if wald_ok.any()
            else float("nan"),
            "n_finite_se": int(wald_ok.sum()),
        }
        pi = np.array([g["profile_interval"] for g in got], dtype=float)
        have = np.isfinite(pi).all(axis=1)
        if have.any():
            covered = (pi[have, 0] <= truth[tau]) & (truth[tau] <= pi[have, 1])
            row["profile_coverage95"] = float(covered.mean())
            row["profile_median_width"] = float(np.median(pi[have, 1] - pi[have, 0]))
        intervals.append(row)
        for label, mask in (
            ("n < median", sizes < np.median(sizes)),
            ("n >= median", sizes >= np.median(sizes)),
        ):
            if mask.any():
                by_size.append(
                    {
                        "method": method,
                        "group": label,
                        "hit_rate": float(np.mean(gap[mask] >= -HIT_TOL)),
                        "median_abs_tau_err": float(np.median(err_tau[mask])),
                    }
                )
    quantiles = np.percentile(sizes, [0, 25, 50, 75, 100])
    return {
        f"{SIM_KEY}: speed and agreement with the best fit ({reps} replicates)": speed,
        f"{SIM_KEY}: answers vs truth (knot in years since start)": answers,
        f"{SIM_KEY}: knot intervals": intervals,
        f"{SIM_KEY}: by data size": by_size,
        f"{SIM_KEY}: sample size n (min, q1, median, q3, max)": [
            dict(
                zip(
                    ["min", "q1", "median", "q3", "max"],
                    quantiles.tolist(),
                    strict=True,
                )
            )
        ],
    }


def sim1000_main(args: argparse.Namespace, data_dir: Path) -> None:
    """Run one shard, or merge the shard files, of the ``sim1000`` study."""
    methods = tuple(args.methods or SIM_METHODS)
    out_dir = args.out.parent
    stem = f"bench_knot_methods_{SIM_KEY}"
    if args.merge:
        records: list[dict[str, Any]] = []
        for path in sorted(out_dir.glob(f"{stem}_shard*.json")):
            records += json.loads(path.read_text())
        records.sort(key=lambda r: r["replicate"])
        tables = summarise_sim1000(records, methods)
        for title, rows in tables.items():
            _print(rows, title)
        (out_dir / f"{stem}.json").write_text(
            json.dumps(tables, indent=2, default=float)
        )
        print(f"\nMerged {len(records)} replicates into {out_dir / (stem + '.json')}")
        return
    total = args.replicates or (4 if args.quick else 200)
    index, count = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    mine = [r for r in range(total) if r % count == index]
    records = run_sim1000(mine, data_dir, args.seed, methods)
    name = f"{stem}_shard{index}.json" if args.shard else f"{stem}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(records, indent=2, default=float))
    if not args.shard:
        for title, rows in summarise_sim1000(records, methods).items():
            _print(rows, title)
    print(f"\nWrote {out_dir / name}")


# ---------------------------------------------------------------------------


def main() -> None:
    """Run the requested sections and print Markdown tables."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument(
        "--families",
        nargs="*",
        choices=("jB", "K-pc", "K-pl"),
        default=["jB", "K-pc", "K-pl"],
    )
    parser.add_argument("--quick", action="store_true", help="few replicates")
    parser.add_argument("--methods", nargs="*", choices=SIM_METHODS, default=None)
    parser.add_argument(
        "--shard", default=None, help="i/N: replicates r with r %% N == i"
    )
    parser.add_argument("--merge", action="store_true", help="merge shard files")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=Path("_output/bench_knot_methods.json")
    )
    args = parser.parse_args()
    data_dir = args.data_dir or (find_project_root().parent / "nt-manuscripts" / "data")
    if args.only == ["sim1000"]:
        sim1000_main(args, data_dir)
        return

    results: dict[str, Any] = {"backend": jax.default_backend()}
    if "check" in args.only:
        results["check"] = run_check(args.families, data_dir)
        _print(results["check"], "Checks on the real ranges")
    reference: dict[str, np.ndarray] = {}
    if "real" in args.only or "sim" in args.only:
        tables, reference = run_real(args.families, data_dir, args.seed)
        results["real"] = tables
        for title, rows in tables.items():
            if not title.endswith("curves"):
                _print(rows, title)
    if "sim" in args.only:
        default = {"jB": 30, "K-pc": 20, "K-pl": 20}
        reps = {
            k: args.replicates or (4 if args.quick else default[k])
            for k in args.families
        }
        results["sim"] = run_sim(args.families, reference, data_dir, reps, args.seed)
        for title, rows in results["sim"].items():
            _print(rows, title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
