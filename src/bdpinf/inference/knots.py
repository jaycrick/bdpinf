r"""Global search for free knot times by maximum likelihood.

With piecewise-constant rates the likelihood is discontinuous in a knot time
(it jumps where the knot crosses an observed $T - a$), so Newton steps over the
knots alone are unreliable.
With exact ages and piecewise-linear rates, or with ages known only to lie in
ranges (:meth:`~bdpinf.inference.likelihood.AgeData.from_ranges`), it is
continuous but can have several local maxima.
:func:`fit_knots` therefore *profiles* the knots: it fits the rates by Newton at
each candidate knot placement and searches over the placements.
The knots enter :func:`~bdpinf.inference.fitting.fit` as fixed parameters, so
the jitted objective is compiled once for the whole search.

One search runs by default, chosen from benchmarks on simulated and real data
(the figures below are for one or two free knots, where the default effort is
64 scanned placements, 3 joint polishes and 15 differential-evolution
generations; with ``k >= 3`` free knots the effort scales with ``k``, see
:func:`fit_knots`):

* one free knot: ``"brent"``, a bounded scalar search
  (70% of replicates at the best likelihood found, 31 s median);
* several free knots: ``"grid-newton"``, a coarse scan then joint Newton
  (49% at the best, 20 min median, against 52% in 69 min for differential
  evolution).

With ``thorough=True`` the knots are searched by ``"brent"`` (cyclic over the
knots when there are several), ``"de-profile"`` (differential evolution over
the profile) and ``"grid-newton"``, in parallel processes, and the best
placement is kept, because each method misses the global maximum on different
data sets (90% at the best likelihood for one knot, 91% for five).

``fit_knots(starts=...)`` adds extra knot placements (for example from
:func:`neighbour_placements`, with warm rates from :func:`nested_start`) that
are profiled and polished in every method and are candidates for the final
maximum, so the result is never worse than the best of them.
:class:`KnotFitResult` also reports how many distinct placements reach nearly
the best likelihood (``n_distinct_optima``, ``runner_up_gap``), to show
multimodality.

``FreeKnots(min_spacing=...)`` keeps knots apart: no placement closer than that
between any two sorted knots (fixed knots included) is evaluated as a candidate
or returned.

:func:`place_knots` and :func:`evenly_spaced_knots` are the public
counterparts of that spacing repair: they build starting knots that are inside
the support and at least ``min_spacing`` apart.
"""

import concurrent.futures
import dataclasses
import logging
import math
import multiprocessing
import os
import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
import scipy.optimize
import scipy.stats.qmc
from numpy.typing import ArrayLike, NDArray

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import FitResult, fit
from bdpinf.inference.likelihood import AgeData
from bdpinf.inference.rate_models import BirthDeathModel, FreeKnots

__all__ = [
    "KnotFitResult",
    "MethodResult",
    "evenly_spaced_knots",
    "fit_knots",
    "neighbour_placements",
    "nested_start",
    "place_knots",
]

logger = logging.getLogger(__name__)

_THOROUGH_METHODS = ("brent", "de-profile", "grid-newton")
_MAX_BRENT_SWEEPS = 5
_N_NEWTON_STARTS = 3
_DE_MAXITER = 15
_DE_MAXITER_MANY = 30
_DE_FEW_KNOTS = 3
_NEAR_OPTIMUM = 0.01
_LOG_EVERY = 10
_SLACK = 1e-9
_MIN_VALID_CANDIDATES = 4
_MAX_REDRAWS = 8
_DE_MIN_POPULATION = 5
# Extra gap added to ``min_spacing`` in ``place_knots`` so that the returned
# knots are at least ``min_spacing`` apart despite floating-point rounding.
_PLACE_TOL = 1e-6


@dataclasses.dataclass(frozen=True)
class MethodResult:
    """The best placement found by one search method.

    Attributes
    ----------
    knot_times : NDArray
        Knot times of the best placement (fixed knots included), increasing.
    log_likelihood : float
        Log-likelihood there, with the rates re-fitted.
    seconds : float
        Wall time of this method.
    n_fits : int
        Rate fits it needed.
    tried : dict[tuple[float, ...], float]
        Log-likelihood of every free-knot placement (rounded sorted knots) the
        process running this method had evaluated, including the joint fits.
        Used for the multimodality diagnostics of :class:`KnotFitResult`.

    """

    knot_times: NDArray[np.float64]
    log_likelihood: float
    seconds: float
    n_fits: int
    tried: dict[tuple[float, ...], float] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )


@dataclasses.dataclass(frozen=True)
class KnotFitResult:
    """Result of :func:`fit_knots`.

    Attributes
    ----------
    fit : FitResult
        The fit at the selected knot times.
        Its covariance is conditional on the knots unless the joint polish
        was accepted (``fit.constraints.fixed`` then has no free ``tau`` entries).
        Usable as ``init`` for :func:`~bdpinf.inference.fitting.fit` and
        :func:`~bdpinf.inference.bayesian.fit_bayesian`.
    knot_times : NDArray
        Selected knot times, increasing.
    log_likelihood : float
        Maximised log-likelihood.
    method : str
        The method whose placement was selected.
    method_results : dict[str, MethodResult]
        What every method found.
    n_candidates : int
        Placements scanned by ``"grid-newton"`` (the value used).
    n_polish : int
        Placements that ``"grid-newton"`` polished by joint Newton.
    de_maxiter : int
        Generations of differential evolution.
    n_starts : int
        Extra placements supplied through ``starts`` (those that were valid).
    n_distinct_optima : int
        Distinct placements (the best one included) whose log-likelihood is
        within 0.01 of the best.
        Two placements are distinct when some sorted free knot differs by more
        than ``max(min_spacing, 1e-3 * range)``.
        A value above 1 signals multimodality, or a flat ridge.
    runner_up_gap : float
        Log-likelihood gap from the best to the best placement that is distinct
        from it in that sense (``inf`` if none was evaluated).

    """

    fit: FitResult
    knot_times: NDArray[np.float64]
    log_likelihood: float
    method: str
    method_results: dict[str, MethodResult]
    n_candidates: int = 0
    n_polish: int = 0
    de_maxiter: int = 0
    n_starts: int = 0
    n_distinct_optima: int = 1
    runner_up_gap: float = float("inf")


def _spacing_ok(knots: Any, min_spacing: float) -> bool:
    """Whether the sorted ``knots`` are all at least ``min_spacing`` apart."""
    if min_spacing <= 0:
        return True
    ordered = np.sort(np.asarray(knots, dtype=np.float64))
    return bool(np.all(np.diff(ordered) >= min_spacing - _SLACK))


def _repair(
    rows: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    spacing: float,
) -> NDArray[np.float64]:
    """Push each sorted row apart to ``spacing`` inside its ranges (vectorised)."""
    out = np.clip(np.sort(rows, axis=1), lo, hi)
    for j in range(1, out.shape[1]):
        out[:, j] = np.maximum(out[:, j], out[:, j - 1] + spacing)
    for j in range(out.shape[1] - 1, -1, -1):
        out[:, j] = np.minimum(out[:, j], hi[j])
        if j + 1 < out.shape[1]:
            out[:, j] = np.minimum(out[:, j], out[:, j + 1] - spacing)
    return out


def _valid_rows(
    rows: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    spacing: float,
    fixed: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Rows inside the ranges and ``spacing`` apart (fixed knots included)."""
    ok = np.all((rows >= lo - _SLACK) & (rows <= hi + _SLACK), axis=1)
    if rows.shape[1] > 1:
        ok &= np.all(np.diff(rows, axis=1) >= spacing - _SLACK, axis=1)
    for f in fixed:
        ok &= np.all(np.abs(rows - f) >= spacing - _SLACK, axis=1)
    return ok


def _candidates(
    ranges: Sequence[tuple[float, float]],
    n_candidates: int,
    seed: int | None,
    *,
    min_spacing: float = 0.0,
    fixed: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Candidate knot placements, one row each (sorted within the row).

    With ``min_spacing > 0`` violating rows are repaired or dropped and the
    set is redrawn larger when too few valid rows remain.

    Raises
    ------
    ValueError
        If no placement at that spacing exists.

    """
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    fixed_knots = np.asarray([] if fixed is None else fixed, dtype=np.float64)
    if min_spacing <= 0:
        if len(ranges) == 1:
            return np.linspace(lo[0], hi[0], n_candidates + 2)[1:-1, None]
        sobol = scipy.stats.qmc.Sobol(
            d=len(ranges), scramble=True, rng=np.random.default_rng(seed)
        )
        unit = sobol.random_base2(max(1, math.ceil(math.log2(n_candidates))))
        return np.sort(lo + (hi - lo) * unit, axis=1)

    rows = np.empty((0, len(ranges)))
    size = n_candidates
    for attempt in range(_MAX_REDRAWS):
        if len(ranges) == 1:
            grid = np.linspace(lo[0], hi[0], size + 2)[1:-1, None]
        else:
            sobol = scipy.stats.qmc.Sobol(
                d=len(ranges),
                scramble=True,
                rng=np.random.default_rng(None if seed is None else seed + attempt),
            )
            unit = sobol.random_base2(max(1, math.ceil(math.log2(size))))
            grid = _repair(lo + (hi - lo) * unit, lo, hi, min_spacing)
        grid = grid[_valid_rows(grid, lo, hi, min_spacing, fixed_knots)]
        rows = np.unique(np.round(grid, 9), axis=0)
        if len(rows) >= min(_MIN_VALID_CANDIDATES, n_candidates):
            return rows
        size *= 4
    if len(rows) == 0:
        msg = (
            f"No placement of {len(ranges)} free knots at least {min_spacing} "
            "apart (and from the fixed knots) fits in the knot ranges"
        )
        raise ValueError(msg)
    return rows


def _key(taus: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(t) for t in np.round(np.sort(taus), 9))


@dataclasses.dataclass(frozen=True)
class _Search:
    """Everything a search method needs, picklable for worker processes."""

    data: AgeData
    model: BirthDeathModel
    user: Constraints
    free_idx: list[int]
    n_candidates: int
    seed: int | None
    init: Any
    fit_kwargs: dict[str, Any]
    n_polish: int = _N_NEWTON_STARTS
    de_maxiter: int = _DE_MAXITER
    starts: tuple[tuple[float, ...], ...] = ()
    start_rates: tuple[tuple[float, ...] | None, ...] = ()


@dataclasses.dataclass
class _BestFit:
    """The best rate fit seen so far by any method, used to warm-start later fits."""

    ll: float = -np.inf
    natural: NDArray[np.float64] | None = None


@dataclasses.dataclass
class _MethodTrack:
    """The best placement and the fit count of the method currently running."""

    ll: float = -np.inf
    taus: NDArray[np.float64] | None = None
    natural: NDArray[np.float64] | None = None
    fits: int = 0


class _KnotSearch:
    """The three knot searches over one job, sharing a cache of rate fits.

    The quantities derived from the job (free knot names and ranges, spacing,
    fixed knots, tolerances) are set once.
    ``tried`` (log-likelihood of every placement already evaluated) and
    ``best`` (best rate fit, the warm start for later fits) persist across the
    methods run in one call, on purpose: a later method reuses the fits and the
    warm start of an earlier one.
    Only ``track`` (the best placement of the method that is running, and its
    fit count) is reset before each method.
    """

    def __init__(self, job: _Search) -> None:
        knots = job.model.knots
        if knots is None:  # unreachable: checked by fit_knots
            raise RuntimeError
        self.job = job
        self.knots = knots
        self.free_idx = job.free_idx
        self.free_names = [knots.names[i] for i in job.free_idx]
        self.ranges = [knots.ranges[i] for i in job.free_idx]
        self.initial = np.array([knots.initial_times[i] for i in job.free_idx])
        self.lo = np.array([r[0] for r in self.ranges])
        self.hi = np.array([r[1] for r in self.ranges])
        self.tol = 1e-3 * (self.hi - self.lo)
        self.n_rates = job.model.n_birth + job.model.n_death
        self.fixed_knots = {
            n: job.user.fixed[n] for n in knots.names if n in job.user.fixed
        }
        self.spacing = knots.min_spacing
        self.fixed_vals = np.array(sorted(self.fixed_knots.values()), dtype=np.float64)
        self.tried: dict[tuple[float, ...], float] = {}
        self.naturals: dict[tuple[float, ...], NDArray[np.float64]] = {}
        self.joint: dict[tuple[float, ...], float] = {}
        self.start_joint: dict[tuple[float, ...], FitResult | None] = {}
        self.warm: dict[tuple[float, ...], NDArray[np.float64]] = {}
        self.starts: list[NDArray[np.float64]] = []
        for i, row in enumerate(job.starts):
            free = np.sort(np.asarray(row, dtype=np.float64))
            self.starts.append(free)
            rates = job.start_rates[i] if i < len(job.start_rates) else None
            if rates is not None:
                self.warm.setdefault(_key(free), np.asarray(rates, dtype=np.float64))
        self.best = _BestFit()
        self.track = _MethodTrack()

    def run(
        self, names: Sequence[str]
    ) -> dict[str, tuple[MethodResult, NDArray[np.float64]]]:
        """Run the named searches in order; return each one's result and vector."""
        searches = {
            "brent": self._brent,
            "de-profile": self._de_profile,
            "grid-newton": self._grid_newton,
        }
        found: dict[str, tuple[MethodResult, NDArray[np.float64]]] = {}
        for name in names:
            self.track = _MethodTrack()
            began = time.perf_counter()
            self._seed_starts()
            searches[name]()
            track = self.track
            if track.taus is None or track.natural is None or not np.isfinite(track.ll):
                logger.debug("method %s found no finite placement", name)
                continue
            found[name] = (
                MethodResult(
                    knot_times=self.all_knots(track.taus),
                    log_likelihood=float(track.ll),
                    seconds=time.perf_counter() - began,
                    n_fits=int(track.fits),
                    tried={**self.tried, **self.joint},
                ),
                track.natural,
            )
        return found

    def valid(self, free: NDArray[np.float64]) -> bool:
        """Whether the free knots and the fixed ones respect ``min_spacing``."""
        if self.spacing <= 0:
            return True
        return _spacing_ok(
            np.concatenate([np.asarray(free), self.fixed_vals]), self.spacing
        )

    def intervals(self, low: float, high: float) -> list[tuple[float, float]]:
        """``[low, high]`` minus the zones within ``spacing`` of a fixed knot."""
        parts = [(low, high)]
        if self.spacing <= 0:
            return parts
        for f in self.fixed_vals:
            cut: list[tuple[float, float]] = []
            for a, b in parts:
                left = min(b, f - self.spacing)
                right = max(a, f + self.spacing)
                if left > a:
                    cut.append((a, left))
                if b > right:
                    cut.append((right, b))
            parts = cut
        return parts

    def bounded(
        self, fun: Callable[[float], float], low: float, high: float, xatol: float
    ) -> tuple[float, float] | None:
        """Best ``(x, -fun)`` of a bounded scalar search over the valid intervals."""
        best_step = None
        for a, b in self.intervals(low, high):
            if not b > a:
                continue
            step = scipy.optimize.minimize_scalar(
                fun, bounds=(a, b), method="bounded", options={"xatol": xatol}
            )
            if best_step is None or step.fun < best_step.fun:
                best_step = step
        return None if best_step is None else (float(best_step.x), -best_step.fun)

    def all_knots(self, free: NDArray[np.float64]) -> NDArray[np.float64]:
        """Every knot time (free ones sorted, fixed ones as given), in model order."""
        values = (
            dict(zip(self.free_names, np.sort(free), strict=True)) | self.fixed_knots
        )
        return np.array([values[n] for n in self.knots.names])

    def fit_at(
        self,
        free: NDArray[np.float64],
        warm: NDArray[np.float64] | None = None,
        **extra: Any,
    ) -> FitResult:
        """Fit the rates with the free knots held at ``free`` (warm-started).

        ``warm`` (rate values) overrides the best fit so far as the start.
        """
        job = self.job
        fixed = {
            **job.user.fixed,
            **dict(zip(self.free_names, np.sort(free), strict=True)),
        }
        start = job.init
        if warm is not None:
            start = np.concatenate([warm[: self.n_rates], self.all_knots(free)])
        elif self.best.natural is not None:
            start = np.concatenate(
                [self.best.natural[: self.n_rates], self.all_knots(free)]
            )
        return fit(
            job.data,
            job.model,
            constraints=Constraints(fixed=fixed, equal=job.user.equal),
            init=start,
            **{**job.fit_kwargs, **extra},
        )

    def record(
        self, value: float, taus: NDArray[np.float64], natural: ArrayLike
    ) -> None:
        """Keep the placement as the running method's best if it improves."""
        if value > self.track.ll:
            self.track.ll = value
            self.track.taus = taus
            self.track.natural = np.asarray(natural)

    def profile(self, free: NDArray[np.float64]) -> float:
        """Log-likelihood with the free knots at ``free`` (``-inf`` if infeasible)."""
        free = np.sort(np.asarray(free, dtype=np.float64))
        key = _key(free)
        if key in self.tried:
            return self.tried[key]
        if not self.valid(free):
            self.tried[key] = -np.inf
            return -np.inf
        self.track.fits += 1
        try:
            result = self.fit_at(
                free, self.warm.pop(key, None), compute_covariance=False
            )
        except ValueError as exc:
            logger.debug("no fit at knots %s: %s", key, exc)
            self.tried[key] = value = -np.inf
            return value
        self.tried[key] = value = result.log_likelihood
        self.naturals[key] = np.asarray(result.natural)
        if value > self.best.ll:
            self.best.ll, self.best.natural = value, result.natural
        self.record(value, free, result.natural)
        if len(self.tried) % _LOG_EVERY == 0:
            logger.info(
                "fit_knots: %d rate fits, best log-likelihood %.4f",
                len(self.tried),
                self.best.ll,
            )
        return value

    def _brent(self) -> None:
        """Bounded Brent over one knot, or cyclic coordinate sweeps over several."""
        if len(self.ranges) == 1:
            self._brent_single()
            return
        current = self._brent_start()
        self.profile(current)
        for _ in range(_MAX_BRENT_SWEEPS):
            if self._sweep(current) <= float(np.max(self.tol)):
                break

    def _brent_single(self) -> None:
        """Bounded scalar search over the only free knot."""

        def single(t: float) -> float:
            return -self.profile(np.array([t]))

        self.bounded(single, self.lo[0], self.hi[0], float(self.tol[0]))

    def _brent_start(self) -> NDArray[np.float64]:
        """Return the initial knots, or a valid candidate if they break spacing."""
        current = self.initial.copy()
        if not self.valid(current):
            current = _candidates(
                self.ranges,
                self.job.n_candidates,
                self.job.seed,
                min_spacing=self.spacing,
                fixed=self.fixed_vals,
            )[0]
        return current

    def _sweep(self, current: NDArray[np.float64]) -> float:
        """One pass over the knots, updating ``current``; the largest move made."""
        moved = 0.0
        for j in range(len(self.ranges)):
            moved = max(moved, self._coordinate_step(current, j))
        return moved

    def _coordinate_step(self, current: NDArray[np.float64], j: int) -> float:
        """Move knot ``j`` of ``current`` to its best valid place; return the move."""
        low = max(self.lo[j], current[j - 1] + self.spacing) if j > 0 else self.lo[j]
        high = (
            min(self.hi[j], current[j + 1] - self.spacing)
            if j + 1 < len(current)
            else self.hi[j]
        )
        if not high > low:
            return 0.0

        def along(t: float) -> float:
            trial = current.copy()
            trial[j] = t
            return -self.profile(trial)

        step = self.bounded(along, low, high, float(self.tol[j]))
        if step is not None and step[1] >= self.profile(current):
            moved = abs(step[0] - current[j])
            current[j] = step[0]
            return moved
        return 0.0

    def _de_population(self) -> Any:
        """Return the initial population for differential evolution."""
        if self.spacing <= 0:
            return "sobol"
        population = _candidates(
            self.ranges,
            8 * len(self.ranges),
            self.job.seed,
            min_spacing=self.spacing,
            fixed=self.fixed_vals,
        )
        if len(population) < _DE_MIN_POPULATION:
            population = np.resize(population, (_DE_MIN_POPULATION, len(self.ranges)))
        return population

    def _de_profile(self) -> None:
        """Differential evolution over the profile likelihood."""
        population = self._de_population()
        with warnings.catch_warnings():
            # Sobol wants a power-of-two population.
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol",
                category=UserWarning,
            )
            scipy.optimize.differential_evolution(
                lambda t: -self.profile(t),
                self.ranges,
                popsize=8,
                maxiter=self.job.de_maxiter,
                tol=1e-4,
                seed=self.job.seed,
                polish=False,
                init=population,
            )

    def _grid_newton(self) -> None:
        """Scan candidate placements, then joint Newton from the best few."""
        candidates = _candidates(
            self.ranges,
            self.job.n_candidates,
            self.job.seed,
            min_spacing=self.spacing,
            fixed=self.fixed_vals,
        )
        for row in candidates:
            self.profile(row)
        order = np.argsort([-self.tried[_key(row)] for row in candidates])
        for row in candidates[order[: self.job.n_polish]]:
            joint = self._joint_fit_from(row)
            if joint is not None:
                self._record_joint(joint)

    def _seed_starts(self) -> None:
        """Profile and joint-polish every user start; they are candidates."""
        for free in self.starts:
            value = self.profile(free)
            key = _key(free)
            natural = self.naturals.get(key)
            if np.isfinite(value) and natural is not None:
                self.record(value, free, natural)
            if not np.isfinite(value):
                continue
            if key not in self.start_joint:
                self.start_joint[key] = self._joint_fit_from(free, natural)
            joint = self.start_joint[key]
            if joint is not None:
                self._record_joint(joint)

    def _joint_fit_from(
        self, row: NDArray[np.float64], start_natural: ArrayLike | None = None
    ) -> FitResult | None:
        """Joint fit of rates and knots started at knots ``row`` (None if it fails).

        ``start_natural`` (a profile fit at ``row``) skips the profile fit.
        """
        job = self.job
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if start_natural is None:
                    start_natural = self.fit_at(row, compute_covariance=False).natural
                return fit(
                    job.data,
                    job.model,
                    constraints=job.user,
                    init=start_natural,
                    compute_covariance=False,
                    **job.fit_kwargs,
                )
            except ValueError:
                return None

    def _record_joint(self, joint: FitResult) -> None:
        """Record a joint fit if it is finite and keeps the knots apart."""
        knot_part = np.asarray(joint.natural)[self.n_rates :]
        if np.isfinite(joint.log_likelihood) and _spacing_ok(knot_part, self.spacing):
            key = _key(knot_part[self.free_idx])
            self.joint[key] = max(self.joint.get(key, -np.inf), joint.log_likelihood)
            self.record(joint.log_likelihood, knot_part[self.free_idx], joint.natural)


def _run_methods(
    names: Sequence[str], job: _Search
) -> dict[str, tuple[MethodResult, NDArray[np.float64]]]:
    """Run the named searches; return each one's result and rates-and-knots vector."""
    return _KnotSearch(job).run(names)


def _run_one(
    job: tuple[str, _Search],
) -> dict[str, tuple[MethodResult, NDArray[np.float64]]]:
    return _run_methods((job[0],), job[1])


def _run_parallel(
    names: Sequence[str], workers: int, job: _Search
) -> dict[str, tuple[MethodResult, NDArray[np.float64]]]:
    """Run each method in its own single-threaded process."""
    old = os.environ.get("XLA_FLAGS")
    flags = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["XLA_FLAGS"] = f"{old} {flags}" if old else flags
    try:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(names)), mp_context=context
        ) as pool:
            parts = list(pool.map(_run_one, [(n, job) for n in names]))
    finally:
        if old is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = old
    return {n: r for part in parts for n, r in part.items()}


def _check_inputs(
    model: BirthDeathModel, constraints: Constraints | None
) -> tuple[FreeKnots, Constraints, list[int]]:
    """Validate the model and constraints; return the knots, constraints, free knots."""
    knots = model.knots
    if knots is None:
        msg = "fit_knots needs a model with free knots (see FreeKnots)"
        raise ValueError(msg)
    user = constraints or Constraints()
    if set(knots.names) & {n for g in user.equal for n in g}:
        msg = "Knot times cannot appear in constraints for fit_knots"
        raise ValueError(msg)
    free_idx = [i for i, n in enumerate(knots.names) if n not in user.fixed]
    if not free_idx:
        msg = "Knot times cannot all be fixed for fit_knots"
        raise ValueError(msg)
    fixed_knots = [user.fixed[n] for n in knots.names if n in user.fixed]
    if not _spacing_ok(fixed_knots, knots.min_spacing):
        msg = f"Fixed knots are closer than min_spacing ({knots.min_spacing})"
        raise ValueError(msg)
    return knots, user, free_idx


def _polished(
    data: AgeData,
    model: BirthDeathModel,
    user: Constraints,
    chosen: FitResult,
    fit_kwargs: dict[str, Any],
) -> FitResult:
    """Joint fit from ``chosen``, kept if it converges and is no worse."""
    knots = model.knots
    if knots is None:  # unreachable: checked by fit_knots
        raise RuntimeError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            joint = fit(
                data,
                model,
                constraints=user,
                init=chosen.natural,
                **fit_kwargs,
            )
        except ValueError:
            return chosen
    if (
        _spacing_ok(joint.knot_times, knots.min_spacing)
        and joint.converged
        and joint.log_likelihood >= chosen.log_likelihood - 1e-9
    ):
        return joint
    return chosen


def fit_knots(  # noqa: PLR0913
    data: AgeData,
    model: BirthDeathModel,
    *,
    constraints: Constraints | None = None,
    thorough: bool = False,
    workers: int | None = None,
    n_candidates: int | None = None,
    n_polish: int | None = None,
    polish: bool = True,
    seed: int | None = None,
    starts: Sequence[ArrayLike] = (),
    start_rates: Sequence[ArrayLike | None] | None = None,
    **fit_kwargs: Any,
) -> KnotFitResult:
    r"""Fit rates and knot times by profile likelihood over the knots.

    Parameters
    ----------
    data : AgeData
        Observed ages or age ranges.
    model : BirthDeathModel
        Must have free knots (built from a
        :class:`~bdpinf.inference.rate_models.FreeKnots`); the number of knots,
        their allowed ranges and their ``min_spacing`` come from there.
    constraints : Constraints, optional
        Constraints on the rate parameters.
        Knots may be fixed (for example ``fixed={"tau1": 800.0}``) and are then
        left out of the search, but cannot appear in equality groups.
    thorough : bool
        By default one search runs: ``"brent"`` for one free knot and
        ``"grid-newton"`` for several.
        With ``thorough=True`` the knots are searched by ``"brent"`` (bounded
        Brent, cyclic over the knots when there are several), ``"de-profile"``
        (differential evolution over the profile) and ``"grid-newton"`` (a scan
        of ``n_candidates`` placements, then joint Newton from the best three),
        and the best placement is kept.
    workers : int, optional
        Parallel processes for ``thorough=True`` (single-threaded each).
        Default: one per method, up to the number of cores.
        ``workers=1`` runs the methods one after another.
        The processes recompile the objective unless
        ``JAX_COMPILATION_CACHE_DIR`` is set.
    n_candidates : int, optional
        Placements scanned by ``"grid-newton"``: an even grid for one knot, a
        scrambled Sobol sequence (rounded up to a power of two) for several.
        Default: ``max(64, 2**ceil(log2(32 k)))`` for ``k`` free knots
        (64 for ``k <= 2``).
    n_polish : int, optional
        Best scanned placements that ``"grid-newton"`` polishes by joint
        Newton.
        Default: ``max(3, k)``.
        Differential evolution runs 15 generations for ``k <= 3`` free knots
        and 30 beyond.
        The values used are recorded on the result.
    polish : bool
        Finish with a joint fit of rates and knots from the best placement,
        kept if it converges without lowering the likelihood.
        This gives knot standard errors for smooth rates.
    seed : int, optional
        Seed for the Sobol sequence and differential evolution.
    starts : sequence of array-like, optional
        Extra placements of the *free* knots (one row of ``k`` times each,
        in any order).
        In every method each is profiled and then polished by a joint fit of
        rates and knots, and each is a candidate for the final maximum, so the
        returned log-likelihood is at least the best profiled start (to
        1e-9).
        Starts that break ``min_spacing`` are skipped.
        See :func:`neighbour_placements`.
    start_rates : sequence of array-like, optional
        Parallel to ``starts``: a natural-scale parameter vector (rates then
        knots, as from :func:`nested_start`) whose rate part warm-starts the
        first fit at that placement (``None`` entries are allowed).
    **fit_kwargs
        Forwarded to :func:`~bdpinf.inference.fitting.fit` (for example
        ``n_starts``, ``n_nodes``, or ``init`` for the first placement).

    Returns
    -------
    KnotFitResult

    Raises
    ------
    ValueError
        If the model has no free knots, every knot is fixed, constraints put a
        knot in an equality group, fixed knots are closer than ``min_spacing``,
        no placement fits at that spacing, none gives a finite likelihood, or a
        start has the wrong length, lies outside the knot ranges, or
        ``start_rates`` does not match ``starts``.

    """
    knots, user, free_idx = _check_inputs(model, constraints)
    fit_kwargs.pop("compute_covariance", None)
    init = fit_kwargs.pop("init", None)
    k = len(free_idx)
    default_candidates, default_polish, de_maxiter = _default_effort(k)
    n_candidates = default_candidates if n_candidates is None else n_candidates
    n_polish = default_polish if n_polish is None else n_polish
    clean_starts, clean_rates = _check_starts(
        starts, start_rates, [knots.ranges[i] for i in free_idx]
    )
    job = _Search(
        data,
        model,
        user,
        free_idx,
        n_candidates,
        seed,
        init,
        dict(fit_kwargs),
        n_polish,
        de_maxiter,
        clean_starts,
        clean_rates,
    )

    if not thorough:
        names = ("brent" if len(free_idx) == 1 else "grid-newton",)
    else:
        names = _THOROUGH_METHODS
    n_workers = workers if workers is not None else min(len(names), os.cpu_count() or 1)
    if len(names) > 1 and n_workers > 1:
        found = _run_parallel(names, n_workers, job)
    else:
        found = _run_methods(names, job)
    if not found:
        msg = "No knot placement gives a finite likelihood"
        raise ValueError(msg)

    winner = max(found, key=lambda m: found[m][0].log_likelihood)
    best_result, best_natural = found[winner]
    free_names = [knots.names[i] for i in free_idx]
    fixed = {
        **user.fixed,
        **{
            n: float(best_result.knot_times[i])
            for i, n in zip(free_idx, free_names, strict=True)
        },
    }
    chosen = fit(
        data,
        model,
        constraints=Constraints(fixed=fixed, equal=user.equal),
        init=best_natural,
        **fit_kwargs,
    )
    if polish:
        chosen = _polished(data, model, user, chosen, fit_kwargs)
    if chosen.log_likelihood < best_result.log_likelihood - 1e-6:
        logger.warning(
            "fit_knots: final fit log-likelihood %.6f is below the %s search "
            "value %.6f; the reported fit may not be the best placement found",
            chosen.log_likelihood,
            winner,
            best_result.log_likelihood,
        )

    knot_times = chosen.knot_times
    if knot_times is None:  # unreachable: the model has free knots
        raise RuntimeError
    tried: dict[tuple[float, ...], float] = {}
    for result, _ in found.values():
        tried.update(result.tried)
    tried[_key(knot_times[free_idx])] = chosen.log_likelihood
    n_distinct, gap = _diagnostics(
        tried,
        _key(knot_times[free_idx]),
        chosen.log_likelihood,
        np.maximum(knots.min_spacing, 1e-3 * (_range_widths(knots, free_idx))),
    )
    return KnotFitResult(
        fit=chosen,
        knot_times=knot_times,
        log_likelihood=chosen.log_likelihood,
        method=winner,
        method_results={n: r for n, (r, _) in found.items()},
        n_candidates=n_candidates,
        n_polish=n_polish,
        de_maxiter=job.de_maxiter,
        n_starts=len(clean_starts),
        n_distinct_optima=n_distinct,
        runner_up_gap=gap,
    )


def _default_effort(k: int) -> tuple[int, int, int]:
    """Default ``(n_candidates, n_polish, de_maxiter)`` for ``k`` free knots."""
    return (
        max(64, 2 ** math.ceil(math.log2(32 * k))),
        max(_N_NEWTON_STARTS, k),
        _DE_MAXITER if k <= _DE_FEW_KNOTS else _DE_MAXITER_MANY,
    )


def _range_widths(knots: FreeKnots, free_idx: list[int]) -> NDArray[np.float64]:
    """Range width of each free knot."""
    return np.array([knots.ranges[i][1] - knots.ranges[i][0] for i in free_idx])


def _check_starts(
    starts: Sequence[ArrayLike],
    start_rates: Sequence[ArrayLike | None] | None,
    ranges: Sequence[tuple[float, float]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...] | None, ...]]:
    """Validate ``starts`` and ``start_rates``; return them as tuples."""
    if start_rates is not None and len(start_rates) != len(starts):
        msg = "start_rates must have one entry per start"
        raise ValueError(msg)
    rows: list[tuple[float, ...]] = []
    rates: list[tuple[float, ...] | None] = []
    for i, row in enumerate(starts):
        free = np.sort(np.atleast_1d(np.asarray(row, dtype=np.float64)))
        if free.shape != (len(ranges),):
            msg = f"start {i} must list {len(ranges)} free knot times"
            raise ValueError(msg)
        if any(
            not (lo - _SLACK <= t <= hi + _SLACK)
            for t, (lo, hi) in zip(free, ranges, strict=True)
        ):
            msg = f"start {i} lies outside the knot ranges"
            raise ValueError(msg)
        rows.append(tuple(float(t) for t in free))
        given = None if start_rates is None else start_rates[i]
        rates.append(
            None if given is None else tuple(float(v) for v in np.ravel(given))
        )
    return tuple(rows), tuple(rates)


def _diagnostics(
    tried: dict[tuple[float, ...], float],
    best_key: tuple[float, ...],
    best_ll: float,
    tol: NDArray[np.float64],
) -> tuple[int, float]:
    """Count distinct near-best placements and the gap to the runner-up."""
    best = np.asarray(best_key)
    distinct: list[NDArray[np.float64]] = [best]
    runner = -np.inf
    for key, value in sorted(tried.items(), key=lambda kv: -kv[1]):
        if not np.isfinite(value):
            continue
        row = np.asarray(key)
        if not np.any(np.abs(row - best) > tol):
            continue
        runner = max(runner, value)
        if value >= best_ll - _NEAR_OPTIMUM and all(
            np.any(np.abs(row - other) > tol) for other in distinct
        ):
            distinct.append(row)
    gap = float("inf") if not np.isfinite(runner) else max(0.0, best_ll - runner)
    return len(distinct), gap


def nested_start(
    model: BirthDeathModel,
    lower_fit: "FitResult | KnotFitResult",
    new_knots: ArrayLike,
) -> jax.Array:
    """Natural parameters of ``model`` reproducing a fit of a smaller model.

    The birth and death rate functions of ``lower_fit`` are re-expressed in
    the parametrisation of ``model`` with its free knots at ``new_knots``
    (:meth:`~bdpinf.inference.rate_models.BirthDeathModel.natural_from_rates`).
    For continuous piecewise-linear rates, and for the ``quadratic``,
    ``cubic_natural`` and ``cubic_clamped`` splines, a knot set that contains
    the knots of the lower fit gives the same functions, hence the same
    log-likelihood: inserting a knot does not change the fit, so the larger
    model can only improve on it.
    For other families the rates are only approximated.

    Parameters
    ----------
    model : BirthDeathModel
        The larger model (free knots; same families as the lower model).
    lower_fit : FitResult or KnotFitResult
        A fit of the smaller model; only its fitted ``birth`` and ``death``
        rate functions are used.
    new_knots : array-like
        Every knot time of ``model`` (all of them, not only the free ones),
        in any order.

    Returns
    -------
    jax.Array
        Natural parameters of ``model``, usable as ``init`` or as an entry of
        ``start_rates`` of :func:`fit_knots`.

    """
    fit_result = lower_fit.fit if isinstance(lower_fit, KnotFitResult) else lower_fit
    times = np.sort(np.asarray(new_knots, dtype=np.float64))
    return model.natural_from_rates(fit_result.birth, fit_result.death, times)


def _insertions(
    base: NDArray[np.float64],
    low_edge: float,
    high_edge: float,
    *,
    min_spacing: float,
    n_random: int,
    rng: np.random.Generator | int | None,
) -> list[NDArray[np.float64]]:
    """Insert one knot into ``base`` at each gap midpoint and random time."""
    edges = np.concatenate([[low_edge], base, [high_edge]])
    points = list((edges[:-1] + edges[1:]) / 2)
    if n_random > 0:
        generator = np.random.default_rng(rng)
        points += list(generator.uniform(low_edge, high_edge, size=n_random))
    rows = []
    for point in points:
        try:
            placed = place_knots([*base, point], low_edge, high_edge, min_spacing)
        except ValueError:
            continue
        rows.append(np.array(placed))
    return rows


def neighbour_placements(
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    *,
    ranges: Sequence[tuple[float, float]],
    min_spacing: float = 0.0,
    fixed: ArrayLike = (),
    rng: np.random.Generator | int | None = None,
    n_random: int = 0,
) -> list[NDArray[np.float64]]:
    """Placements of ``k`` free knots next to a fit with ``k - 1`` or ``k + 1``.

    From ``lower`` (``k - 1`` knots) one knot is inserted at the midpoint of
    every gap, including the gaps to the two ends of the support, and
    ``n_random`` more are inserted at uniform random times; each result is
    repaired by :func:`place_knots`.
    From ``upper`` (``k + 1`` knots) each knot is deleted in turn.
    Rows that are not valid or duplicate an earlier one are dropped.

    The support of the knots is ``[min lo, max hi]`` over ``ranges``; the
    insertion gaps and random times use that overall support.
    A row is valid when its sorted ``j``-th knot lies in ``ranges[j]``, all
    knots (and the ``fixed`` ones) are at least ``min_spacing`` apart, so
    placements that need a different range layout are dropped.

    Parameters
    ----------
    lower, upper : array-like, optional
        Free-knot placements of the fits with ``k - 1`` and ``k + 1`` knots.
    ranges : sequence of (float, float)
        Allowed range of each of the ``k`` sorted free knots.
    min_spacing : float
        Minimum distance between knots.
    fixed : array-like
        Fixed knot times that also have to be ``min_spacing`` away.
    rng : numpy.random.Generator or int, optional
        Randomness for ``n_random``.
    n_random : int
        Random insertions from ``lower``.

    Returns
    -------
    list of NDArray
        Sorted placements of ``k`` knots, in a deterministic order.

    Raises
    ------
    ValueError
        If ``lower`` or ``upper`` has the wrong length, or neither is given.

    """
    k = len(ranges)
    lo = np.array([r[0] for r in ranges], dtype=np.float64)
    hi = np.array([r[1] for r in ranges], dtype=np.float64)
    low_edge, high_edge = float(lo.min()), float(hi.max())
    fixed_knots = np.atleast_1d(np.asarray(fixed, dtype=np.float64))
    raw: list[NDArray[np.float64]] = []
    if lower is None and upper is None:
        msg = "give lower and/or upper"
        raise ValueError(msg)
    if lower is not None:
        base = np.sort(np.atleast_1d(np.asarray(lower, dtype=np.float64)))
        if len(base) != k - 1:
            msg = f"lower must have {k - 1} knots, not {len(base)}"
            raise ValueError(msg)
        raw.extend(
            _insertions(
                base,
                low_edge,
                high_edge,
                min_spacing=min_spacing,
                n_random=n_random,
                rng=rng,
            )
        )
    if upper is not None:
        above = np.sort(np.atleast_1d(np.asarray(upper, dtype=np.float64)))
        if len(above) != k + 1:
            msg = f"upper must have {k + 1} knots, not {len(above)}"
            raise ValueError(msg)
        raw.extend(np.delete(above, j) for j in range(k + 1))
    out: list[NDArray[np.float64]] = []
    seen: set[tuple[float, ...]] = set()
    for row in raw:
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        if _valid_rows(row[None, :], lo, hi, min_spacing, fixed_knots)[0]:
            out.append(row)
    return out


def place_knots(
    knots: Sequence[float],
    lo: float,
    hi: float,
    min_spacing: float = 0.0,
    *,
    margin: float = 0.0,
) -> list[float]:
    """Sort knots into ``(lo, hi)`` and push them at least ``min_spacing`` apart.

    The knots are sorted and clipped to ``[lo + margin, hi - margin]``.
    Each is then moved up where needed to clear its predecessor, and if the
    last one then exceeds the upper limit, the knots are moved back down from
    it.
    The gap used is ``min_spacing + 1e-6``, so the returned knots are at least
    ``min_spacing`` apart despite floating-point rounding.

    Parameters
    ----------
    knots : sequence of float
        Candidate knot times, in any order.
    lo, hi : float
        Support of the knots; ``hi`` must exceed ``lo``.
    min_spacing : float
        Minimum distance between neighbouring knots (non-negative).
    margin : float
        Distance kept from each end of the support (non-negative).

    Returns
    -------
    list of float
        The sorted, spaced knots.

    Raises
    ------
    ValueError
        If ``margin`` or ``min_spacing`` is negative, ``hi <= lo``, or the
        knots cannot fit in the support at that spacing.

    """
    if margin < 0 or min_spacing < 0:
        msg = "margin and min_spacing must be non-negative"
        raise ValueError(msg)
    if hi <= lo:
        msg = f"hi ({hi}) must exceed lo ({lo})"
        raise ValueError(msg)
    gap = min_spacing + _PLACE_TOL
    x = np.sort(np.clip(np.asarray(knots, dtype=np.float64), lo + margin, hi - margin))
    for i in range(1, len(x)):
        x[i] = max(x[i], x[i - 1] + gap)
    if len(x) and x[-1] > hi - margin:
        x[-1] = hi - margin
        for i in range(len(x) - 2, -1, -1):
            x[i] = min(x[i], x[i + 1] - gap)
    if len(x) and x[0] < lo + margin - _PLACE_TOL:
        msg = f"{len(x)} knots {min_spacing} apart do not fit in ({lo}, {hi})"
        raise ValueError(msg)
    return [float(v) for v in x]


def evenly_spaced_knots(
    k: int,
    lo: float,
    hi: float,
    min_spacing: float = 0.0,
    *,
    margin: float = 0.0,
) -> list[float]:
    """Return ``k`` evenly spaced knots in ``(lo, hi)``, repaired by spacing.

    The knots start at ``lo + (hi - lo) * (i + 1) / (k + 1)`` and are passed to
    :func:`place_knots`.

    Parameters
    ----------
    k : int
        Number of knots (non-negative).
    lo, hi : float
        Support of the knots.
    min_spacing : float
        Minimum distance between neighbouring knots.
    margin : float
        Distance kept from each end of the support.

    Returns
    -------
    list of float
        The ``k`` sorted knots.

    Raises
    ------
    ValueError
        If ``k`` is negative or :func:`place_knots` rejects the arguments.

    """
    if k < 0:
        msg = "k must be non-negative"
        raise ValueError(msg)
    return place_knots(
        [lo + (hi - lo) * (i + 1) / (k + 1) for i in range(k)],
        lo,
        hi,
        min_spacing,
        margin=margin,
    )
