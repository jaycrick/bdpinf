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

One search runs by default, chosen from benchmarks on simulated and real data:

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

``FreeKnots(min_spacing=...)`` keeps knots apart: no placement closer than that
between any two sorted knots (fixed knots included) is evaluated as a candidate
or returned.
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

import numpy as np
import scipy.optimize
import scipy.stats.qmc
from numpy.typing import ArrayLike, NDArray

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import FitResult, fit
from bdpinf.inference.likelihood import AgeData
from bdpinf.inference.rate_models import BirthDeathModel, FreeKnots

__all__ = ["KnotFitResult", "MethodResult", "fit_knots"]

logger = logging.getLogger(__name__)

_THOROUGH_METHODS = ("brent", "de-profile", "grid-newton")
_MAX_BRENT_SWEEPS = 5
_N_NEWTON_STARTS = 3
_LOG_EVERY = 10
_SLACK = 1e-9
_MIN_VALID_CANDIDATES = 4
_MAX_REDRAWS = 8
_DE_MIN_POPULATION = 5


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

    """

    knot_times: NDArray[np.float64]
    log_likelihood: float
    seconds: float
    n_fits: int


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

    """

    fit: FitResult
    knot_times: NDArray[np.float64]
    log_likelihood: float
    method: str
    method_results: dict[str, MethodResult]


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

    def fit_at(self, free: NDArray[np.float64], **extra: Any) -> FitResult:
        """Fit the rates with the free knots held at ``free`` (warm-started)."""
        job = self.job
        fixed = {
            **job.user.fixed,
            **dict(zip(self.free_names, np.sort(free), strict=True)),
        }
        start = job.init
        if self.best.natural is not None:
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
            result = self.fit_at(free, compute_covariance=False)
        except ValueError as exc:
            logger.debug("no fit at knots %s: %s", key, exc)
            self.tried[key] = value = -np.inf
            return value
        self.tried[key] = value = result.log_likelihood
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
                maxiter=15,
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
        for row in candidates[order[:_N_NEWTON_STARTS]]:
            joint = self._joint_fit_from(row)
            if joint is not None:
                self._record_joint(joint)

    def _joint_fit_from(self, row: NDArray[np.float64]) -> FitResult | None:
        """Joint fit of rates and knots started at knots ``row`` (None if it fails)."""
        job = self.job
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                start = self.fit_at(row, compute_covariance=False)
                return fit(
                    job.data,
                    job.model,
                    constraints=job.user,
                    init=start.natural,
                    compute_covariance=False,
                    **job.fit_kwargs,
                )
            except ValueError:
                return None

    def _record_joint(self, joint: FitResult) -> None:
        """Record a joint fit if it is finite and keeps the knots apart."""
        knot_part = np.asarray(joint.natural)[self.n_rates :]
        if np.isfinite(joint.log_likelihood) and _spacing_ok(knot_part, self.spacing):
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


def fit_knots(
    data: AgeData,
    model: BirthDeathModel,
    *,
    constraints: Constraints | None = None,
    thorough: bool = False,
    workers: int | None = None,
    n_candidates: int = 64,
    polish: bool = True,
    seed: int | None = None,
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
    n_candidates : int
        Placements scanned by ``"grid-newton"``: an even grid for one knot, a
        scrambled Sobol sequence (rounded up to a power of two) for several.
    polish : bool
        Finish with a joint fit of rates and knots from the best placement,
        kept if it converges without lowering the likelihood.
        This gives knot standard errors for smooth rates.
    seed : int, optional
        Seed for the Sobol sequence and differential evolution.
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
        no placement fits at that spacing, or none gives a finite likelihood.

    """
    knots, user, free_idx = _check_inputs(model, constraints)
    fit_kwargs.pop("compute_covariance", None)
    init = fit_kwargs.pop("init", None)
    job = _Search(
        data, model, user, free_idx, n_candidates, seed, init, dict(fit_kwargs)
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

    knot_times = chosen.knot_times
    if knot_times is None:  # unreachable: the model has free knots
        raise RuntimeError
    return KnotFitResult(
        fit=chosen,
        knot_times=knot_times,
        log_likelihood=chosen.log_likelihood,
        method=winner,
        method_results={n: r for n, (r, _) in found.items()},
    )
