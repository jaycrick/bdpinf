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
"""

import concurrent.futures
import dataclasses
import logging
import math
import multiprocessing
import os
import time
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.optimize
import scipy.stats.qmc
from numpy.typing import NDArray

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import FitResult, fit
from bdpinf.inference.likelihood import AgeData
from bdpinf.inference.rate_models import BirthDeathModel

__all__ = ["KnotFitResult", "MethodResult", "fit_knots"]

logger = logging.getLogger(__name__)

_THOROUGH_METHODS = ("brent", "de-profile", "grid-newton")
_MAX_BRENT_SWEEPS = 5
_N_NEWTON_STARTS = 3
_LOG_EVERY = 10


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


def _candidates(
    ranges: Sequence[tuple[float, float]], n_candidates: int, seed: int | None
) -> NDArray[np.float64]:
    """Candidate knot placements, one row each (sorted within the row)."""
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    if len(ranges) == 1:
        return np.linspace(lo[0], hi[0], n_candidates + 2)[1:-1, None]
    sobol = scipy.stats.qmc.Sobol(
        d=len(ranges), scramble=True, rng=np.random.default_rng(seed)
    )
    unit = sobol.random_base2(max(1, math.ceil(math.log2(n_candidates))))
    return np.sort(lo + (hi - lo) * unit, axis=1)


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


def _run_methods(
    names: Sequence[str], job: _Search
) -> dict[str, tuple[MethodResult, NDArray[np.float64]]]:
    """Run the named searches; return each one's result and rates-and-knots vector."""
    knots = job.model.knots
    if knots is None:  # unreachable: checked by fit_knots
        raise RuntimeError
    data, model, user = job.data, job.model, job.user
    free_idx = job.free_idx
    free_names = [knots.names[i] for i in free_idx]
    ranges = [knots.ranges[i] for i in free_idx]
    initial = np.array([knots.initial_times[i] for i in free_idx])
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    tol = 1e-3 * (hi - lo)
    n_rates = model.n_birth + model.n_death
    fixed_knots = {n: user.fixed[n] for n in knots.names if n in user.fixed}

    tried: dict[tuple[float, ...], float] = {}
    best: dict[str, Any] = {"ll": -np.inf, "natural": None}
    track: dict[str, Any] = {"ll": -np.inf, "taus": None, "natural": None, "fits": 0}

    def all_knots(free: NDArray[np.float64]) -> NDArray[np.float64]:
        values = dict(zip(free_names, np.sort(free), strict=True)) | fixed_knots
        return np.array([values[n] for n in knots.names])

    def fit_at(free: NDArray[np.float64], **extra: Any) -> FitResult:
        fixed = {**user.fixed, **dict(zip(free_names, np.sort(free), strict=True))}
        start = job.init
        if best["natural"] is not None:
            start = np.concatenate([best["natural"][:n_rates], all_knots(free)])
        return fit(
            data,
            model,
            constraints=Constraints(fixed=fixed, equal=user.equal),
            init=start,
            **{**job.fit_kwargs, **extra},
        )

    def record(value: float, taus: NDArray[np.float64], natural: Any) -> None:
        if value > track["ll"]:
            track.update(ll=value, taus=taus, natural=np.asarray(natural))

    def profile(free: NDArray[np.float64]) -> float:
        free = np.sort(np.asarray(free, dtype=np.float64))
        key = _key(free)
        if key in tried:
            return tried[key]
        track["fits"] += 1
        try:
            result = fit_at(free, compute_covariance=False)
        except ValueError as exc:
            logger.debug("no fit at knots %s: %s", key, exc)
            tried[key] = value = -np.inf
            return value
        tried[key] = value = result.log_likelihood
        if value > best["ll"]:
            best["ll"], best["natural"] = value, result.natural
        record(value, free, result.natural)
        if len(tried) % _LOG_EVERY == 0:
            logger.info(
                "fit_knots: %d rate fits, best log-likelihood %.4f",
                len(tried),
                best["ll"],
            )
        return value

    def brent() -> None:
        if len(ranges) == 1:
            scipy.optimize.minimize_scalar(
                lambda t: -profile(np.array([t])),
                bounds=(lo[0], hi[0]),
                method="bounded",
                options={"xatol": float(tol[0])},
            )
            return
        current = initial.copy()
        profile(current)
        for _ in range(_MAX_BRENT_SWEEPS):
            moved = 0.0
            for j in range(len(ranges)):
                low = max(lo[j], current[j - 1]) if j > 0 else lo[j]
                high = min(hi[j], current[j + 1]) if j + 1 < len(current) else hi[j]
                if not high > low:
                    continue

                def along(t: float, j: int = j) -> float:
                    trial = current.copy()
                    trial[j] = t
                    return -profile(trial)

                step = scipy.optimize.minimize_scalar(
                    along,
                    bounds=(low, high),
                    method="bounded",
                    options={"xatol": float(tol[j])},
                )
                if -step.fun >= profile(current):
                    moved = max(moved, abs(float(step.x) - current[j]))
                    current[j] = step.x
            if moved <= float(np.max(tol)):
                break

    def de_profile() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Sobol wants a power-of-two population
            scipy.optimize.differential_evolution(
                lambda t: -profile(t),
                ranges,
                popsize=8,
                maxiter=15,
                tol=1e-4,
                seed=job.seed,
                polish=False,
                init="sobol",
            )

    def grid_newton() -> None:
        candidates = _candidates(ranges, job.n_candidates, job.seed)
        for row in candidates:
            profile(row)
        order = np.argsort([-tried[_key(row)] for row in candidates])
        for row in candidates[order[:_N_NEWTON_STARTS]]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    start = fit_at(row, compute_covariance=False)
                    joint = fit(
                        data,
                        model,
                        constraints=user,
                        init=start.natural,
                        compute_covariance=False,
                        **job.fit_kwargs,
                    )
                except ValueError:
                    continue
            if np.isfinite(joint.log_likelihood):
                taus = np.asarray(joint.natural)[n_rates:][free_idx]
                record(joint.log_likelihood, taus, joint.natural)

    searches = {"brent": brent, "de-profile": de_profile, "grid-newton": grid_newton}
    found: dict[str, tuple[MethodResult, NDArray[np.float64]]] = {}
    for name in names:
        track.update(ll=-np.inf, taus=None, natural=None, fits=0)
        began = time.perf_counter()
        searches[name]()
        if track["taus"] is None or not np.isfinite(track["ll"]):
            logger.debug("method %s found no finite placement", name)
            continue
        found[name] = (
            MethodResult(
                knot_times=all_knots(track["taus"]),
                log_likelihood=float(track["ll"]),
                seconds=time.perf_counter() - began,
                n_fits=int(track["fits"]),
            ),
            track["natural"],
        )
    return found


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
        :class:`~bdpinf.inference.rate_models.FreeKnots`); the number of knots
        and their allowed ranges come from there.
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
        knot in an equality group, or no placement gives a finite likelihood.

    """
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
                joint = None
        if (
            joint is not None
            and joint.converged
            and joint.log_likelihood >= chosen.log_likelihood - 1e-9
        ):
            chosen = joint

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
