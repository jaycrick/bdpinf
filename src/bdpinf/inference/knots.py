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

Several search methods are available (see :data:`KNOT_METHODS`).
By default only ``"grid-newton"`` runs, which gave the best accuracy for its
cost when several knots are free.
Give several names (or ``"thorough"``, which is :data:`THOROUGH_KNOT_METHODS`)
to run them all and keep the best estimate, because each method misses the
global maximum in a different set of data sets.
With ``workers`` the methods run in parallel processes.
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
from bdpinf.inference.fitting import (
    _PENALTY,  # pyright: ignore[reportPrivateUsage]
    FitResult,
    _problem,  # pyright: ignore[reportPrivateUsage]
    fit,
)
from bdpinf.inference.likelihood import AgeData
from bdpinf.inference.rate_models import BirthDeathModel
from bdpinf.models.math_general import DEFAULT_NODES

__all__ = ["KNOT_METHODS", "KnotFitResult", "MethodResult", "fit_knots"]

logger = logging.getLogger(__name__)

KNOT_METHODS = ("brent", "nelder-mead", "de-profile", "grid-newton", "profile")
"""Names accepted by the ``methods`` argument of :func:`fit_knots`."""

THOROUGH_KNOT_METHODS = ("brent", "de-profile", "grid-newton")
"""The methods ``methods="thorough"`` runs: the best triple in the NTVMR benchmark."""

_MAX_REFINE_EVALS_PER_KNOT = 60
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
        Placements already tried by an earlier method in the same call are
        free, so later methods can look cheaper than they are.

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
    profile_points : NDArray
        Every free-knot placement tried by any method,
        shape ``[n_tried, n_free_knots]``.
    profile_log_likelihood : NDArray
        Log-likelihood with the rates re-fitted at each placement (``-inf`` where
        no fit was possible).
    method : str
        The method whose placement was selected.
    method_results : dict[str, MethodResult]
        What every method found, in the order they ran.

    """

    fit: FitResult
    knot_times: NDArray[np.float64]
    log_likelihood: float
    profile_points: NDArray[np.float64]
    profile_log_likelihood: NDArray[np.float64]
    method: str = "brent"
    method_results: dict[str, MethodResult] = dataclasses.field(default_factory=dict)


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


def _as_methods(methods: str | Sequence[str]) -> tuple[str, ...]:
    """Validate ``methods`` and drop repeats, keeping the order."""
    if methods == "thorough":
        return THOROUGH_KNOT_METHODS
    names = (methods,) if isinstance(methods, str) else tuple(methods)
    unknown = [m for m in names if m not in KNOT_METHODS]
    if unknown or not names:
        msg = f"Unknown knot methods {unknown}; choose from {list(KNOT_METHODS)}."
        raise ValueError(msg)
    return tuple(dict.fromkeys(names))


def _run_one_method(job: tuple[str, dict[str, Any]]) -> tuple[str, MethodResult | None]:
    """Run one method in a worker process and return its result (if any)."""
    name, kwargs = job
    try:
        got = fit_knots(methods=(name,), **kwargs)
    except ValueError:
        return name, None
    return name, got.method_results.get(name)


def _parallel_methods(
    names: tuple[str, ...], workers: int, kwargs: dict[str, Any]
) -> dict[str, MethodResult]:
    """Run each method in its own single-threaded process, in the given order."""
    old = os.environ.get("XLA_FLAGS")
    flags = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["XLA_FLAGS"] = f"{old} {flags}" if old else flags
    try:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(names)), mp_context=context
        ) as pool:
            done = dict(pool.map(_run_one_method, [(n, kwargs) for n in names]))
    finally:
        if old is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = old
    return {n: r for n in names if (r := done[n]) is not None}


def fit_knots(
    data: AgeData,
    model: BirthDeathModel,
    *,
    constraints: Constraints | None = None,
    methods: str | Sequence[str] = "grid-newton",
    n_candidates: int = 12,
    knot_tol: float | Sequence[float] | None = None,
    refine: bool = True,
    n_refine: int = 2,
    polish: bool = True,
    seed: int | None = None,
    workers: int | None = None,
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
    methods : str or sequence of str
        The search method, or several to run and compare, keeping the best
        placement; ``"thorough"`` runs :data:`THOROUGH_KNOT_METHODS`.
        Any of :data:`KNOT_METHODS`:

        ``"brent"``
            Bounded Brent search of the profile (cyclic over the knots when
            there are several), with no starting grid.
        ``"de-profile"``
            Differential evolution over the profile.
        ``"grid-newton"``
            A coarse scan of ``n_candidates`` placements, then joint Newton on
            rates and knots from the best three.
        ``"nelder-mead"``
            Derivative-free Nelder-Mead on rates and knots together.
        ``"profile"``
            The scan, then Nelder-Mead on the profile from the best
            ``n_refine`` placements (with ``refine=False``, the scan alone).
    n_candidates : int
        Placements scanned by ``"grid-newton"`` and ``"profile"``: an even grid
        for one knot, a scrambled Sobol sequence (rounded up to a power of two)
        for several.
    knot_tol : float or sequence of float, optional
        Absolute knot precision for ``"brent"``, one value or one per free knot.
        Default: 1e-3 of each knot's range.
    refine : bool
        With ``"profile"``, refine the best scanned placements.
    n_refine : int
        With ``"profile"``, how many of the best placements to refine from.
    polish : bool
        Finish with a joint fit of rates and knots from the best placement,
        kept if it converges without lowering the likelihood.
        This gives knot standard errors for smooth rates.
    seed : int, optional
        Seed for the Sobol sequence and differential evolution.
    workers : int, optional
        With several methods, run them in up to this many parallel processes
        (single-threaded each, so give about one core per method).
        The profile points of the returned result then come from the final
        fits only, and the processes recompile the objective unless
        ``JAX_COMPILATION_CACHE_DIR`` is set.
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
        knot in an equality group, a method name is unknown, or no placement
        gives a finite likelihood.

    """
    knots = model.knots
    if knots is None:
        msg = "fit_knots needs a model with free knots (see FreeKnots)"
        raise ValueError(msg)
    names = _as_methods(methods)
    user = constraints or Constraints()
    tau_names = set(knots.names)
    if tau_names & {n for g in user.equal for n in g}:
        msg = "Knot times cannot appear in constraints for fit_knots"
        raise ValueError(msg)
    free_idx = [i for i, n in enumerate(knots.names) if n not in user.fixed]
    if not free_idx:
        msg = "Knot times cannot all be fixed for fit_knots"
        raise ValueError(msg)
    free_names = [knots.names[i] for i in free_idx]
    ranges = [knots.ranges[i] for i in free_idx]
    initial = np.array([knots.initial_times[i] for i in free_idx])
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    if knot_tol is None:
        tol = 1e-3 * (hi - lo)
    else:
        tol = np.broadcast_to(np.asarray(knot_tol, dtype=np.float64), lo.shape)
    fit_kwargs.pop("compute_covariance", None)
    init = fit_kwargs.pop("init", None)
    n_rates = model.n_birth + model.n_death
    fixed_knots = {n: user.fixed[n] for n in knots.names if n in user.fixed}

    tried: dict[tuple[float, ...], float] = {}
    best: dict[str, Any] = {"ll": -np.inf, "natural": None}
    track: dict[str, Any] = {"ll": -np.inf, "taus": None, "fits": 0}

    def all_knots(free: NDArray[np.float64]) -> NDArray[np.float64]:
        values = dict(zip(free_names, np.sort(free), strict=True)) | fixed_knots
        return np.array([values[n] for n in knots.names])

    def free_of(natural: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(natural)[n_rates:][free_idx]

    def fit_at(free: NDArray[np.float64], **extra: Any) -> FitResult:
        fixed = {**user.fixed, **dict(zip(free_names, np.sort(free), strict=True))}
        start = init
        if best["natural"] is not None:
            start = np.concatenate([best["natural"][:n_rates], all_knots(free)])
        return fit(
            data,
            model,
            constraints=Constraints(fixed=fixed, equal=user.equal),
            init=start,
            **{**fit_kwargs, **extra},
        )

    def profile(free: NDArray[np.float64]) -> float:
        free = np.sort(np.asarray(free, dtype=np.float64))
        key = _key(free)
        if key in tried:
            value = tried[key]
        else:
            track["fits"] += 1
            try:
                result = fit_at(free, compute_covariance=False)
            except ValueError as exc:
                logger.debug("no fit at knots %s: %s", key, exc)
                tried[key] = value = -np.inf
            else:
                tried[key] = value = result.log_likelihood
                if value > best["ll"]:
                    best["ll"], best["natural"] = value, result.natural
            if len(tried) % _LOG_EVERY == 0:
                logger.info(
                    "fit_knots: %d rate fits, best log-likelihood %.4f",
                    len(tried),
                    best["ll"],
                )
        if value > track["ll"]:
            track["ll"], track["taus"] = value, free
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
                seed=seed,
                polish=False,
                init="sobol",
            )

    def scan() -> NDArray[np.float64]:
        candidates = _candidates(ranges, n_candidates, seed)
        for row in candidates:
            profile(row)
        return candidates

    def grid_newton() -> None:
        candidates = scan()
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
                        **fit_kwargs,
                    )
                except ValueError:
                    continue
            if joint.log_likelihood > track["ll"] and np.isfinite(joint.log_likelihood):
                track["ll"], track["taus"] = (
                    joint.log_likelihood,
                    free_of(joint.natural),
                )

    def nelder_mead() -> None:
        problem = _problem(
            data,
            model,
            user,
            condition_on_survival=fit_kwargs.get("condition_on_survival", True),
            n_nodes=fit_kwargs.get("n_nodes", DEFAULT_NODES),
            pad_to_bucket=fit_kwargs.get("pad_to_bucket", True),
            hessian_mode=fit_kwargs.get("hessian_mode", "fwd-fwd"),
        )
        profile(initial)
        start_fit = fit_at(initial, compute_covariance=False)
        x = np.asarray(problem.compiled.reduce(start_fit.natural), dtype=np.float64)

        def value(u: NDArray[np.float64]) -> float:
            return float(problem.nll(u))

        res = None
        for _ in range(2):  # restart from the optimum to escape a collapsed simplex
            res = scipy.optimize.minimize(
                value,
                x,
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-8, "maxfev": 3000},
            )
            x = np.asarray(res.x)
        if res is not None and res.fun < _PENALTY and -res.fun > track["ll"]:
            natural = np.asarray(problem.compiled.expand(x))
            track["ll"], track["taus"] = -float(res.fun), free_of(natural)

    def profile_search() -> None:
        candidates = scan()
        if not refine:
            return
        order = np.argsort([-tried[_key(row)] for row in candidates])

        def to_knots(u: NDArray[np.float64]) -> NDArray[np.float64]:
            return lo + (hi - lo) / (1.0 + np.exp(-u))

        for row in candidates[order[:n_refine]]:
            frac = np.clip((row - lo) / (hi - lo), 1e-6, 1 - 1e-6)
            scipy.optimize.minimize(
                lambda u: -profile(to_knots(u)),
                np.log(frac / (1 - frac)),
                method="Nelder-Mead",
                options={
                    "xatol": 1e-3,
                    "fatol": 1e-9,
                    "maxfev": _MAX_REFINE_EVALS_PER_KNOT * len(ranges),
                },
            )

    searches = {
        "brent": brent,
        "de-profile": de_profile,
        "grid-newton": grid_newton,
        "nelder-mead": nelder_mead,
        "profile": profile_search,
    }
    found: dict[str, MethodResult] = {}
    if workers is not None and workers > 1 and len(names) > 1:
        found = _parallel_methods(
            names,
            workers,
            {
                "data": data,
                "model": model,
                "constraints": constraints,
                "n_candidates": n_candidates,
                "knot_tol": knot_tol,
                "refine": refine,
                "n_refine": n_refine,
                "polish": False,
                "seed": seed,
                "init": init,
                **fit_kwargs,
            },
        )
    for name in () if found else names:
        track.update(ll=-np.inf, taus=None, fits=0)
        began = time.perf_counter()
        searches[name]()
        if track["taus"] is None or not np.isfinite(track["ll"]):
            logger.debug("method %s found no finite placement", name)
            continue
        found[name] = MethodResult(
            knot_times=all_knots(track["taus"]),
            log_likelihood=float(track["ll"]),
            seconds=time.perf_counter() - began,
            n_fits=int(track["fits"]),
        )
    if not found:
        msg = "No knot placement gives a finite likelihood"
        raise ValueError(msg)

    winner = max(found, key=lambda m: found[m].log_likelihood)
    best_free = np.array([found[winner].knot_times[i] for i in free_idx])
    chosen = fit_at(best_free)
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
        profile_points=np.array(list(tried)),
        profile_log_likelihood=np.array(list(tried.values())),
        method=winner,
        method_results=found,
    )
