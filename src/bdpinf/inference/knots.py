r"""Global search for free knot times by maximum likelihood.

With piecewise-constant rates the likelihood is discontinuous in a knot time
(it jumps where the knot crosses an observed $T - a$), so Newton steps over the
knots alone are unreliable.
:func:`fit_knots` instead *profiles* the knots: it fits the rates by Newton at
each candidate knot placement, scans candidates over the allowed ranges, and
refines the best ones with a derivative-free search.
The knots enter :func:`~bdpinf.inference.fitting.fit` as fixed parameters, so
the jitted objective is compiled once for the whole search.
"""

import dataclasses
import logging
import math
import warnings
from typing import Any

import numpy as np
import scipy.optimize
import scipy.stats.qmc
from numpy.typing import NDArray

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import FitResult, fit
from bdpinf.inference.likelihood import AgeData
from bdpinf.inference.rate_models import BirthDeathModel

__all__ = ["KnotFitResult", "fit_knots"]

logger = logging.getLogger(__name__)

_MAX_REFINE_EVALS_PER_KNOT = 60


@dataclasses.dataclass(frozen=True)
class KnotFitResult:
    """Result of :func:`fit_knots`.

    Attributes
    ----------
    fit : FitResult
        The fit at the selected knot times.
        Its covariance is conditional on the knots unless the joint polish
        was accepted (``fit.constraints.fixed`` then has no ``tau`` entries).
        Usable as ``init`` for :func:`~bdpinf.inference.fitting.fit` and
        :func:`~bdpinf.inference.bayesian.fit_bayesian`.
    knot_times : NDArray
        Selected knot times, increasing.
    log_likelihood : float
        Maximised log-likelihood.
    profile_points : NDArray
        Every knot placement tried, shape ``[n_tried, n_knots]``.
    profile_log_likelihood : NDArray
        Log-likelihood with the rates re-fitted at each placement (``-inf`` where
        no fit was possible).

    """

    fit: FitResult
    knot_times: NDArray[np.float64]
    log_likelihood: float
    profile_points: NDArray[np.float64]
    profile_log_likelihood: NDArray[np.float64]


def _candidates(
    ranges: tuple[tuple[float, float], ...], n_candidates: int, seed: int | None
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


def fit_knots(
    data: AgeData,
    model: BirthDeathModel,
    *,
    constraints: Constraints | None = None,
    n_candidates: int = 64,
    refine: bool = True,
    n_refine: int = 2,
    polish: bool = True,
    seed: int | None = None,
    **fit_kwargs: Any,
) -> KnotFitResult:
    r"""Fit rates and knot times by profile likelihood over the knots.

    Parameters
    ----------
    data : AgeData
        Observed ages.
    model : BirthDeathModel
        Must have free knots (built from a
        :class:`~bdpinf.inference.rate_models.FreeKnots`); the number of knots
        and their allowed ranges come from there.
    constraints : Constraints, optional
        Constraints on the rate parameters (knot times cannot be constrained).
    n_candidates : int
        Placements scanned before refining: an even grid for one knot, a
        scrambled Sobol sequence (rounded up to a power of two) for several.
    refine : bool
        Refine the best placements with Nelder-Mead on the unconstrained
        scale (each evaluation re-fits the rates).
    n_refine : int
        Number of best scanned placements to refine from.
    polish : bool
        Finish with a joint fit of rates and knots from the best placement,
        kept if it converges without lowering the likelihood.
        This gives knot standard errors for smooth (piecewise-linear) rates.
    seed : int, optional
        Seed for the Sobol sequence.
    **fit_kwargs
        Forwarded to :func:`~bdpinf.inference.fitting.fit` (for example
        ``n_starts``, ``n_nodes``, or ``init`` for the first placement).

    Returns
    -------
    KnotFitResult

    Raises
    ------
    ValueError
        If the model has no free knots, constraints touch the knots, or no
        placement gives a finite likelihood.

    """
    knots = model.knots
    if knots is None:
        msg = "fit_knots needs a model with free knots (see FreeKnots)"
        raise ValueError(msg)
    user = constraints or Constraints()
    tau_names = set(knots.names)
    if tau_names & set(user.fixed) or any(tau_names & set(g) for g in user.equal):
        msg = "Knot times cannot appear in constraints for fit_knots"
        raise ValueError(msg)
    fit_kwargs.pop("compute_covariance", None)
    init = fit_kwargs.pop("init", None)
    lo = np.array([r[0] for r in knots.ranges])
    hi = np.array([r[1] for r in knots.ranges])
    n_rates = model.n_birth + model.n_death

    tried: dict[tuple[float, ...], float] = {}
    best: dict[str, Any] = {"ll": -np.inf, "natural": None}

    def fit_at(taus: NDArray[np.float64], **extra: Any) -> FitResult:
        fixed = {**user.fixed, **dict(zip(knots.names, np.sort(taus), strict=True))}
        start = init
        if best["natural"] is not None:
            start = np.concatenate([best["natural"][:n_rates], np.sort(taus)])
        return fit(
            data,
            model,
            constraints=Constraints(fixed=fixed, equal=user.equal),
            init=start,
            **{**fit_kwargs, **extra},
        )

    def profile(taus: NDArray[np.float64]) -> float:
        key = _key(taus)
        if key in tried:
            return tried[key]
        try:
            result = fit_at(taus, compute_covariance=False)
        except ValueError as exc:
            logger.debug("no fit at knots %s: %s", key, exc)
            tried[key] = -np.inf
            return -np.inf
        tried[key] = result.log_likelihood
        if result.log_likelihood > best["ll"]:
            best["ll"], best["natural"] = result.log_likelihood, result.natural
        return result.log_likelihood

    candidates = _candidates(knots.ranges, n_candidates, seed)
    for row in candidates:
        profile(row)
    if best["natural"] is None:
        msg = "No knot placement gives a finite likelihood"
        raise ValueError(msg)

    if refine:
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
                    "maxfev": _MAX_REFINE_EVALS_PER_KNOT * len(knots.ranges),
                },
            )

    best_taus = np.sort(best["natural"][n_rates:])
    chosen = fit_at(best_taus)
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
    )
