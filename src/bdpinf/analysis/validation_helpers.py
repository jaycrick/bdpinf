"""Shared helpers for validation notebooks and scripts."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from bdpinf.inference import (
    AgeData,
    BirthDeathModel,
    Constraints,
    FitResult,
    fit,
)

logger = logging.getLogger(__name__)

# Failures a replicate fit is allowed to end in: validation errors, non-finite
# arithmetic, singular matrices, and runtime errors from the optimiser or JAX.
# Anything else (TypeError, AttributeError, ...) is a programming error and propagates.
_EXPECTED_FIT_ERRORS = (
    ValueError,
    RuntimeError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True)
class FitModelConfig:
    """Configuration for single-model fitting helpers.

    Parameters
    ----------
    times : float | NDArray
        Epoch end times (the last is the observation time).
    is_ordered : bool
        Treat the first age as the spine.
    n_starts, perturb_std, random_seed
        Multi-start settings forwarded to :func:`~bdpinf.inference.fit`.
    initial_birth_rates, initial_death_rates : NDArray | None
        Starting values for the birth and death parameters (the epoch rates of
        the default piecewise-constant model).
        Give one or both; a missing family starts from the moment estimators.
    condition_on_survival : bool
        Condition the population term on non-extinction.
    method, tol, max_iter
        Optimiser settings forwarded to :func:`~bdpinf.inference.fit`.
    constraints : Constraints | None
        Fixed parameters and equality groups.
    model : BirthDeathModel | None
        Family to fit; defaults to piecewise-constant rates on ``times``.

    """

    times: float | NDArray[np.float64]
    is_ordered: bool = True
    n_starts: int = 1
    perturb_std: float = 0.3
    random_seed: int | None = None
    initial_birth_rates: NDArray[np.float64] | None = None
    initial_death_rates: NDArray[np.float64] | None = None
    condition_on_survival: bool = True
    method: str = "trust-exact"
    tol: float = 1e-6
    max_iter: int = 200
    constraints: Constraints | None = None
    model: BirthDeathModel | None = None


def fit_single_model(
    ages: NDArray[np.float64],
    config: FitModelConfig,
    *,
    compute_covariance: bool = False,
) -> FitResult | None:
    """Fit one model from a single age vector.

    Returns ``None`` for empty ``ages`` or when the fit raises; the exception is
    logged at warning level so failed replicates stay visible.
    """
    if len(ages) == 0:
        return None

    epoch_end_times = np.atleast_1d(np.asarray(config.times, dtype=np.float64))
    model = config.model or BirthDeathModel.piecewise_constant(epoch_end_times)

    init: dict[str, float] | None = None
    if config.initial_birth_rates is not None or config.initial_death_rates is not None:
        init = {}
        for names, values in (
            (model.birth_names, config.initial_birth_rates),
            (model.death_names, config.initial_death_rates),
        ):
            if values is not None:
                init.update(zip(names, np.atleast_1d(values).tolist(), strict=True))

    try:
        data = AgeData.from_groups(
            [np.asarray(ages, dtype=np.float64)],
            float(epoch_end_times[-1]),
            is_ordered=config.is_ordered,
        )
        return fit(
            data,
            model,
            constraints=config.constraints,
            init=init,
            method=config.method,
            n_starts=config.n_starts,
            perturb_std=config.perturb_std,
            seed=config.random_seed,
            tol=config.tol,
            max_iter=config.max_iter,
            condition_on_survival=config.condition_on_survival,
            compute_covariance=compute_covariance,
        )
    except _EXPECTED_FIT_ERRORS as exc:
        logger.warning("fit raised %s: %s", type(exc).__name__, exc)
        return None


def iqr_plotting_subset(
    values_by_name: Mapping[str, NDArray[np.float64] | list[float]],
    whisker_width: float = 1.5,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.bool_]]:
    """Return a joint plotting subset with IQR outliers removed row-wise."""
    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in values_by_name.items()
    }
    if not arrays:
        return {}, np.array([], dtype=bool)

    lengths = {array.shape[0] for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("All plotting arrays must have the same length")

    length = lengths.pop()
    if length == 0:
        return {name: array.copy() for name, array in arrays.items()}, np.zeros(
            0,
            dtype=bool,
        )

    mask = np.ones(length, dtype=bool)
    for array in arrays.values():
        finite_mask = np.isfinite(array)
        mask &= finite_mask

        finite_values = array[finite_mask]
        if finite_values.size == 0:
            mask &= False
            continue

        q1, q3 = np.percentile(finite_values, [25.0, 75.0])
        iqr = float(q3 - q1)
        if not np.isfinite(iqr) or iqr == 0.0:
            continue

        lower_bound = float(q1 - whisker_width * iqr)
        upper_bound = float(q3 + whisker_width * iqr)
        mask &= (array >= lower_bound) & (array <= upper_bound)

    return {name: array[mask] for name, array in arrays.items()}, mask


def boundary_corrected_kde_on_support(
    samples: NDArray[np.float64],
    support_upper: float,
    n_grid: int = 300,
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Reflection KDE on [0, support_upper], normalized on support.

    Parameters
    ----------
    samples : NDArray[np.float64]
        1-D array of non-negative sample values.
    support_upper : float
        Upper boundary of the support.
    n_grid : int
        Number of evaluation grid points. Default 300.
    weights : NDArray[np.float64] | None
        Optional per-sample weights (same length as *samples*). Non-finite
        samples are dropped together with their corresponding weights before the
        KDE is fit.  When *None* all samples are weighted equally (Scott's rule
        bandwidth; unchanged from the unweighted behaviour).

    """
    samples = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(samples)
    finite_samples = samples[finite_mask]
    if len(finite_samples) < 2 or np.std(finite_samples) == 0.0:
        return None, None

    finite_weights: NDArray[np.float64] | None = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        finite_weights = w[finite_mask]

    kde = gaussian_kde(finite_samples, weights=finite_weights)
    x_grid = np.linspace(0.0, float(support_upper), int(n_grid))
    y_kde = kde(x_grid) + kde(-x_grid) + kde(2.0 * support_upper - x_grid)

    normalization = np.trapezoid(y_kde, x_grid)
    if normalization > 0.0 and np.isfinite(normalization):
        y_kde = y_kde / normalization
    return x_grid, y_kde


def boundary_corrected_kde_at_zero(
    samples: NDArray[np.float64],
    n_grid: int = 300,
    upper_padding: float = 1.15,
    weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Reflection KDE for nonnegative support [0, inf), normalized on grid.

    Parameters
    ----------
    samples : NDArray[np.float64]
        1-D array of non-negative sample values.
    n_grid : int
        Number of evaluation grid points. Default 300.
    upper_padding : float
        Multiplier applied to the sample maximum to set the grid upper bound.
        Default 1.15.
    weights : NDArray[np.float64] | None
        Optional per-sample weights (same length as *samples*). Non-finite
        samples are dropped together with their corresponding weights before the
        KDE is fit.  When *None* all samples are weighted equally.

    """
    samples = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(samples)
    finite_samples = samples[finite_mask]
    if len(finite_samples) < 2 or np.std(finite_samples) == 0.0:
        return None, None

    finite_weights: NDArray[np.float64] | None = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        finite_weights = w[finite_mask]

    x_max = float(np.max(finite_samples))
    if not np.isfinite(x_max) or x_max <= 0.0:
        x_max = 1.0
    x_max *= float(upper_padding)

    kde = gaussian_kde(finite_samples, weights=finite_weights)
    x_grid = np.linspace(0.0, x_max, int(n_grid))
    y_kde = kde(x_grid) + kde(-x_grid)

    normalization = np.trapezoid(y_kde, x_grid)
    if normalization > 0.0 and np.isfinite(normalization):
        y_kde = y_kde / normalization
    return x_grid, y_kde


def safe_mean(values: Any) -> float:
    """Return mean of finite values in array or NaN if none."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def safe_var(values: Any) -> float:
    """Return sample variance (ddof=1) for finite values or NaN if insufficient."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.var(arr, ddof=1))


def theoretical_age_density(
    log_lik_fn: Callable[[float], Any],
    support_upper: float,
    n_grid: int = 300,
    include_endpoint: bool = True,
    proportion_not_at_upper: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a log-likelihood function on a grid to obtain a density curve.

    Parameters
    ----------
    log_lik_fn : Callable[[float], Any]
        Function mapping a scalar age to its log-likelihood (float or JAX/numpy scalar).
    support_upper : float
        Upper end of the evaluation grid (the epoch time span).
    n_grid : int
        Number of grid points. Default 300.
    include_endpoint : bool
        Whether to include ``support_upper`` in the grid. Default ``True``.
    proportion_not_at_upper : float | None
        If provided and positive, the density is rescaled by
        ``1 / proportion_not_at_upper`` to account for censoring at the upper
        bound.

    Returns
    -------
    x_grid : NDArray[np.float64]
        Grid of age values in ``[0, support_upper]``.
    y_vals : NDArray[np.float64]
        Density values at each grid point.

    """
    import math

    x_grid = np.linspace(0.0, support_upper, n_grid, endpoint=include_endpoint)
    y_vals = []
    for x_val in x_grid:
        log_density = float(log_lik_fn(float(x_val)))
        y_vals.append(math.exp(log_density) if math.isfinite(log_density) else 0.0)
    y_vals_arr = np.asarray(y_vals, dtype=np.float64)
    if proportion_not_at_upper is not None:
        p = float(proportion_not_at_upper)
        if np.isfinite(p) and p > 0.0:
            y_vals_arr = y_vals_arr * (1.0 / p)
    return x_grid, y_vals_arr


def proportion_not_at_upper(
    samples: NDArray[np.float64],
    support_upper: float,
    tol: float = 1e-12,
) -> float:
    """Return the fraction of samples strictly below the upper support bound.

    Parameters
    ----------
    samples : NDArray[np.float64]
        Array of observed age values (may contain non-finite entries).
    support_upper : float
        The censoring threshold (epoch time span).
    tol : float
        Tolerance for the strict-less-than comparison. Default ``1e-12``.

    Returns
    -------
    float
        Proportion of finite samples that are less than
        ``support_upper - tol``. Returns ``1.0`` if there are no finite
        samples.

    """
    samples = np.asarray(samples, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return 1.0
    return float(np.mean(finite < (float(support_upper) - float(tol))))


def robust_y_max(
    arrays: list[NDArray[np.float64] | None],
    quantile: float = 99.0,
    pad: float = 1.1,
) -> float:
    """Return a robust upper y-axis limit from a list of value arrays.

    Parameters
    ----------
    arrays : list[NDArray[np.float64] | None]
        Arrays of y-values. ``None`` entries and non-finite values are skipped.
    quantile : float
        Upper percentile used to clip outliers. Default 99.0.
    pad : float
        Multiplicative padding applied to the percentile value. Default 1.1.

    Returns
    -------
    float
        A positive upper limit suitable for ``yaxis_range``. Falls back to
        ``1.0`` when all inputs are empty or non-finite.

    """
    finite_parts: list[NDArray[np.float64]] = []
    for arr in arrays:
        if arr is None:
            continue
        values = np.asarray(arr, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            finite_parts.append(finite)
    if not finite_parts:
        return 1.0
    joined = np.concatenate(finite_parts)
    q = float(np.nanpercentile(joined, quantile))
    if not np.isfinite(q) or q <= 0.0:
        q = float(np.nanmax(joined)) if joined.size > 0 else 1.0
    if not np.isfinite(q) or q <= 0.0:
        q = 1.0
    return q * float(pad)


def build_hist_bins(
    primary_values: Any,
    secondary_values: Any,
    bins: int = 30,
) -> dict[str, float]:
    """Compute shared histogram bin parameters from two value arrays.

    Parameters
    ----------
    primary_values : array-like
        First set of values (e.g., observed ages).
    secondary_values : array-like
        Second set of values (e.g., theoretical ages) to include in the range.
    bins : int
        Desired number of bins. Default 30.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"start"``, ``"end"``, and ``"size"``; ``"size"``
        is the bin step suitable for ``alt.Bin(step=...)`` so two series can
        share identical bin edges.

    """
    combined = np.concatenate(
        [np.asarray(primary_values), np.asarray(secondary_values)]
    )
    if combined.size == 0:
        return {"start": 0.0, "end": 1.0, "size": 1.0 / bins}
    max_x = float(np.max(combined))
    upper = max_x * 1.05 if max_x > 0 else 1.0
    return {"start": 0.0, "end": upper, "size": upper / bins}


def has_ages_in_each_epoch(
    ages: NDArray[np.float64],
    times: NDArray[np.float64],
) -> bool:
    """Return True when ages span at least two consecutive epochs.

    Parameters
    ----------
    ages : NDArray[np.float64]
        Array of individual ages.
    times : NDArray[np.float64]
        Epoch boundary times (e.g., ``[epoch_1_end, final_time]``).

    Returns
    -------
    bool
        ``True`` if at least one age falls in the first epoch (< ``times[0]``)
        **and** at least one age falls in a later epoch (>= ``times[0]`` and
        <= ``times[-1]``).

    """
    ages = np.asarray(ages, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if ages.size == 0 or times.size == 0:
        return False
    first_epoch_end = float(times[0])
    final_time = float(times[-1])
    return bool(np.any(ages < first_epoch_end)) and bool(
        np.any((ages >= first_epoch_end) & (ages <= final_time))
    )
