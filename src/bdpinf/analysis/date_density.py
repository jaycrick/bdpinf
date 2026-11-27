"""Date density estimation for items with interval-censored dates.

Items are dated as intervals ``[early, late]``.  The functions here convert
those ranges into a smooth probability density over calendar years and,
optionally, resample a fixed number of exact integer dates from that density
for downstream birth-death inference.

Methodology
-----------
Each item with range ``[e, l]`` is *exploded* into ``n = l - e + 1``
pseudo-points, one per integer year, each carrying weight ``1/n``.  The
resulting (year, weight) pairs are converted to ages (``age = current_year - year``)
and fed into the boundary-corrected, weighted KDE from
:func:`~bdpinf.analysis.validation_helpers.boundary_corrected_kde_on_support`.
The density is then mapped back into calendar-year space for plotting or
quantile resampling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

__all__ = [
    "RangeBandwidthSelection",
    "date_range_density",
    "explode_date_ranges",
    "quantile_resample_dates",
    "range_loo_log_likelihood",
    "select_range_bandwidth",
]


def explode_date_ranges(
    early: NDArray[np.int_],
    late: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Explode item date ranges into per-year pseudo-points with 1/n weights.

    Each item with date range ``[early, late]`` contributes
    ``n = late - early + 1`` pseudo-points (one per integer year), each with
    weight ``1/n``.  Exact-dated items (``early == late``) contribute a
    single point with weight ``1.0``.  The total weight across all pseudo-points
    equals the number of items.

    Parameters
    ----------
    early : NDArray[np.int_]
        Array of earliest year estimates (length M).
    late : NDArray[np.int_]
        Array of latest year estimates (length M); must satisfy ``late >= early``
        element-wise.

    Returns
    -------
    years : NDArray[np.float64]
        Flat array of pseudo-point years; length equals ``sum(late - early + 1)``.
    weights : NDArray[np.float64]
        Flat array of per-pseudo-point weights (sums to M, one unit per
        item).

    Raises
    ------
    ValueError
        If any entry has ``late < early``, or if ``early`` and ``late`` differ
        in length.

    """
    early_arr = np.asarray(early, dtype=np.int64)
    late_arr = np.asarray(late, dtype=np.int64)
    if early_arr.shape != late_arr.shape:
        msg = "early and late must have the same shape"
        raise ValueError(msg)
    if np.any(late_arr < early_arr):
        msg = "All late years must be >= early years"
        raise ValueError(msg)

    years_parts: list[NDArray[np.float64]] = []
    weights_parts: list[NDArray[np.float64]] = []
    for early_val, late_val in zip(early_arr.flat, late_arr.flat, strict=True):
        n = int(late_val) - int(early_val) + 1
        yr_range = np.arange(int(early_val), int(late_val) + 1, dtype=np.float64)
        years_parts.append(yr_range)
        weights_parts.append(np.full(n, 1.0 / n, dtype=np.float64))

    return np.concatenate(years_parts), np.concatenate(weights_parts)


def date_range_density(
    early: NDArray[np.int_],
    late: NDArray[np.int_],
    current_year: int,
    n_grid: int = 300,
    bandwidth: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate a kernel density of item creation dates, honouring date ranges.

    Each item's ``[early, late]`` range is exploded into integer-year
    pseudo-points with ``1/n`` weights.  A boundary-corrected, weighted KDE is
    fit in age space (``age = current_year - year``) to exploit the existing
    nonnegative-age convention, then the result is transformed back to
    calendar-year space.

    Parameters
    ----------
    early : NDArray[np.int_]
        Array of earliest year estimates.
    late : NDArray[np.int_]
        Array of latest year estimates.
    current_year : int
        Reference year used to compute ages (``age = current_year - year``).
    n_grid : int
        Number of grid points for the density estimate. Default 300.
    bandwidth : float | None
        Absolute kernel standard deviation in years.
        *None* (the default) uses Scott's rule; see
        :func:`~bdpinf.analysis.validation_helpers.boundary_corrected_kde_on_support`
        and :func:`select_range_bandwidth` for choosing one.

    Returns
    -------
    year_grid : NDArray[np.float64]
        Monotonically increasing calendar-year grid spanning
        ``[min(early), max(late)]``.
    density : NDArray[np.float64]
        Density values in calendar-year space. Integrates to approximately 1
        over *year_grid*.

    Raises
    ------
    ValueError
        If the KDE cannot be estimated (fewer than 2 finite samples, or zero
        variance after range explosion).

    """
    # Import here to avoid a circular dependency in the analysis sub-package.
    from bdpinf.analysis.validation_helpers import (
        boundary_corrected_kde_on_support,
    )

    years, weights = explode_date_ranges(early, late)
    ages = float(current_year) - years  # age = current_year - year

    max_age = float(np.max(ages))
    age_grid, age_density = boundary_corrected_kde_on_support(
        ages,
        support_upper=max_age,
        n_grid=n_grid,
        weights=weights,
        bandwidth=bandwidth,
    )
    if age_grid is None or age_density is None:
        msg = "Could not estimate date density from the provided items"
        raise ValueError(msg)

    # Transform from age space back to calendar-year space.
    # age_grid is increasing (0 → max_age); year = current_year - age is therefore
    # decreasing, so we reverse both arrays.
    year_grid = float(current_year) - age_grid[::-1]
    density = age_density[::-1]
    return year_grid, density


def quantile_resample_dates(
    early: NDArray[np.int_] | Sequence[int],
    late: NDArray[np.int_] | Sequence[int],
    current_year: int = 2024,
    n_samples: int | None = None,
    n_grid: int = 300,
) -> NDArray[np.int_]:
    """Resample exact item dates from the weighted date-range KDE.

    Builds the date density via :func:`date_range_density`, integrates it
    to form a CDF, then uses Hazen plotting positions
    ``q_k = (k - 0.5) / n_samples`` for ``k = 1 … n_samples`` to deterministically
    pick *n_samples* quantiles.  Each quantile is inverted via linear interpolation
    on the CDF grid and the result is rounded to the nearest integer year.

    The returned dates are sorted in ascending order.  All values lie within
    ``[min(early), max(late)]``.

    Parameters
    ----------
    early : NDArray[np.int_] | Sequence[int]
        Array or list of earliest year estimates.
    late : NDArray[np.int_] | Sequence[int]
        Array or list of latest year estimates.
    current_year : int
        Reference year used internally for age computation. Default 2024.
    n_samples : int | None
        Number of exact dates to return.  Defaults to ``len(early)`` (one
        resample per input item).
    n_grid : int
        Number of density grid points. Default 300.

    Returns
    -------
    NDArray[np.int_]
        Sorted array of *n_samples* integer calendar years whose empirical
        distribution approximates the item date density.

    """
    early_arr = np.asarray(early, dtype=np.int64)
    late_arr = np.asarray(late, dtype=np.int64)
    if n_samples is None:
        n_samples = len(early_arr)

    year_grid, density = date_range_density(
        early_arr,
        late_arr,
        current_year,
        n_grid=n_grid,
    )

    # Build CDF via cumulative trapezoidal integration, normalized to [0, 1].
    # Each element: cdf[i] = integral of density from year_grid[0] to year_grid[i].
    dx = np.diff(year_grid)
    avg_density = (density[:-1] + density[1:]) / 2.0
    cdf: NDArray[np.float64] = np.concatenate([[0.0], np.cumsum(avg_density * dx)])
    final = float(cdf[-1])
    if final > 0.0:
        cdf = cdf / final

    # Hazen plotting positions: (k - 0.5) / n for k = 1..n.
    quantiles = (np.arange(1, n_samples + 1, dtype=np.float64) - 0.5) / float(n_samples)

    # Invert CDF via linear interpolation, then clamp to the physical date range.
    # KDE bandwidth can place a small tail beyond max(late); dates there are artifacts.
    resampled_years = np.interp(quantiles, cdf, year_grid)
    min_year = float(np.min(early_arr))
    max_year = float(np.max(late_arr))
    resampled_years = np.clip(resampled_years, min_year, max_year)
    return np.round(resampled_years).astype(np.int64)


LeaveOut = Literal["self", "identical"]


@dataclass(frozen=True)
class _RangeKdeData:
    """Per-source arrays shared by every bandwidth in the leave-one-out score."""

    ages: NDArray[np.float64]  # age of each integer-year pseudo-point, shape [Y]
    support: float  # upper support limit (the largest age)
    lo: NDArray[np.float64]  # integration window per item, in age, shape [n]
    hi: NDArray[np.float64]
    weights: NDArray[np.float64]  # item-by-year pseudo-point weights, shape [n, Y]
    multiplicity: NDArray[np.float64]  # items removed with item i, shape [n]


def _prepare_range_data(
    early: NDArray[np.int_] | Sequence[int],
    late: NDArray[np.int_] | Sequence[int],
    current_year: int,
    leave_out: LeaveOut,
) -> _RangeKdeData:
    if leave_out not in ("self", "identical"):
        msg = "leave_out must be 'self' or 'identical'"
        raise ValueError(msg)
    e = np.asarray(early, dtype=np.int64)
    l_ = np.asarray(late, dtype=np.int64)
    if e.shape != l_.shape or e.ndim != 1:
        msg = "early and late must be 1-D arrays of the same length"
        raise ValueError(msg)
    if np.any(l_ < e):
        msg = "All late years must be >= early years"
        raise ValueError(msg)
    if e.size < 2:
        msg = "At least two items are needed"
        raise ValueError(msg)
    first = int(e.min())
    years = np.arange(first, int(l_.max()) + 1)
    in_range = (years[None, :] >= e[:, None]) & (years[None, :] <= l_[:, None])
    weights = in_range / in_range.sum(axis=1, keepdims=True)
    support = float(current_year - first)
    if leave_out == "identical":
        _, inverse, counts = np.unique(
            np.stack([e, l_]), axis=1, return_inverse=True, return_counts=True
        )
        multiplicity = counts[inverse.ravel()].astype(np.float64)
    else:
        multiplicity = np.ones(e.size)
    return _RangeKdeData(
        ages=float(current_year) - years.astype(np.float64),
        support=support,
        # Each item covers the integer-year cells [early - 0.5, late + 0.5].
        lo=np.clip(current_year - l_ - 0.5, 0.0, support),
        hi=np.clip(current_year - e + 0.5, 0.0, support),
        weights=weights,
        multiplicity=multiplicity,
    )


def _loo_terms(data: _RangeKdeData, bandwidth: float) -> NDArray[np.float64]:
    """Per-item leave-out log probabilities for one bandwidth."""
    support = data.support
    images = np.stack([data.ages, -data.ages, 2.0 * support - data.ages])  # [3, Y]
    # Mass of each pseudo-point's three reflected kernels inside [0, support].
    year_mass = (
        ndtr((support - images) / bandwidth) - ndtr((0.0 - images) / bandwidth)
    ).sum(axis=0)
    total_weights = data.weights.sum(axis=0)
    total_den = float(total_weights @ year_mass)
    own_den = data.weights @ year_mass
    den = total_den - data.multiplicity * own_den

    num = np.empty(data.lo.size)
    chunk = max(1, 4_000_000 // max(images.shape[1], 1))
    for start in range(0, data.lo.size, chunk):
        sl = slice(start, start + chunk)
        # Mass of each year's reflected kernels in each item's window: [c, Y].
        k = np.zeros((len(data.lo[sl]), images.shape[1]))
        for image in images:
            k += ndtr((data.hi[sl, None] - image) / bandwidth)
            k -= ndtr((data.lo[sl, None] - image) / bandwidth)
        num[sl] = k @ total_weights - data.multiplicity[sl] * np.einsum(
            "iy,iy->i", k, data.weights[sl]
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.maximum(num, 0.0)) - np.log(den)


def range_loo_log_likelihood(
    early: NDArray[np.int_] | Sequence[int],
    late: NDArray[np.int_] | Sequence[int],
    bandwidth: float,
    current_year: int,
    *,
    leave_out: LeaveOut = "self",
) -> float:
    r"""Leave-one-out range log-likelihood of a date-range KDE bandwidth.

    The estimate $\hat g_h$ is the one :func:`date_range_density` fits in age
    space: every item is exploded into integer-year pseudo-points of weight
    $1/\text{width}$, a weighted Gaussian KDE with standard deviation $h$
    is placed on them, reflected at age 0 and at the largest age
    $U = \text{current\_year} - \min(\text{early})$, and normalised on $[0, U]$.
    The score is
    $\sum_i \log \int_{\text{range}_i} \hat g_{h,-i}$,
    where $\hat g_{h,-i}$ drops item $i$'s pseudo-points and renormalises.

    Both integrals are in closed form with the Gaussian CDF, including the two
    reflected kernels and the support normalisation, so no grid is used.
    Pseudo-points are point masses at integer years, exactly as in
    :func:`date_range_density`.
    Item $i$'s range is integrated over its integer-year cells,
    ``[early - 0.5, late + 0.5]`` in years (clipped to the support), so an
    exact-dated item has a positive probability, one year wide.
    Points for the estimator and cells for the scored range keep the score
    consistent with how :func:`date_range_density` treats a year, and avoid a
    zero-width integral for exact-dated items.

    Parameters
    ----------
    early, late : array-like of int
        Earliest and latest year of each item, with ``late >= early``.
    bandwidth : float
        Kernel standard deviation in years.
    current_year : int
        Reference year; ``age = current_year - year``.
    leave_out : {"self", "identical"}
        ``"self"`` drops only item $i$.
        ``"identical"`` drops every item with the same ``(early, late)``,
        which avoids under-smoothing when ties (for example century-wide
        ranges) let a tied item predict itself.

    Returns
    -------
    float
        The summed log-likelihood; larger is better.

    Raises
    ------
    ValueError
        If the bandwidth is not positive, the inputs are malformed, or
        ``leave_out`` is unknown.

    """
    if not (np.isfinite(bandwidth) and bandwidth > 0):
        msg = "bandwidth must be a positive finite number"
        raise ValueError(msg)
    data = _prepare_range_data(early, late, current_year, leave_out)
    return float(np.sum(_loo_terms(data, float(bandwidth))))


@dataclass(frozen=True)
class RangeBandwidthSelection:
    """Result of :func:`select_range_bandwidth`.

    Attributes
    ----------
    bandwidths : NDArray
        The bandwidth grid, in years (ascending).
    curves : dict[str, NDArray]
        Leave-one-out log-likelihood of each source at each bandwidth.
    optimal : dict[str, float]
        Maximising bandwidth of each source's own curve.
    shared_optimum : float
        Bandwidth maximising the sum of the curves.
    loss_at_shared : dict[str, float]
        Each source's own best minus its value at ``shared_optimum``
        (non-negative, in log-likelihood units).
    leave_out : str
        The leave-out rule used.

    """

    bandwidths: NDArray[np.float64]
    curves: dict[str, NDArray[np.float64]]
    optimal: dict[str, float]
    shared_optimum: float
    loss_at_shared: dict[str, float]
    leave_out: str


def select_range_bandwidth(
    sources: Mapping[str, tuple[NDArray[np.int_], NDArray[np.int_], int]],
    bandwidths: Sequence[float] | NDArray[np.float64],
    *,
    leave_out: LeaveOut = "self",
) -> RangeBandwidthSelection:
    """Choose a date-range KDE bandwidth by leave-one-out range likelihood.

    Evaluates :func:`range_loo_log_likelihood` for every source on a common
    bandwidth grid, then reports each source's optimum and the single
    bandwidth maximising the summed score.
    The per-source data are prepared once, so a grid of about 30 bandwidths on
    a few thousand items takes seconds.

    Parameters
    ----------
    sources : Mapping[str, tuple]
        Source name to ``(early, late, current_year)``.
    bandwidths : array-like of float
        Positive bandwidths in years; sorted internally.
    leave_out : {"self", "identical"}
        Leave-out rule, see :func:`range_loo_log_likelihood`.

    Returns
    -------
    RangeBandwidthSelection

    Raises
    ------
    ValueError
        If there are no sources or the grid is empty or not positive.

    """
    grid = np.sort(np.asarray(bandwidths, dtype=np.float64))
    if not sources or grid.size == 0 or not np.all(grid > 0):
        msg = "sources and a positive bandwidth grid are required"
        raise ValueError(msg)
    curves: dict[str, NDArray[np.float64]] = {}
    for name, (early, late, current_year) in sources.items():
        data = _prepare_range_data(early, late, current_year, leave_out)
        curves[name] = np.array([np.sum(_loo_terms(data, float(h))) for h in grid])
    total = np.sum(list(curves.values()), axis=0)
    best = int(np.argmax(total))
    return RangeBandwidthSelection(
        bandwidths=grid,
        curves=curves,
        optimal={name: float(grid[int(np.argmax(c))]) for name, c in curves.items()},
        shared_optimum=float(grid[best]),
        loss_at_shared={name: float(np.max(c) - c[best]) for name, c in curves.items()},
        leave_out=leave_out,
    )
