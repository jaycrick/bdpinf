"""Date density estimation for manuscripts with date ranges.

Manuscripts in the NTVMR dataset are dated as intervals ``[earlyYear, lateYear]``.
The functions here convert those ranges into a smooth probability density over
calendar years and, optionally, resample a fixed number of exact integer dates
from that density for downstream birth-death inference.

Methodology
-----------
Each manuscript with range ``[e, l]`` is *exploded* into ``n = l - e + 1``
pseudo-points, one per integer year, each carrying weight ``1/n``.  The
resulting (year, weight) pairs are converted to ages (``age = current_year - year``)
and fed into the boundary-corrected, weighted KDE from
:func:`~inf_bp_texts.analysis.validation_helpers.boundary_corrected_kde_on_support`.
The density is then mapped back into calendar-year space for plotting or
quantile resampling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def explode_date_ranges(
    early: NDArray[np.int_],
    late: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Explode manuscript date ranges into per-year pseudo-points with 1/n weights.

    Each manuscript with date range ``[early, late]`` contributes
    ``n = late - early + 1`` pseudo-points (one per integer year), each with
    weight ``1/n``.  Exact-dated manuscripts (``early == late``) contribute a
    single point with weight ``1.0``.  The total weight across all pseudo-points
    equals the number of manuscripts.

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
        manuscript).

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


def manuscript_date_density(
    early: NDArray[np.int_],
    late: NDArray[np.int_],
    current_year: int,
    n_grid: int = 300,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate a kernel density of manuscript creation dates, honouring date ranges.

    Each manuscript's ``[early, late]`` range is exploded into integer-year
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
    from inf_bp_texts.analysis.validation_helpers import (  # noqa: PLC0415
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
    )
    if age_grid is None or age_density is None:
        msg = "Could not estimate date density from the provided manuscripts"
        raise ValueError(msg)

    # Transform from age space back to calendar-year space.
    # age_grid is increasing (0 → max_age); year = current_year - age is therefore
    # decreasing, so we reverse both arrays.
    year_grid = float(current_year) - age_grid[::-1]
    density = age_density[::-1]
    return year_grid, density


def quantile_resample_dates(
    early: NDArray[np.int_],
    late: NDArray[np.int_],
    current_year: int,
    n_samples: int,
    n_grid: int = 300,
) -> NDArray[np.int_]:
    """Resample *n_samples* exact manuscript dates from the weighted date-range KDE.

    Builds the date density via :func:`manuscript_date_density`, integrates it
    to form a CDF, then uses Hazen plotting positions
    ``q_k = (k - 0.5) / n_samples`` for ``k = 1 … n_samples`` to deterministically
    pick *n_samples* quantiles.  Each quantile is inverted via linear interpolation
    on the CDF grid and the result is rounded to the nearest integer year.

    The returned dates are sorted in ascending order.  All values lie within
    ``[min(early), max(late)]``.

    Parameters
    ----------
    early : NDArray[np.int_]
        Array of earliest year estimates.
    late : NDArray[np.int_]
        Array of latest year estimates.
    current_year : int
        Reference year used internally for age computation.
    n_samples : int
        Number of exact dates to return (typically equal to the number of
        manuscripts, ``len(early)``).
    n_grid : int
        Number of density grid points. Default 300.

    Returns
    -------
    NDArray[np.int_]
        Sorted array of *n_samples* integer calendar years whose empirical
        distribution approximates the manuscript date density.

    """
    year_grid, density = manuscript_date_density(
        early,
        late,
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
    min_year = float(np.min(early))
    max_year = float(np.max(late))
    resampled_years = np.clip(resampled_years, min_year, max_year)
    return np.round(resampled_years).astype(np.int64)
