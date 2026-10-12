"""Bootstrap KDE-quantile resamples of interval-censored date uncertainty.

The current deterministic pipeline
(:func:`~bdpinf.analysis.date_density.quantile_resample_dates`) produces a
single set of resampled years from the weighted, boundary-corrected KDE of
``[early, late]`` date ranges.  To quantify date uncertainty downstream, this
module generates **100,000 bootstrap resamples** of the same pipeline via a
vectorized reimplementation that avoids calling scipy's ``gaussian_kde`` in a
Python loop.

Algorithm
---------
The key insight is that all pseudo-points live on a **fixed integer-year
support**.  A bootstrap only changes the per-item multiplicity *k_m* (how many
times item *m* is drawn), hence the per-year weight — not the kernel
locations.  Every quantity that ``scipy.gaussian_kde`` with Scott's rule needs
is therefore computable from per-item aggregates:

1. **Per-year weights** ``W[y] = (1/N) Σ_m k_m c_m 1[y ∈ [e_m, l_m]]``,
   where ``c_m = 1/(l_m - e_m + 1)`` and ``N = Σ k_m``.
   Built via a sparse range-add / difference-array cumsum (precomputed once).
2. **Scott's bandwidth** derived from ``n_eff = N² / Σ k_m c_m`` (effective
   sample size), ``data_cov`` = Bessel-corrected weighted variance (formula
   derived from per-item mean and within-item variance), and
   ``factor = n_eff^{-1/5}``.
3. **Reflection KDE on age grid** ``[0, max_age]`` as per
   ``boundary_corrected_kde_on_support``: evaluate
   ``K(x) + K(-x) + K(2·max_age - x)`` for Gaussian kernel *K* with the
   computed bandwidth, normalise by ``trapz``, map back to year space.
4. **CDF → Hazen quantiles → resampled years** (identical formula to
   ``quantile_resample_dates``).

The identity multiplicity (every *k_m = 1*) exactly reproduces
``quantile_resample_dates(early, late, n_samples=M)`` up to float64 rounding.

Public API
----------
- :func:`bootstrap_quantile_resample_dates` — generate bootstrap resamples.
- :func:`save_bootstrap_samples` — persist to a compressed ``.npz`` file.
- :func:`load_bootstrap_samples` — load from disk.

Storage format
--------------
``.npz`` via :func:`numpy.savez_compressed` with keys:

``item_id_samples``
    Unsigned integer array of shape ``[n_bootstrap, n]``, width chosen from
    the input id range (see :func:`bootstrap_quantile_resample_dates`).  Row
    *b* lists the item identifiers of the bootstrap sample in ascending
    ``mid_value`` order.
``resampled_years``
    ``int16`` array of shape ``[n_bootstrap, n]``.  Row *b* lists the
    corresponding resampled calendar years (sorted ascending), matching
    the ordering of ``item_id_samples``.
``subgroup``, ``n_bootstrap_meta``, ``n_meta``, ``current_year_meta``,
``seed_meta``, ``order_meta``, ``schema_version_meta``, ``item_id_dtype_meta``
    Scalar metadata arrays (see :func:`save_bootstrap_samples`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bdpinf.analysis.spike_detection import seed_from_parts

_DEFAULT_SEED_TAG = "bootstrap_kde_v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_weight_matrix(
    early: NDArray[np.int64],
    late: NDArray[np.int64],
    y_min: int,
    y_max: int,
) -> NDArray[np.float64]:
    """Build weight matrix A[M, Y] via difference-array cumsum.

    ``A[m, y] = c_m`` (= ``1 / n_m``) if year ``y_min + y`` falls within
    ``[early_m, late_m]``, else zero.  Used to convert bootstrap
    multiplicity vectors ``k[B, M]`` into per-year weights
    ``W = k @ A / N`` without an explicit loop over items.

    Parameters
    ----------
    early, late : NDArray[np.int64]
        Length-M arrays of per-item earliest/latest year.
    y_min, y_max : int
        Global support boundaries (inclusive).

    Returns
    -------
    NDArray[np.float64]
        Shape ``[M, Y]`` where ``Y = y_max - y_min + 1``.
    """
    M = len(early)
    Y = y_max - y_min + 1
    c = 1.0 / (late - early + 1).astype(np.float64)  # [M]

    # Vectorised difference-array: +c[m] at column (early[m]-y_min),
    # -c[m] at column (late[m]-y_min+1).  Different rows → no conflicts.
    diff = np.zeros((M, Y + 1), dtype=np.float64)
    rows = np.arange(M)
    diff[rows, (early - y_min).astype(np.intp)] = c
    diff[rows, (late - y_min + 1).astype(np.intp)] = -c
    return np.cumsum(diff[:, :-1], axis=1)  # [M, Y]


def _resample_batch(
    k: NDArray[np.float64],
    A: NDArray[np.float64],
    D0_sq: NDArray[np.float64],
    D1_sq: NDArray[np.float64],
    D2_sq: NDArray[np.float64],
    x_grid: NDArray[np.float64],
    year_grid: NDArray[np.float64],
    quantiles: NDArray[np.float64],
    mid_age: NDArray[np.float64],
    within_var: NDArray[np.float64],
    c: NDArray[np.float64],
    min_year: float,
    max_year: float,
    grid_chunk: int,
) -> NDArray[np.float64]:
    """Apply the vectorized KDE-quantile pipeline to a batch of multiplicity arrays.

    All precomputed quantities (``A``, ``D*_sq``, ``x_grid``, etc.) are
    produced once per subgroup by :func:`bootstrap_quantile_resample_dates`
    and reused across batches.

    Parameters
    ----------
    k : NDArray[np.float64], shape ``[B, M]``
        Per-item multiplicities for ``B`` bootstraps.
    A : NDArray[np.float64], shape ``[M, Y]``
        Weight matrix; see :func:`_build_weight_matrix`.
    D0_sq, D1_sq, D2_sq : NDArray[np.float64], shape ``[n_grid, Y]``
        Squared age-space distances for the main and two reflection kernel
        terms; all precomputed from the fixed grid and year-age mapping.
    x_grid : NDArray[np.float64], shape ``[n_grid]``
        Age grid ``[0, max_age]`` (uniform).
    year_grid : NDArray[np.float64], shape ``[n_grid]``
        Corresponding year grid ``[y_min, current_year]`` (uniform).
    quantiles : NDArray[np.float64], shape ``[M]``
        Hazen plotting positions ``(k - 0.5) / M`` for ``k = 1..M``.
    mid_age : NDArray[np.float64], shape ``[M]``
        ``current_year - mid_value`` for each item.
    within_var : NDArray[np.float64], shape ``[M]``
        ``(n_m^2 - 1) / 12`` — within-item age variance.
    c : NDArray[np.float64], shape ``[M]``
        ``1 / n_m`` for each item.
    min_year, max_year : float
        Clamp limits for the resampled years.
    grid_chunk : int
        Number of grid points to process per inner iteration (controls
        the peak memory for the ``[B, chunk, Y]`` exp buffer).

    Returns
    -------
    NDArray[np.float64], shape ``[B, M]``
        Resampled years (sorted ascending, unrounded float64).
    """
    B, _M = k.shape
    n_grid = len(x_grid)

    N = k.sum(axis=1)  # [B]  (= M for standard bootstrap)
    N_safe = np.where(N > 0, N, 1.0)

    # ------------------------------------------------------------------ #
    # Scott's-rule bandwidth  (exact match to scipy gaussian_kde)          #
    # ------------------------------------------------------------------ #
    # Normalised per-point weight for pseudo-point of item m:
    #   w_i = (1/n_m) / N  ⟹  Σ wᵢ² = Σ_m k_m c_m / N²
    sum_kc = (k * c[None, :]).sum(axis=1)  # [B]
    mu = (k * mid_age[None, :]).sum(axis=1) / N_safe  # [B]

    # Raw weighted variance in age space (derived analytically):
    #   raw_wvar = (1/N) Σ_m k_m [(mid_age_m − μ)² + (n_m²−1)/12]
    raw_wvar = (k * ((mid_age[None, :] - mu[:, None]) ** 2 + within_var[None, :])).sum(
        axis=1
    ) / N_safe  # [B]

    # Bessel correction: divide by (1 − Σwᵢ²) = (1 − sum_kc/N²)
    bessel = 1.0 - sum_kc / (N_safe * N_safe)  # [B]
    bessel = np.where(bessel > 1e-12, bessel, 1e-12)
    data_cov = raw_wvar / bessel  # [B]
    data_cov = np.where(data_cov > 0, data_cov, 1e-12)

    # neff = N² / Σ_m k_m c_m ;  Scott factor = neff^(−1/(d+4)) for d=1
    neff = (N_safe * N_safe) / np.where(sum_kc > 0, sum_kc, 1.0)  # [B]
    h_sq = data_cov * neff ** (-0.4)  # [B]  h² = σ² · factor²
    h_sq = np.where(h_sq > 1e-12, h_sq, 1e-12)
    neg_half_inv_h2 = -0.5 / h_sq  # [B]

    # ------------------------------------------------------------------ #
    # Per-year weights  W[b, y] = Σ_m k_m c_m 1[y∈range_m] / N          #
    # ------------------------------------------------------------------ #
    W = (k @ A) / N_safe[:, None]  # [B, Y]

    # ------------------------------------------------------------------ #
    # Reflection KDE on age grid (three terms, chunked over grid axis)    #
    # ------------------------------------------------------------------ #
    #   p[b, j] = Σ_y W[b,y] · [exp(−D0[j,y]/(2h²)) + … (D1, D2)]     #
    # The 1/(h·√(2π)) prefactor is the same for all terms and cancels   #
    # in the trapz normalisation below.                                   #
    p = np.zeros((B, n_grid), dtype=np.float64)
    for D_sq in (D0_sq, D1_sq, D2_sq):
        for j0 in range(0, n_grid, grid_chunk):
            j1 = min(j0 + grid_chunk, n_grid)
            Dj = D_sq[j0:j1]  # [chunk, Y]
            # [B, chunk, Y]; einsum sums over Y without materialising product
            exps = np.exp(Dj[None, :, :] * neg_half_inv_h2[:, None, None])
            p[:, j0:j1] += np.einsum("bcy,by->bc", exps, W, optimize=True)

    # ------------------------------------------------------------------ #
    # Normalise (trapz over age grid) then map to year space              #
    # ------------------------------------------------------------------ #
    norm = np.trapezoid(p, x_grid, axis=1)  # [B]
    norm = np.where(norm > 0, norm, 1.0)
    p /= norm[:, None]

    # Reverse age→year: p[:, 0] was density at age 0 (year=current_year);
    # after reversal p_year[:, 0] is density at year y_min.
    p_year = p[:, ::-1]  # [B, n_grid]

    # ------------------------------------------------------------------ #
    # CDF → Hazen quantile inversion                                      #
    # ------------------------------------------------------------------ #
    dy = np.diff(year_grid)  # [n_grid−1]
    avg_dens = (p_year[:, :-1] + p_year[:, 1:]) / 2.0  # [B, n_grid−1]
    cdf = np.concatenate(
        [np.zeros((B, 1)), np.cumsum(avg_dens * dy[None, :], axis=1)],
        axis=1,
    )  # [B, n_grid]
    final = cdf[:, -1]
    final = np.where(final > 0, final, 1.0)
    cdf /= final[:, None]

    # For each b: invert CDF at each Hazen quantile via linear interpolation.
    # searchsorted across B rows via list comprehension (C-speed per call).
    j_idx = np.array(
        [np.searchsorted(cdf[b], quantiles, side="right") for b in range(B)]
    )  # [B, M]
    j_idx = np.clip(j_idx, 1, n_grid - 1)
    j0_idx = j_idx - 1

    b_rows = np.arange(B)[:, None]  # [B, 1]
    x0 = cdf[b_rows, j0_idx]  # [B, M]
    x1 = cdf[b_rows, j_idx]  # [B, M]
    y0 = year_grid[j0_idx]  # [B, M]
    y1 = year_grid[j_idx]  # [B, M]
    dx_cdf = np.where(x1 - x0 > 1e-15, x1 - x0, 1e-15)
    resampled = y0 + (y1 - y0) * (quantiles[None, :] - x0) / dx_cdf  # [B, M]

    return np.clip(resampled, min_year, max_year)  # [B, M]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bootstrap_quantile_resample_dates(
    early: NDArray[np.int_] | list[int],
    late: NDArray[np.int_] | list[int],
    item_ids: NDArray[np.int_] | list[int],
    mid_value: NDArray[np.floating[Any]] | list[float],
    *,
    n_bootstrap: int = 100_000,
    current_year: int = 2024,
    n_grid: int = 300,
    batch_size: int = 500,
    grid_chunk: int = 30,
    seed: int | None = None,
) -> tuple[NDArray[np.unsignedinteger[Any]], NDArray[np.int16]]:
    """Generate bootstrap KDE-quantile resamples of interval-censored dates.

    Runs the same weighted boundary-corrected KDE + Hazen-quantile pipeline as
    :func:`~bdpinf.analysis.date_density.quantile_resample_dates`, but
    bootstraps the **dataset** 100,000 times to capture date uncertainty.

    Each bootstrap:

    1. Draws *n* item indices with replacement.
    2. Computes the bandwidth-corrected weighted KDE of date ranges (vectorised;
       see module docstring for the exact formulae).
    3. Draws *n* Hazen-quantile years from the resulting CDF.
    4. Assigns sorted years to items sorted by ``mid_value``.
    5. Returns the item id and year for each slot.

    With the identity multiplicity (``k_m = 1`` for all *m*) the output
    equals ``quantile_resample_dates(early, late, n_samples=M)`` exactly.

    Parameters
    ----------
    early : array-like of int
        Length-*M* array of per-item earliest year estimates.
    late : array-like of int
        Length-*M* array of per-item latest year estimates (clamped to
        ``>= early`` internally).
    item_ids : array-like of int
        Length-*M* non-negative item identifier values.
    mid_value : array-like of float
        Length-*M* midpoint year values (used only for rank-ordering the
        resampled years, not for bandwidth computation).
    n_bootstrap : int
        Number of bootstrap resamples. Default 100,000.
    current_year : int
        Reference year for age computation. Default 2024 (matching the
        existing pipeline).
    n_grid : int
        Number of KDE grid points in age/year space. Default 300.
    batch_size : int
        Number of bootstrap samples to compute simultaneously. Controls the
        peak memory for the ``[batch_size, grid_chunk, Y]`` exp buffer.
        Reduce if RAM is tight.
    grid_chunk : int
        Number of grid points to process per inner KDE iteration. Reduces
        peak memory at the cost of slightly more loop overhead.
    seed : int or None
        Seed for the NumPy random generator.  ``None`` uses a stable default
        derived from :func:`~bdpinf.analysis.spike_detection.seed_from_parts`.

    Returns
    -------
    item_id_samples : NDArray[np.unsignedinteger], shape ``[n_bootstrap, M]``
        Bootstrap item id values, sorted by ``mid_value`` within each row.
        The dtype width is chosen from the actual id range via
        ``np.promote_types(np.min_scalar_type(item_ids.max()), np.uint16)``,
        so it is never narrower than ``uint16`` and never silently wraps for
        larger id spaces.
    resampled_years : NDArray[np.int16], shape ``[n_bootstrap, M]``
        Corresponding resampled calendar years, sorted ascending.  ``int16``
        is sufficient (years 100–1899 ≪ 32,767).

    Raises
    ------
    ValueError
        If the input arrays are empty, have mismatched lengths, or
        ``item_ids`` contains a negative value.
    """
    early_arr = np.asarray(early, dtype=np.int64)
    late_arr = np.asarray(late, dtype=np.int64)
    # Clamp late ≥ early
    late_arr = np.maximum(late_arr, early_arr)
    item_ids_arr = np.asarray(item_ids, dtype=np.int64)
    if item_ids_arr.size and item_ids_arr.min() < 0:
        msg = "item_ids must be non-negative."
        raise ValueError(msg)
    mid_value_arr = np.asarray(mid_value, dtype=np.float64)

    M = len(early_arr)
    if M == 0:
        msg = "Cannot bootstrap an empty dataset."
        raise ValueError(msg)
    if not (len(late_arr) == len(item_ids_arr) == len(mid_value_arr) == M):
        msg = "early, late, item_ids, mid_value must all have the same length."
        raise ValueError(msg)

    y_min = int(early_arr.min())
    y_max = int(late_arr.max())

    # Output id dtype: widened from the actual id range, never narrower than
    # uint16, so ids past 65535 are not silently wrapped.
    item_id_dtype: np.dtype[np.unsignedinteger[Any]] = np.promote_types(
        np.min_scalar_type(int(item_ids_arr.max())), np.uint16
    )

    # ------------------------------------------------------------------
    # Per-item precomputed quantities (invariant to bootstraps)
    # ------------------------------------------------------------------
    n_m = (late_arr - early_arr + 1).astype(np.float64)  # [M]
    c = 1.0 / n_m  # [M]  = 1/n_m
    mid_value_mid = (early_arr + late_arr) / 2.0  # [M]
    mid_age = float(current_year) - mid_value_mid  # [M]  age of midpoint
    within_var = (n_m**2 - 1.0) / 12.0  # [M]  intra-range age variance

    # Weight matrix: A[M, Y], A[m, y] = c_m if year (y_min+y) ∈ [e_m, l_m]
    A = _build_weight_matrix(early_arr, late_arr, y_min, y_max)  # [M, Y]

    # Fixed age/year grids (use global min/max so all bootstraps share
    # the same grid; bootstraps missing the earliest item use a
    # slightly wider grid, harmless since density → 0 there).
    max_age = float(current_year - y_min)
    x_grid = np.linspace(0.0, max_age, n_grid)  # [n_grid] age space
    year_grid = (float(current_year) - x_grid[::-1]).copy()  # [n_grid] year space

    # Squared age-space distances for three reflection kernel terms
    Y = y_max - y_min + 1
    y_ages = float(current_year) - (y_min + np.arange(Y, dtype=np.float64))
    xg = x_grid[:, None]  # [n_grid, 1]
    ag = y_ages[None, :]  # [1, Y]
    D0_sq = (xg - ag) ** 2  # main kernel
    D1_sq = (xg + ag) ** 2  # reflection at age = 0
    D2_sq = (xg - 2.0 * max_age + ag) ** 2  # reflection at age = max_age

    # Hazen plotting positions (same for every bootstrap since M is fixed)
    quantiles = (np.arange(1, M + 1, dtype=np.float64) - 0.5) / float(M)

    min_year_f = float(early_arr.min())
    max_year_f = float(late_arr.max())

    # RNG
    actual_seed = seed if seed is not None else seed_from_parts(_DEFAULT_SEED_TAG)
    rng = np.random.default_rng(actual_seed)

    # Output arrays (pre-allocated)
    item_id_out = np.empty((n_bootstrap, M), dtype=item_id_dtype)
    years_out = np.empty((n_bootstrap, M), dtype=np.int16)

    b_done = 0
    while b_done < n_bootstrap:
        B = min(batch_size, n_bootstrap - b_done)

        # Draw bootstrap row-indices [B, M] with replacement
        idx = rng.choice(M, size=(B, M), replace=True)  # int64

        # Compute multiplicities k[B, M]  (float64 for downstream arithmetic)
        k = np.zeros((B, M), dtype=np.float64)
        for bi in range(B):
            np.add.at(k[bi], idx[bi], 1.0)

        # Vectorised KDE + quantile pipeline
        resampled_f = _resample_batch(
            k,
            A,
            D0_sq,
            D1_sq,
            D2_sq,
            x_grid,
            year_grid,
            quantiles,
            mid_age,
            within_var,
            c,
            min_year_f,
            max_year_f,
            grid_chunk,
        )  # [B, M] float64, already sorted ascending

        # Sort bootstrapped items by mid_value; resampled_f is already
        # sorted so position i → i-th smallest mid_value item.
        mid_b = mid_value_arr[idx]  # [B, M]
        sort_orders = np.argsort(mid_b, kind="stable", axis=1)  # [B, M]
        item_b = item_ids_arr[idx]  # [B, M]

        b_rows = np.arange(B)[:, None]
        item_id_out[b_done : b_done + B] = item_b[b_rows, sort_orders].astype(
            item_id_dtype
        )
        years_out[b_done : b_done + B] = np.round(resampled_f).astype(np.int16)

        b_done += B

    return item_id_out, years_out


def bootstrap_epoch_counts(
    resampled_years: NDArray[np.int16] | NDArray[np.int64],
    epoch_bounds: list[tuple[int, int]],
) -> NDArray[np.intp]:
    """Count items per epoch for each bootstrap draw.

    Parameters
    ----------
    resampled_years : NDArray, shape ``[n_bootstrap, n]``
        Integer calendar years from :func:`bootstrap_quantile_resample_dates`
        or :func:`load_bootstrap_samples`.
    epoch_bounds : list of ``(lo, hi)`` tuples
        Inclusive-lower, exclusive-upper epoch year boundaries, derived from
        the ``(start, end, name)`` triples of an :data:`~bdpinf.types.EpochDefinition`
        sequence.

    Returns
    -------
    NDArray[np.intp], shape ``[n_bootstrap, n_eras]``
        Per-draw epoch counts — entry ``[b, k]`` is the number of
        items in bootstrap draw *b* whose resampled year falls
        in ``[epoch_bounds[k][0], epoch_bounds[k][1])``.
    """
    yrs = np.asarray(resampled_years, dtype=np.int64)
    n_bootstrap = yrs.shape[0]
    n_eras = len(epoch_bounds)
    counts = np.zeros((n_bootstrap, n_eras), dtype=np.intp)
    for k, (lo, hi) in enumerate(epoch_bounds):
        mask = (yrs >= lo) & (yrs < hi)
        counts[:, k] = mask.sum(axis=1)
    return counts


def bootstrap_target_ranges(
    resampled_years: NDArray[np.int16] | NDArray[np.int64],
    epoch_bounds: list[tuple[int, int]],
    *,
    quantile_lo: float = 0.025,
    quantile_hi: float = 0.975,
) -> NDArray[np.intp]:
    """Build per-era ``[lo, hi]`` target ranges from bootstrap 95 % CI.

    Computes the per-epoch item count distribution across all
    bootstrap draws and returns the ``[quantile_lo, quantile_hi]``
    interval as the acceptance window for each era.  The **last era is
    always unconditioned** (its bounds are set to ``[0, 2^31 - 1]``),
    matching the convention of
    :func:`~bdpinf.simulation.survival_conditioning.build_target_ranges`.

    Parameters
    ----------
    resampled_years : NDArray, shape ``[n_bootstrap, n]``
        Integer calendar years from :func:`load_bootstrap_samples`.
    epoch_bounds : list of ``(lo, hi)`` tuples
        Inclusive-lower, exclusive-upper epoch year boundaries.
    quantile_lo, quantile_hi : float
        Lower and upper quantile levels for the CI.  Defaults give a
        95 % interval (2.5th–97.5th percentiles).

    Returns
    -------
    NDArray[np.intp], shape ``[n_eras, 2]``
        Inclusive ``[lo, hi]`` bounds per era; last era is unconditioned.

    Examples
    --------
    >>> import numpy as np
    >>> years = np.array([[100, 200, 500], [150, 400, 600]], dtype=np.int16)
    >>> bounds = [(100, 325), (325, 636), (636, 867)]
    >>> bootstrap_target_ranges(years, bounds).tolist()
    [[1, 2], [1, 1], [0, 2147483647]]
    """
    counts = bootstrap_epoch_counts(resampled_years, epoch_bounds)
    n_eras = len(epoch_bounds)
    ranges = np.zeros((n_eras, 2), dtype=np.intp)
    for k in range(n_eras - 1):
        col = counts[:, k]
        ranges[k, 0] = max(0, int(np.quantile(col, quantile_lo)))
        ranges[k, 1] = int(np.quantile(col, quantile_hi))
    ranges[-1, 0] = 0
    ranges[-1, 1] = 2**31 - 1
    return ranges


_SCHEMA_VERSION = 2


def save_bootstrap_samples(
    item_id_samples: NDArray[np.unsignedinteger[Any]],
    resampled_years: NDArray[np.int16],
    path: str | Path,
    *,
    subgroup: str = "",
    n_bootstrap: int | None = None,
    current_year: int = 2024,
    seed: int = -1,
    order: str = "sorted_by_mid_value",
) -> None:
    """Persist bootstrap KDE samples to a compressed ``.npz`` file.

    Parameters
    ----------
    item_id_samples : NDArray[np.unsignedinteger], shape ``[n_bootstrap, n]``
        Bootstrap item id arrays.  The dtype is stored as-is (see
        :func:`bootstrap_quantile_resample_dates` for how it is chosen) so a
        wider id space is never silently truncated.
    resampled_years : NDArray[np.int16], shape ``[n_bootstrap, n]``
        Corresponding resampled years (same shape).
    path : str or Path
        Output path; ``.npz`` is appended by NumPy if absent.
    subgroup : str
        Name of the item subgroup (e.g. ``"non_lectionary"``).
    n_bootstrap : int or None
        Number of bootstrap samples (inferred from array shape if ``None``).
    current_year : int
        Reference year used during resampling. Default 2024.
    seed : int
        RNG seed used, or -1 if not recorded.
    order : str
        Row-ordering convention description.

    Raises
    ------
    ValueError
        If ``item_id_samples`` and ``resampled_years`` have different shapes.
    """
    ids = np.asarray(item_id_samples)
    yrs = np.asarray(resampled_years, dtype=np.int16)
    if ids.shape != yrs.shape:
        msg = f"item_id_samples shape {ids.shape} != resampled_years shape {yrs.shape}"
        raise ValueError(msg)

    n_boot = n_bootstrap if n_bootstrap is not None else ids.shape[0]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        item_id_samples=ids,
        resampled_years=yrs,
        subgroup=np.array(subgroup, dtype=object),
        n_bootstrap_meta=np.array(n_boot, dtype=np.int64),
        n_meta=np.array(ids.shape[1] if ids.ndim == 2 else 0, dtype=np.int64),
        current_year_meta=np.array(current_year, dtype=np.int64),
        seed_meta=np.array(seed, dtype=np.int64),
        order_meta=np.array(order, dtype=object),
        schema_version_meta=np.array(_SCHEMA_VERSION, dtype=np.int64),
        item_id_dtype_meta=np.array(str(ids.dtype), dtype=object),
    )


def load_bootstrap_samples(path: str | Path) -> dict[str, Any]:
    """Load bootstrap KDE samples from a ``.npz`` file.

    The file must have been written by :func:`save_bootstrap_samples`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file (with or without the ``.npz`` extension).

    Returns
    -------
    dict[str, Any]
        ``"item_id_samples"``
            Unsigned integer array of shape ``[n_bootstrap, n]``.
        ``"resampled_years"``
            ``int16`` array of shape ``[n_bootstrap, n]``.
        ``"metadata"``
            Dict with keys ``subgroup``, ``n_bootstrap``, ``n``,
            ``current_year``, ``seed``, ``order``, ``schema_version``,
            ``item_id_dtype``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (with or without ``.npz`` extension).
    ValueError
        If the file has no ``item_id_samples`` array, i.e. it was not written by
        :func:`save_bootstrap_samples`.
    """
    p = Path(path)
    if not p.exists() and not p.with_suffix(".npz").exists():
        msg = f"Bootstrap samples file not found: {p}"
        raise FileNotFoundError(msg)

    np_path = p if p.exists() else p.with_suffix(".npz")
    data = np.load(np_path, allow_pickle=True)

    def _scalar(key: str, default: Any) -> Any:
        if key not in data:
            return default
        v = data[key]
        return v[()] if v.ndim == 0 else (v.item() if v.size == 1 else v.tolist())

    if "item_id_samples" not in data:
        msg = (
            f"{np_path} has no 'item_id_samples' array; "
            "regenerate it with save_bootstrap_samples"
        )
        raise ValueError(msg)
    ids_raw = np.asarray(data["item_id_samples"])

    metadata: dict[str, Any] = {
        "subgroup": str(_scalar("subgroup", "")),
        "n_bootstrap": int(_scalar("n_bootstrap_meta", 0)),
        "n": int(_scalar("n_meta", 0)),
        "current_year": int(_scalar("current_year_meta", 2024)),
        "seed": int(_scalar("seed_meta", -1)),
        "order": str(_scalar("order_meta", "sorted_by_mid_value")),
        "schema_version": int(_scalar("schema_version_meta", _SCHEMA_VERSION)),
        "item_id_dtype": str(_scalar("item_id_dtype_meta", str(ids_raw.dtype))),
    }

    return {
        "item_id_samples": ids_raw.astype(metadata["item_id_dtype"]),
        "resampled_years": np.asarray(data["resampled_years"], dtype=np.int16),
        "metadata": metadata,
    }
