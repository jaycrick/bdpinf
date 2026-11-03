"""Fit a birth-death process to each of several named DataFrame slices, in parallel.

Given a set of :class:`DataSubset` objects — each a named slice of some larger
dataset with its own age column — :func:`fit_subset` runs a single MLE fit,
and :func:`fit_subsets_bayesian` runs NumPyro NUTS fits across all subsets
using ``joblib.Parallel(n_jobs=-1)``, mirroring the ``n_jobs=-1`` convention
used in the validation notebooks.  Each worker subprocess inherits
``JAX_PLATFORMS=cpu`` (set at package import) so CUDA is not attempted in
subprocesses.  NUTS chains remain ``chain_method="vectorized"`` (JAX vmap)
inside each worker.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bdpinf.inference.bayesian import BayesianMCMCConfig, BayesianPriorConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSubset:
    """A named slice of a DataFrame, ready to fit.

    Attributes
    ----------
    key : str
        Short identifier (e.g. ``"non_lectionary"``, ``"section_e"``).
    label : str
        Human-readable label used in plots and tables.
    frame : pd.DataFrame
        Subset of the source DataFrame.
    """

    key: str
    label: str
    frame: pd.DataFrame


@dataclass
class SubsetFitResult:
    """Result of a birth-death MLE fit for one data subset.

    Attributes
    ----------
    key : str
        Short identifier matching the :class:`DataSubset`.
    label : str
        Human-readable label.
    success : bool
        Whether the optimiser converged.
    n_individuals : int
        Total individuals in the subset (may be 0).
    times : NDArray[np.float64]
        Adjusted epoch end times from
        ``BirthDeathProcess.adjusted_epoch_end_times``.
    ages : NDArray[np.float64]
        Individual ages (in years, relative to the last epoch end).
    nll_history : list[float]
        Negative log-likelihood history from the best optimiser attempt.
    log_likelihood : float
        Log-likelihood at the MLE, or ``-inf`` on failure.
    epoch_results : list[dict[str, Any]]
        Per-epoch result dicts (one per epoch).  Keys present on success:
        ``period``, ``start``, ``end``, ``n_individuals``, ``birth_rates``,
        ``death_rates``, ``half_life``, ``success``.  Optional keys when
        covariance is available: ``birth_rate_se``, ``death_rate_se``,
        ``birth_rate_ci``, ``death_rate_ci``.
    """

    key: str
    label: str
    success: bool
    n_individuals: int
    times: NDArray[np.float64]
    ages: NDArray[np.float64]
    nll_history: list[float]
    log_likelihood: float
    epoch_results: list[dict[str, Any]] = field(default_factory=list)


def _range_columns(
    early_column: str | None, late_column: str | None
) -> tuple[str, str] | None:
    """Validate that the date-range columns are given together."""
    if early_column is None and late_column is None:
        return None
    if early_column is None or late_column is None:
        msg = "early_column and late_column must be given together"
        raise ValueError(msg)
    return early_column, late_column


def _subset_age_data(
    frame: pd.DataFrame,
    ages: NDArray[np.float64],
    times: NDArray[np.float64],
    epochs: Sequence[tuple[int, int, str]],
    start_time: int | float,
    range_cols: tuple[str, str] | None,
) -> Any:
    """Point-age or date-range ``AgeData`` for one subset."""
    from bdpinf.inference import AgeData

    if range_cols is None:
        return AgeData.from_groups([ages], float(times[-1]))
    return AgeData.from_year_ranges(
        frame[range_cols[0]].to_numpy(dtype=np.float64),
        frame[range_cols[1]].to_numpy(dtype=np.float64),
        start_year=float(start_time),
        observation_year=float(epochs[-1][1]),
    )


def fit_subset(
    subset: DataSubset,
    epochs: Sequence[tuple[int, int, str]],
    *,
    start_time: int | float,
    age_column: str,
    random_seed: int,
    early_column: str | None = None,
    late_column: str | None = None,
    **fit_kwargs: Any,
) -> SubsetFitResult:
    """Fit a birth-death process to one data subset.

    Builds a :class:`~bdpinf.BirthDeathProcess` from
    ``subset.frame[age_column]``, calls
    :func:`~bdpinf.inference.fit` with piecewise-constant rates on the epochs,
    then parses per-epoch rates, standard errors, and 95% confidence intervals
    (via the delta method on log-parameters).

    Parameters
    ----------
    subset : DataSubset
        Subset frame to fit.
    epochs : Sequence[tuple[int, int, str]]
        Epoch definitions as ``(start, end, name)`` triples.
    start_time : int | float
        Epoch start year (calendar) — typically ``epochs[0][0]``.
    age_column : str
        Column in ``subset.frame`` holding the creation years.
    random_seed : int
        Random seed forwarded to :func:`~bdpinf.inference.fit`.
    early_column, late_column : str, optional
        Columns holding each individual's earliest and latest creation year.
        When both are given the fit is over those date ranges
        (:meth:`~bdpinf.inference.AgeData.from_year_ranges`) instead of the
        point years in ``age_column``, which then only supplies the plotted
        ``ages`` and per-epoch counts.
    **fit_kwargs
        Additional keyword arguments forwarded to :func:`~bdpinf.inference.fit`
        (e.g. ``method``, ``n_starts``, ``perturb_std``, ``tol``).

    Returns
    -------
    SubsetFitResult
        On convergence failure, ``success=False`` and ``epoch_results`` is
        empty.  The result never raises — all exceptions are caught and
        logged as warnings.
    """
    from bdpinf.inference import BirthDeathModel, fit
    from bdpinf.process import BirthDeathProcess

    frame = subset.frame
    n_total = len(frame)
    n_epochs = len(epochs)

    # Placeholder arrays for early returns
    _zero_times: NDArray[np.float64] = np.zeros(n_epochs, dtype=np.float64)
    _empty_ages: NDArray[np.float64] = np.empty(0, dtype=np.float64)

    def _fail(
        msg: str,
        *,
        times: NDArray[np.float64] | None = None,
        ages: NDArray[np.float64] | None = None,
        nll: list[float] | None = None,
    ) -> SubsetFitResult:
        logger.warning("Subset %s: %s", subset.key, msg)
        return SubsetFitResult(
            key=subset.key,
            label=subset.label,
            success=False,
            n_individuals=n_total,
            times=times if times is not None else _zero_times,
            ages=ages if ages is not None else _empty_ages,
            nll_history=nll or [],
            log_likelihood=float("-inf"),
        )

    # Guard: too few individuals or column missing
    if n_total < 2 or age_column not in frame.columns:
        return _fail(
            f"too few individuals ({n_total}) or missing column '{age_column}'"
        )
    range_cols = _range_columns(early_column, late_column)
    if range_cols is not None and not set(range_cols) <= set(frame.columns):
        return _fail(f"missing date-range columns {range_cols}")

    # Build BirthDeathProcess
    try:
        creation_years: list[float] = frame[age_column].astype(float).tolist()
        bdp = BirthDeathProcess(
            creation_times=creation_years,
            epoch_end_times=[int(e[1]) for e in epochs],
            start_time=float(start_time),
        )
    except Exception as exc:
        return _fail(f"BirthDeathProcess construction failed: {exc}")

    ages_tensor = bdp.ages
    times: NDArray[np.float64] = np.asarray(
        bdp.adjusted_epoch_end_times, dtype=np.float64
    )
    ages_np: NDArray[np.float64] = np.asarray(ages_tensor, dtype=np.float64)

    # Per-epoch individual counts.  The last epoch uses an inclusive upper bound
    # so individuals dated exactly at the final year are not silently excluded.
    epoch_n: list[int] = [
        int(
            (
                (frame[age_column] >= e_start)
                & (
                    frame[age_column] <= e_end
                    if i == len(epochs) - 1
                    else frame[age_column] < e_end
                )
            ).sum()
        )
        for i, (e_start, e_end, _) in enumerate(epochs)
    ]

    # Run the fit
    try:
        result = fit(
            _subset_age_data(frame, ages_np, times, epochs, start_time, range_cols),
            BirthDeathModel.piecewise_constant(times),
            seed=random_seed,
            **fit_kwargs,
        )
    except Exception as exc:
        return _fail(f"fit raised: {exc}", times=times, ages=ages_np)

    nll_history: list[float] = [float(v) for v in result.nll_history]
    log_lik = float(result.log_likelihood)

    if not result.success:
        return SubsetFitResult(
            key=subset.key,
            label=subset.label,
            success=False,
            n_individuals=n_total,
            times=times,
            ages=ages_np,
            nll_history=nll_history,
            log_likelihood=log_lik,
        )

    # Parse per-epoch rates
    birth_rates_arr = np.atleast_1d(result.birth_values)
    death_rates_arr = np.atleast_1d(result.death_values)

    # Per-epoch uncertainty (delta method on log-parameters)
    birth_ses: np.ndarray | None = None
    death_ses: np.ndarray | None = None
    birth_cis: np.ndarray | None = None
    death_cis: np.ndarray | None = None
    standard_errors = result.standard_errors
    intervals = result.confidence_intervals()
    if standard_errors is not None and intervals is not None:
        names = result.model.names
        rate_ses = np.array([standard_errors[n] for n in names], dtype=np.float64)
        rate_cis = np.array([intervals[n] for n in names], dtype=np.float64)
        birth_ses = rate_ses[:n_epochs]
        death_ses = rate_ses[n_epochs:]
        birth_cis = rate_cis[:n_epochs]
        death_cis = rate_cis[n_epochs:]

    # Build per-epoch result dicts
    epoch_results: list[dict[str, Any]] = []
    for i, (e_start, e_end, period_name) in enumerate(epochs):
        br = float(
            birth_rates_arr[i] if birth_rates_arr.size > 1 else birth_rates_arr[0]
        )
        dr = float(
            death_rates_arr[i] if death_rates_arr.size > 1 else death_rates_arr[0]
        )
        half_life = float(np.log(2) / dr) if dr > 0.0 else float("inf")
        entry: dict[str, Any] = {
            "period": period_name,
            "start": e_start,
            "end": e_end,
            "n_individuals": epoch_n[i],
            "birth_rates": br,
            "death_rates": dr,
            "half_life": half_life,
            "success": True,
        }
        if (
            birth_ses is not None
            and death_ses is not None
            and birth_cis is not None
            and death_cis is not None
        ):
            entry["birth_rate_se"] = float(birth_ses[i])
            entry["death_rate_se"] = float(death_ses[i])
            entry["birth_rate_ci"] = (float(birth_cis[i, 0]), float(birth_cis[i, 1]))
            entry["death_rate_ci"] = (float(death_cis[i, 0]), float(death_cis[i, 1]))
        epoch_results.append(entry)

    return SubsetFitResult(
        key=subset.key,
        label=subset.label,
        success=True,
        n_individuals=n_total,
        times=times,
        ages=ages_np,
        nll_history=nll_history,
        log_likelihood=log_lik,
        epoch_results=epoch_results,
    )


# ---------------------------------------------------------------------------
# Bayesian parallel driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianSubsetConfig:
    """Configuration bundle for a single Bayesian subset fit.

    Passed to :func:`fit_subset_bayesian` (the per-worker function) and
    assembled by the caller before calling :func:`fit_subsets_bayesian`.

    Attributes
    ----------
    subset : DataSubset
        Data (must already carry the ``age_column``).
    epochs : tuple[tuple[int, int, str], ...]
        Epoch definitions as ``(start, end, name)`` triples.
    start_time : int
        Epoch start year matching ``epochs[0][0]``.
    prior_config : BayesianPriorConfig
        NumPyro prior configuration for this subset.
    age_column : str
        Column in ``subset.frame`` holding the creation years.
    early_column, late_column : str, optional
        Columns with the earliest and latest creation year; when both are given
        the fit is over those date ranges and ``age_column`` is not fitted.
    """

    subset: DataSubset
    epochs: tuple[tuple[int, int, str], ...]
    start_time: int
    prior_config: BayesianPriorConfig
    age_column: str
    early_column: str | None = None
    late_column: str | None = None


def fit_subset_bayesian(
    config: BayesianSubsetConfig,
    *,
    mcmc_config: BayesianMCMCConfig,
    model_seed: int,
    mle_perturb_std: float = 0.1,
) -> dict[str, Any]:
    """Fit a Bayesian birth-death model to one data subset.

    This is the **per-worker function** executed inside each ``joblib``
    subprocess.  It must be a module-level function (not a closure) to be
    picklable by the ``loky`` backend.

    Reproduces the notebook pattern:

    1. Build :class:`~bdpinf.BirthDeathProcess` from
       ``config.subset.frame[config.age_column]``.
    2. Run an MLE reference fit with :func:`~bdpinf.inference.fit`.
    3. Start NUTS from the MLE result.
    4. Run NumPyro NUTS via :func:`~bdpinf.inference.fit_bayesian`.
    5. Convert all JAX arrays to plain NumPy before returning so the result
       crosses the process boundary cleanly.

    Parameters
    ----------
    config : BayesianSubsetConfig
        Subset data + epoch definitions + prior + age column.
    mcmc_config : BayesianMCMCConfig
        NUTS/MCMC settings (chains, samples, warmup, …).
    model_seed : int
        RNG seed for both the MLE perturbation and the NumPyro sampler.
    mle_perturb_std : float
        If positive, the MLE reference fit also tries a start perturbed by this
        standard deviation on the log scale and keeps the better one
        (default 0.1).

    Returns
    -------
    dict[str, Any]
        NumPy-only result dict with keys: ``key``, ``label``,
        ``n_individuals``, ``epoch_names``, ``n_epochs``, ``times``,
        ``birth_rates``, ``death_rates``, ``posterior_samples``
        (``{"birth_rates": ndarray, "death_rates": ndarray}``),
        ``posterior_samples_by_chain``, ``posterior_log_samples_by_chain``,
        ``diagnostics``, ``num_divergences``, ``mle_birth_rates``,
        ``mle_death_rates``, ``mle_success``, ``elapsed``, ``success``.
    """
    from bdpinf.inference import (
        BirthDeathModel,
        FitResult,
        fit,
        fit_bayesian,
    )
    from bdpinf.process import BirthDeathProcess

    subset = config.subset
    epochs = config.epochs
    start_time = config.start_time
    age_column = config.age_column
    n_epochs = len(epochs)
    ep_names = [e[2] for e in epochs]

    range_cols = _range_columns(config.early_column, config.late_column)
    # Guard: insufficient data
    if (
        len(subset.frame) < 2
        or age_column not in subset.frame.columns
        or (range_cols is not None and not set(range_cols) <= set(subset.frame.columns))
    ):
        logger.warning(
            "Bayesian fit skipped for %s: insufficient data (n=%d)",
            subset.key,
            len(subset.frame),
        )
        return {
            "key": subset.key,
            "label": subset.label,
            "success": False,
            "n_individuals": len(subset.frame),
            "epoch_names": ep_names,
        }

    # Build BirthDeathProcess
    try:
        bdp = BirthDeathProcess(
            creation_times=subset.frame[age_column].astype(float).tolist(),
            epoch_end_times=[int(e[1]) for e in epochs],
            start_time=float(start_time),
        )
    except Exception as exc:
        logger.warning(
            "Bayesian fit skipped for %s: BDP construction failed: %s", subset.key, exc
        )
        return {
            "key": subset.key,
            "label": subset.label,
            "success": False,
            "n_individuals": len(subset.frame),
            "epoch_names": ep_names,
        }

    all_ages_np: NDArray[np.float64] = np.asarray(bdp.ages, dtype=np.float64)
    times_np: NDArray[np.float64] = np.asarray(
        bdp.adjusted_epoch_end_times, dtype=np.float64
    )
    data = _subset_age_data(
        subset.frame,
        all_ages_np,
        times_np,
        epochs,
        start_time,
        _range_columns(config.early_column, config.late_column),
    )
    model = BirthDeathModel.piecewise_constant(times_np)

    # MLE reference fit (for NUTS initialisation)
    mle: FitResult | None = None
    try:
        mle = fit(
            data,
            model,
            n_starts=2 if mle_perturb_std > 0 else 1,
            perturb_std=mle_perturb_std,
            seed=model_seed,
            compute_covariance=False,
        )
    except Exception as exc:
        logger.warning("MLE reference fit failed for %s: %s", subset.key, exc)

    # NumPyro NUTS fit
    t0 = time.perf_counter()
    bay = fit_bayesian(
        data,
        model,
        prior=config.prior_config,
        mcmc=mcmc_config,
        init=mle if mle is not None and mle.success else None,
        rng_seed=model_seed,
    )
    elapsed = time.perf_counter() - t0

    def _np(x: Any) -> NDArray[np.float64]:
        """Convert a JAX/numpy array to a plain numpy float64 array."""
        return np.asarray(x, dtype=np.float64)

    mle_success = mle is not None and mle.success
    posterior = bay.to_dict()
    return {
        "key": subset.key,
        "label": subset.label,
        "success": True,
        "n_individuals": len(subset.frame),
        "epoch_names": ep_names,
        "n_epochs": n_epochs,
        "times": _np(times_np),
        "birth_rates": _np(posterior["birth_rates"]),
        "death_rates": _np(posterior["death_rates"]),
        "posterior_samples": {
            k: _np(v) for k, v in posterior["posterior_samples"].items()
        },
        "posterior_samples_by_chain": {
            k: _np(v) for k, v in posterior["posterior_samples_by_chain"].items()
        },
        "posterior_log_samples_by_chain": {
            k: _np(v) for k, v in posterior["posterior_log_samples_by_chain"].items()
        },
        "diagnostics": posterior["diagnostics"],
        "num_divergences": int(posterior["num_divergences"]),
        "mle_birth_rates": _np(mle.birth_values) if mle_success and mle else None,
        "mle_death_rates": _np(mle.death_values) if mle_success and mle else None,
        "mle_success": mle_success,
        "elapsed": elapsed,
    }


def fit_subsets_bayesian(
    configs: list[BayesianSubsetConfig],
    *,
    mcmc_config: BayesianMCMCConfig,
    model_seed: int = 42,
    n_jobs: int = -1,
    mle_perturb_std: float = 0.1,
) -> dict[str, dict[str, Any]]:
    """Fit Bayesian birth-death models to multiple subsets in parallel.

    Uses ``joblib.Parallel(n_jobs=n_jobs)`` with the ``loky`` backend
    (process-level isolation).  Each worker calls
    :func:`fit_subset_bayesian` and returns a plain NumPy dict.

    Following the convention in the validation notebooks
    (e.g. ``qmd_scripts/validation/multi_epoch/03_bulk_inference.qmd``),
    ``n_jobs=-1`` uses all available CPU cores.  Set ``n_jobs=1`` to disable
    parallelism (useful for debugging or in environments with restricted fork
    behaviour).

    Parameters
    ----------
    configs : list[BayesianSubsetConfig]
        One config per subset.  Subsets with fewer than 2 individuals or
        missing the configured age column are skipped silently.
    mcmc_config : BayesianMCMCConfig
        Shared NUTS/MCMC settings for all subsets.
    model_seed : int
        RNG seed forwarded to all workers (default 42).
    n_jobs : int
        Number of parallel workers (``-1`` = all cores, default ``-1``).
    mle_perturb_std : float
        Standard deviation of the MLE initial-parameter perturbation.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping ``{subset_key: result_dict}``.  Only successfully completed
        fits are included (``success=True``).
    """
    from joblib import Parallel, delayed

    valid = [
        c
        for c in configs
        if len(c.subset.frame) >= 2
        and c.age_column in c.subset.frame.columns
        and {x for x in (c.early_column, c.late_column) if x is not None}
        <= set(c.subset.frame.columns)
    ]

    results: list[dict[str, Any]] = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
        delayed(fit_subset_bayesian)(
            c,
            mcmc_config=mcmc_config,
            model_seed=model_seed,
            mle_perturb_std=mle_perturb_std,
        )
        for c in valid
    )

    return {r["key"]: r for r in results if r.get("success", False)}
