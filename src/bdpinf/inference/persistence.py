"""Posterior sample and recovery-study persistence for birth-death inference.

Thin save/load wrappers around ``numpy.savez_compressed`` so that posterior
samples produced by :func:`~bdpinf.inference.fit_bayesian` (or the subset
wrapper :func:`~bdpinf.analysis.subset_fitting.fit_subsets_bayesian`)
can be written to disk once and reloaded into downstream analysis notebooks
without re-running the slow MCMC step.

File format
-----------
``.npz`` with keys:

``birth_rates``
    Float64 array of shape ``[n_draws, n_epochs]`` — posterior birth-rate
    samples, flat over chains.
``death_rates``
    Float64 array of shape ``[n_draws, n_epochs]`` — posterior death-rate
    samples, flat over chains.
``epoch_end_times``
    Float64 array of shape ``[n_epochs]`` — era boundary end times in *process
    time* (i.e. years since the process start, not calendar years).
``epoch_names``
    Object array of strings of shape ``[n_epochs]`` — human-readable era names
    (e.g. "Epoch 1").

All four keys are always present; ``epoch_end_times`` and ``epoch_names`` are
set to empty arrays when the caller omits them.

This module also holds :func:`save_recovery_results` /
:func:`load_recovery_results`, a schema-agnostic ``.npz`` round-tripper for
the tidy per-replicate DataFrames produced by simulate-then-fit validation
studies (e.g. :mod:`~bdpinf.analysis.single_epoch_recovery`,
:mod:`~bdpinf.analysis.multi_epoch_recovery`,
:mod:`~bdpinf.analysis.asymptotic_consistency`). Those studies are the
most expensive cells in the ``qmd_scripts/validation`` notebooks; precomputing
them once via ``scripts/generate_recovery_studies.py`` and loading the result
here keeps the notebooks themselves fast to render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def save_posterior_samples(
    posterior_samples: dict[str, Any],
    path: str | Path,
    *,
    epoch_end_times: NDArray[np.float64] | list[float] | None = None,
    epoch_names: list[str] | None = None,
) -> None:
    """Persist posterior birth/death-rate samples to a compressed ``.npz`` file.

    Parameters
    ----------
    posterior_samples : dict[str, Any]
        The ``"posterior_samples"`` sub-dict from
        :func:`~bdpinf.inference.fit_bayesian`, with keys
        ``"birth_rates"`` and ``"death_rates"`` (each shape
        ``[n_draws, n_epochs]``).
    path : str or Path
        Output file path.  The ``.npz`` extension is appended by NumPy if
        absent.
    epoch_end_times : array-like of float, optional
        Era boundary end times in process time (years since process start).
        Stored verbatim so the loading notebook does not need to re-derive
        them.  Defaults to an empty float64 array.
    epoch_names : list[str], optional
        Human-readable era names (e.g. "Epoch 1").  Defaults to an
        empty object array.

    Raises
    ------
    KeyError
        If ``posterior_samples`` is missing ``"birth_rates"`` or
        ``"death_rates"``.
    ValueError
        If ``birth_rates`` and ``death_rates`` have incompatible shapes.
    """
    birth = np.asarray(posterior_samples["birth_rates"], dtype=np.float64)
    death = np.asarray(posterior_samples["death_rates"], dtype=np.float64)

    if birth.shape != death.shape:
        msg = f"birth_rates shape {birth.shape} != death_rates shape {death.shape}"
        raise ValueError(msg)

    et = (
        np.asarray(epoch_end_times, dtype=np.float64)
        if epoch_end_times is not None
        else np.empty(0, dtype=np.float64)
    )
    names = (
        np.asarray(epoch_names, dtype=object)
        if epoch_names is not None
        else np.empty(0, dtype=object)
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        birth_rates=birth,
        death_rates=death,
        epoch_end_times=et,
        epoch_names=names,
    )


def load_posterior_samples(path: str | Path) -> dict[str, Any]:
    """Load posterior samples from a ``.npz`` file.

    The file must have been written by :func:`save_posterior_samples`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file (with or without the extension).

    Returns
    -------
    dict[str, Any]
        Keys:

        ``"birth_rates"``
            Float64 array of shape ``[n_draws, n_epochs]``.
        ``"death_rates"``
            Float64 array of shape ``[n_draws, n_epochs]``.
        ``"epoch_end_times"``
            Float64 array of shape ``[n_epochs]`` (empty if not stored).
        ``"epoch_names"``
            Object array of strings of shape ``[n_epochs]`` (empty if not
            stored).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required keys are absent from the ``.npz`` archive.
    """
    p = Path(path)
    if not p.exists() and not p.with_suffix(".npz").exists():
        msg = f"Posterior samples file not found: {p}"
        raise FileNotFoundError(msg)

    data = np.load(p if p.exists() else p.with_suffix(".npz"), allow_pickle=True)
    return {
        "birth_rates": np.asarray(data["birth_rates"], dtype=np.float64),
        "death_rates": np.asarray(data["death_rates"], dtype=np.float64),
        "epoch_end_times": np.asarray(data["epoch_end_times"], dtype=np.float64),
        "epoch_names": np.asarray(data["epoch_names"], dtype=object),
    }


def _column_array(series: pd.Series) -> NDArray[Any]:
    """Convert one DataFrame column to an ``.npz``-friendly array.

    Numeric and boolean columns keep their native NumPy dtype; everything
    else (strings, mixed types) is stored as an object array so arbitrary
    per-replicate metadata (regime names, optimizer labels, …) round-trips.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.to_numpy()
    return np.asarray(series.tolist(), dtype=object)


def save_recovery_results(
    results_df: pd.DataFrame,
    path: str | Path,
    *,
    metadata: dict[str, int | float | str] | None = None,
) -> None:
    """Persist a tidy per-replicate recovery-study DataFrame to ``.npz``.

    Parameters
    ----------
    results_df : pd.DataFrame
        One row per simulate-then-fit replicate. Column names and dtypes are
        preserved verbatim; there is no fixed schema, so different studies
        (single-epoch bulk inference, asymptotic consistency, multi-epoch
        constrained fits, …) can each store whatever columns their
        downstream notebook cells consume.
    path : str or Path
        Output file path. The ``.npz`` extension is appended by NumPy if
        absent.
    metadata : dict[str, int | float | str], optional
        Small scalar metadata (e.g. ``n_simulated``, ``n_kept``) stored
        alongside the table.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "_columns": np.asarray(list(results_df.columns), dtype=object),
    }
    for column in results_df.columns:
        payload[f"col__{column}"] = _column_array(results_df[column])

    meta = metadata or {}
    payload["_metadata_keys"] = np.asarray(list(meta.keys()), dtype=object)
    for key, value in meta.items():
        payload[f"meta__{key}"] = np.asarray(value)

    np.savez_compressed(out, **payload)


def load_recovery_results(path: str | Path) -> dict[str, Any]:
    """Load a recovery-study DataFrame written by :func:`save_recovery_results`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file (with or without the extension).

    Returns
    -------
    dict[str, Any]
        ``"results"``
            The reconstructed ``pd.DataFrame``, column order preserved.
        ``"metadata"``
            Dict of scalar metadata stored at save time (empty if none).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists() and not p.with_suffix(".npz").exists():
        msg = f"Recovery results file not found: {p}"
        raise FileNotFoundError(msg)

    data = np.load(p if p.exists() else p.with_suffix(".npz"), allow_pickle=True)

    columns = [str(c) for c in data["_columns"]]
    results_df = pd.DataFrame({column: data[f"col__{column}"] for column in columns})

    metadata: dict[str, Any] = {}
    if "_metadata_keys" in data:
        for key in data["_metadata_keys"]:
            key = str(key)
            value = data[f"meta__{key}"]
            metadata[key] = value.item() if value.ndim == 0 else value

    return {"results": results_df, "metadata": metadata}
