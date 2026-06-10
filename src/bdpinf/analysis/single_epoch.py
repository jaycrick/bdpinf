"""Helper utilities for single-epoch validation analyses."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class _HasManuscriptRecords(Protocol):
    @property
    def manuscript_records(self) -> pd.DataFrame | None: ...


def extract_alive_ages(
    simulation_df: _HasManuscriptRecords,
    time_span: float,
) -> NDArray[np.float64]:
    """Return manuscript ages for extant records sorted by spine flag."""
    records = simulation_df.manuscript_records
    if records is None:
        msg = "extract_alive_ages requires manuscript_records"
        raise ValueError(msg)

    alive_mask = records["death_time"].isna()
    alive = records.loc[alive_mask].sort_values("is_spine", ascending=False)
    ages = time_span - alive["creation_time"].to_numpy(dtype=np.float64)
    return np.asarray(ages, dtype=np.float64)


def extract_rate_arrays(
    fitted_models: Sequence[dict[str, Any] | None],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract birth/death rate arrays from successful model fits."""
    successful = [
        model
        for model in fitted_models
        if model is not None and model.get("success", False)
    ]

    if not successful:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    birth_rates = np.concatenate(
        [np.atleast_1d(model["birth_rates"]) for model in successful],
    )
    death_rates = np.concatenate(
        [np.atleast_1d(model["death_rates"]) for model in successful],
    )
    return (
        np.asarray(birth_rates, dtype=np.float64),
        np.asarray(death_rates, dtype=np.float64),
    )


def compute_parameter_summary(
    birth_rates: Sequence[float] | NDArray[np.float64],
    death_rates: Sequence[float] | NDArray[np.float64],
    true_birth_rate: float,
    true_death_rate: float,
) -> pd.DataFrame:
    """Build a standardized summary table for rate estimates."""
    birth = np.asarray(birth_rates, dtype=np.float64)
    death = np.asarray(death_rates, dtype=np.float64)

    if birth.size == 0 or death.size == 0:
        msg = "Cannot summarize empty rate arrays"
        raise ValueError(msg)

    summary = {
        "Parameter": ["Birth Rate", "Death Rate"],
        "True Value": [true_birth_rate, true_death_rate],
        "Mean": [float(np.mean(birth)), float(np.mean(death))],
        "Median": [float(np.median(birth)), float(np.median(death))],
        "Std": [float(np.std(birth)), float(np.std(death))],
        "Min": [float(np.min(birth)), float(np.min(death))],
        "Max": [float(np.max(birth)), float(np.max(death))],
        "25%": [float(np.percentile(birth, 25)), float(np.percentile(death, 25))],
        "75%": [float(np.percentile(birth, 75)), float(np.percentile(death, 75))],
    }
    return pd.DataFrame(summary)


def summarize_fit_success(
    fitted_models: Sequence[dict[str, Any] | None],
    total_models: int,
    warning_threshold: float = 80.0,
) -> dict[str, float | int | bool]:
    """Summarize fit success/failure counts and a warning flag."""
    successful = sum(
        1
        for model in fitted_models
        if model is not None and model.get("success", False)
    )
    failed = total_models - successful
    success_rate = (successful / total_models * 100.0) if total_models > 0 else 0.0
    return {
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "warn_low_success": success_rate < warning_threshold,
    }
