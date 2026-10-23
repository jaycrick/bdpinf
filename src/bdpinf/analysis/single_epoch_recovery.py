"""Simulate-then-fit replicate logic for single-epoch bulk-inference recovery studies.

Used by ``06_bulk_inference.qmd``; the heavy per-replicate work is run once by
``scripts/generate_recovery_studies.py`` and persisted via
:func:`~bdpinf.inference.save_recovery_results`, so notebook renders only load
the results.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bdpinf.analysis.validation_helpers import FitModelConfig, fit_single_model
from bdpinf.inference import FitResult
from bdpinf.simulation import simulate_bdp_ages_from_distributions


def sample_single_epoch_ages(
    seed: int,
    birth_rate: float,
    death_rate: float,
    time_span: float,
) -> NDArray[np.float64]:
    """Sample one replicate's age vector from the birth-death age distribution."""
    rng = np.random.default_rng(seed)
    sampled = simulate_bdp_ages_from_distributions(
        birth_rate=float(birth_rate),
        death_rate=float(death_rate),
        max_time=float(time_span),
        rng=rng,
    )
    assert sampled.spine_ages is not None and sampled.non_spine_ages is not None
    return np.asarray(
        sampled.spine_ages.tolist() + sampled.non_spine_ages.tolist(),
        dtype=np.float64,
    )


def run_single_epoch_recovery_replicate(
    simulation_index: int,
    seed: int,
    birth_rate: float,
    death_rate: float,
    time_span: float,
) -> dict[str, Any]:
    """Sample and fit one replicate; return a tidy result row.

    Used by the ``single_epoch_bulk_inference`` recovery study
    (``06_bulk_inference.qmd``).
    """
    ages = sample_single_epoch_ages(seed, birth_rate, death_rate, time_span)
    return _row_from_fit(
        simulation_index,
        ages,
        time_span,
        fit_single_model(
            ages,
            FitModelConfig(
                times=time_span,
                is_ordered=True,
                initial_birth_rates=np.array([birth_rate], dtype=float),
                initial_death_rates=np.array([death_rate], dtype=float),
                condition_on_survival=True,
            ),
        ),
    )


def _row_from_fit(
    simulation_index: int,
    ages: NDArray[np.float64],
    time_span: float,
    result: FitResult | None,
) -> dict[str, Any]:
    if len(ages) == 0:
        return {
            "simulation_index": simulation_index,
            "success": False,
            "population": 0,
            "mean_age": np.nan,
            "has_boundary_age": False,
            "birth_rate": np.nan,
            "death_rate": np.nan,
            "optimizer": "none",
        }

    population = len(ages)
    mean_age = float(np.mean(ages))

    if result is None or not result.success:
        return {
            "simulation_index": simulation_index,
            "success": False,
            "population": population,
            "mean_age": mean_age,
            "has_boundary_age": False,
            "birth_rate": np.nan,
            "death_rate": np.nan,
            "optimizer": "none",
        }

    has_boundary_age = bool(np.max(ages) == time_span)
    return {
        "simulation_index": simulation_index,
        "success": True,
        "population": population,
        "mean_age": mean_age,
        "has_boundary_age": has_boundary_age,
        "birth_rate": float(result.birth_values[0]),
        "death_rate": float(result.death_values[0]),
        "optimizer": result.method,
    }
