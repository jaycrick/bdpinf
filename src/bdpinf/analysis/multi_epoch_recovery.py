"""Simulate-then-fit replicate logic for multi-epoch bulk-inference recovery studies.

Used by the bulk-inference notebooks in ``qmd_scripts/validation/multi_epoch``;
the per-replicate simulate+fit work is run once by
``scripts/generate_recovery_studies.py`` and persisted via
:func:`~bdpinf.inference.save_recovery_results`:

- :func:`run_two_rate_replicate` --- ``03_bulk_inference.qmd``: a genuine
  two-epoch process (distinct birth/death rate per epoch).
- :func:`run_single_rate_replicate` --- ``04_bulk_inference_single_rate.qmd``:
  single-rate generative data fit with a two-epoch model.
- :func:`run_zero_birth_replicate` --- ``05_bulk_inference_zero_birth.qmd``:
  structural-zero second-epoch birth rate with a shared death rate, fit under
  :class:`~bdpinf.inference.fitting.ParameterConstraints`.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bdpinf.analysis.single_epoch import extract_alive_ages
from bdpinf.analysis.single_epoch_recovery import sample_single_epoch_ages
from bdpinf.analysis.validation_helpers import (
    FitModelConfig,
    fit_single_model,
    has_ages_in_each_epoch,
)
from bdpinf.inference.fitting import (
    ParameterConstraints,
)
from bdpinf.simulation import simulate_birth_death_process


def _epoch_row(
    simulation_index: int, model: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Convert a two-epoch fit result into a tidy row, or ``None`` on failure."""
    if model is None or not model.get("success", False):
        return None
    birth = np.atleast_1d(model["birth_rates"])
    death = np.atleast_1d(model["death_rates"])
    return {
        "simulation_index": simulation_index,
        "epoch_1_birth": float(birth[0]),
        "epoch_1_death": float(death[0]),
        "epoch_2_birth": float(birth[1]),
        "epoch_2_death": float(death[1]),
    }


def run_two_rate_replicate(
    simulation_index: int,
    sim_seed: int,
    birth_rates_true: NDArray[np.float64],
    death_rates_true: NDArray[np.float64],
    times: NDArray[np.float64],
    fit_config: FitModelConfig,
) -> dict[str, Any] | None:
    """Simulate one two-epoch replicate (distinct rates) and fit it.

    Returns ``None`` when the simulated ages don't span both epochs (matching
    the ``has_ages_in_each_epoch`` filter applied before fitting in
    ``03_bulk_inference.qmd``), so replicates without a usable age vector are
    dropped rather than persisted with placeholder values.
    """
    simulated = simulate_birth_death_process(
        birth_rates=birth_rates_true,
        death_rates=death_rates_true,
        times=times,
        random_seed=sim_seed,
    )
    ages = extract_alive_ages(simulated, float(times[-1]))
    if not has_ages_in_each_epoch(ages, times):
        return None
    return _epoch_row(simulation_index, fit_single_model(ages, fit_config))


def run_single_rate_replicate(
    simulation_index: int,
    sim_seed: int,
    birth_rate: float,
    death_rate: float,
    time_span: float,
    fit_times: NDArray[np.float64],
    fit_config: FitModelConfig,
) -> dict[str, Any] | None:
    """Simulate one single-rate replicate and fit a two-epoch model to it.

    Returns ``None`` when the simulated ages don't span both fit epochs,
    matching the filter applied in ``04_bulk_inference_single_rate.qmd``.
    """
    ages = sample_single_epoch_ages(sim_seed, birth_rate, death_rate, time_span)
    if not has_ages_in_each_epoch(ages, fit_times):
        return None
    return _epoch_row(simulation_index, fit_single_model(ages, fit_config))


def simulate_zero_birth_second_epoch(
    first_epoch_seed: int,
    death_time_seed: int,
    birth_rate_epoch_1: float,
    death_rate: float,
    times: NDArray[np.float64],
) -> pd.DataFrame:
    """Simulate epoch 1, then advance survivors with death-only draws in epoch 2."""
    split_time = float(times[0])
    final_time = float(times[-1])
    first_epoch = simulate_birth_death_process(
        birth_rates=np.array([birth_rate_epoch_1], dtype=float),
        death_rates=np.array([death_rate], dtype=float),
        times=np.array([split_time], dtype=float),
        random_seed=first_epoch_seed,
    )

    assert first_epoch.records is not None
    simulation = first_epoch.records.copy()
    alive_mask = simulation["death_time"].isna()
    alive_indices = simulation.index[alive_mask]
    if len(alive_indices) > 0:
        rng = np.random.default_rng(death_time_seed)
        time_to_death = rng.exponential(
            scale=1.0 / float(death_rate), size=len(alive_indices)
        )
        death_times = split_time + time_to_death
        simulation.loc[alive_indices, "death_time"] = np.where(
            death_times < final_time,
            death_times,
            np.nan,
        )
    return simulation


def fit_zero_birth_replicate(
    ages: NDArray[np.float64],
    config: FitModelConfig,
    constraints: ParameterConstraints,
) -> dict[str, Any] | None:
    """Fit one age vector under the structural-zero-birth constraint set."""
    result = fit_single_model(
        np.asarray(ages, dtype=np.float64),
        dataclasses.replace(
            config,
            perturb_initial_params_std=config.perturb_initial_params_std or 0.0,
            param_constraints=constraints,
        ),
    )
    if result is None:
        return None

    if result.get("success", False):
        birth_rates = np.asarray(result.get("birth_rates"), dtype=np.float64)
        if birth_rates.size >= 2:
            birth_rates = birth_rates.copy()
            birth_rates[1] = 0.0
            result["birth_rates"] = birth_rates
    return result


def run_zero_birth_replicate(
    simulation_index: int,
    first_epoch_seed: int,
    death_time_seed: int,
    birth_rate_epoch_1: float,
    death_rate: float,
    times: NDArray[np.float64],
    fit_config: FitModelConfig,
    param_constraints: ParameterConstraints,
) -> dict[str, Any]:
    """Simulate and fit one structural-zero-second-epoch-birth replicate.

    Used by the ``multi_epoch_zero_birth`` recovery study
    (``05_bulk_inference_zero_birth.qmd``).
    """
    simulated = simulate_zero_birth_second_epoch(
        first_epoch_seed=first_epoch_seed,
        death_time_seed=death_time_seed,
        birth_rate_epoch_1=birth_rate_epoch_1,
        death_rate=death_rate,
        times=times,
    )
    final_time = float(times[-1])
    alive = simulated.loc[simulated["death_time"].isna()].sort_values(
        "is_spine", ascending=False
    )
    ages = np.asarray(final_time - alive["creation_time"].to_numpy(dtype=np.float64))

    model = fit_zero_birth_replicate(ages, fit_config, param_constraints)

    if len(ages) == 0 or model is None or not model.get("success", False):
        return {
            "simulation_index": simulation_index,
            "success": False,
            "population": len(ages),
            "mean_age": float(np.mean(ages)) if len(ages) else np.nan,
            "has_boundary_age": bool(np.max(ages) == final_time)
            if len(ages)
            else False,
            "birth_rate": np.nan,
            "death_rate": np.nan,
        }

    return {
        "simulation_index": simulation_index,
        "success": True,
        "population": len(ages),
        "mean_age": float(np.mean(ages)),
        "has_boundary_age": bool(np.max(ages) == final_time),
        "birth_rate": float(np.atleast_1d(model["birth_rates"])[0]),
        "death_rate": float(np.atleast_1d(model["death_rates"])[0]),
    }
