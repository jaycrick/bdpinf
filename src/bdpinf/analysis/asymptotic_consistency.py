"""Sampling and fitting helpers for asymptotic-consistency validation.

This module contains the sampling and fitting functions moved out of the
Quarto script so they can be imported and tested independently.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from bdpinf.analysis.spike_detection import bootstrap_seed, replicate_seed
from bdpinf.inference import ConvergenceConfig, fit_birth_death_processes
from bdpinf.simulation import (
    simulate_bdp_ages_from_distributions,
    simulate_birth_death_process,
)

OptimizerKwargs = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class FittingRunConfig:
    """Shared fitting configuration for a validation run."""

    seed: int
    fit_optimizers: list[str]
    fit_optimizer_kwargs: OptimizerKwargs
    convergence_config: ConvergenceConfig


def sample_age_vector(
    regime_config: Mapping[str, Any],
    observation_time: float,
    simulation_index: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample ages for one regime, time point, and replicate index."""
    rep_seed = replicate_seed(
        seed,
        regime_config["name"],
        regime_config["sampling"],
        observation_time,
        simulation_index,
    )
    rng = np.random.default_rng(int(rep_seed))

    birth_rate = float(regime_config["birth_rate"])
    death_rate = float(regime_config["death_rate"])
    ages = np.asarray([], dtype=np.float64)

    if regime_config["sampling"] == "distribution":
        sampled = simulate_bdp_ages_from_distributions(
            birth_rate=birth_rate,
            death_rate=death_rate,
            max_time=observation_time,
            rng=rng,
            ensure_non_extinct=True,
        )
        if isinstance(sampled, dict):
            ages = np.asarray(sampled["spine"] + sampled["non_spine"], dtype=np.float64)
        else:
            spine_ages = sampled.spine_ages
            non_spine_ages = sampled.non_spine_ages
            if spine_ages is not None and non_spine_ages is not None:
                ages = np.asarray(
                    spine_ages.tolist() + non_spine_ages.tolist(),
                    dtype=np.float64,
                )
    else:
        simulated_process = simulate_birth_death_process(
            birth_rates=birth_rate,
            death_rates=death_rate,
            times=observation_time,
            random_seed=int(rep_seed),
            ensure_non_extinct=True,
        )
        survivors = getattr(simulated_process, "surviving_records", None)
        if survivors is None and hasattr(simulated_process, "columns"):
            survivors = simulated_process[simulated_process["survived"]]
        if survivors is not None and len(survivors) > 0:
            ages = np.asarray(survivors["age"], dtype=np.float64)

    return ages


def fit_replicate(
    regime_config: Mapping[str, Any],
    observation_time: float,
    ages: np.ndarray,
    simulation_index: int,
    *,
    fit_config: FittingRunConfig,
) -> dict[str, Any]:
    """Fit one replicate and return a serializable result record."""
    rep_seed = replicate_seed(
        fit_config.seed,
        regime_config["name"],
        regime_config["sampling"],
        observation_time,
        simulation_index,
    )
    ages = np.array(ages, dtype=np.float64, copy=True)
    if ages.size == 0:
        return {
            "regime": regime_config["name"],
            "sampling": regime_config["sampling"],
            "T": float(observation_time),
            "simulation_index": int(simulation_index),
            "replicate_seed": int(rep_seed),
            "sample_seed": int(rep_seed),
            "fit_seed": int(rep_seed),
            "bootstrap_seed": int(bootstrap_seed(int(rep_seed))),
            "success": False,
            "population_size": 0,
            "birth_estimate": np.nan,
            "death_estimate": np.nan,
        }

    fit_result = fit_birth_death_processes(
        list_of_ages=[np.asarray(ages, dtype=np.float64)],
        times=np.asarray([float(observation_time)], dtype=np.float64),
        is_ordered=True,
        perturb_initial_params_std=0.0,
        random_seed=int(rep_seed),
        initial_birth_rates=np.asarray(
            [float(regime_config["birth_rate"])],
            dtype=np.float64,
        ),
        initial_death_rates=np.asarray(
            [float(regime_config["death_rate"])],
            dtype=np.float64,
        ),
        optimizers=fit_config.fit_optimizers,
        optimizer_kwargs=fit_config.fit_optimizer_kwargs,
        convergence_config=fit_config.convergence_config,
        estimate_uncertainty=False,
    )

    if not fit_result.get("success", False):
        return {
            "regime": regime_config["name"],
            "sampling": regime_config["sampling"],
            "T": float(observation_time),
            "simulation_index": int(simulation_index),
            "replicate_seed": int(rep_seed),
            "sample_seed": int(rep_seed),
            "fit_seed": int(rep_seed),
            "bootstrap_seed": int(bootstrap_seed(int(rep_seed))),
            "success": False,
            "population_size": len(ages),
            "birth_estimate": np.nan,
            "death_estimate": np.nan,
        }

    birth_estimate = float(np.atleast_1d(fit_result["birth_rates"])[0])
    death_estimate = float(np.atleast_1d(fit_result["death_rates"])[0])

    return {
        "regime": regime_config["name"],
        "sampling": regime_config["sampling"],
        "T": float(observation_time),
        "simulation_index": int(simulation_index),
        "replicate_seed": int(rep_seed),
        "sample_seed": int(rep_seed),
        "fit_seed": int(rep_seed),
        "bootstrap_seed": int(bootstrap_seed(int(rep_seed))),
        "success": True,
        "population_size": len(ages),
        "birth_estimate": birth_estimate,
        "death_estimate": death_estimate,
    }


def run_single_replicate(
    regime_config: Mapping[str, Any],
    observation_time: float,
    simulation_index: int,
    *,
    fit_config: FittingRunConfig,
) -> dict[str, Any]:
    """Sample and fit one replicate for a given regime and time point."""
    ages = sample_age_vector(
        regime_config=regime_config,
        observation_time=observation_time,
        simulation_index=simulation_index,
        seed=fit_config.seed,
    )
    return fit_replicate(
        regime_config=regime_config,
        observation_time=observation_time,
        ages=ages,
        simulation_index=simulation_index,
        fit_config=fit_config,
    )


def reproduce_replicate(
    row: Mapping[str, Any],
    regime_lookup: Mapping[str, Mapping[str, Any]],
    *,
    fit_config: FittingRunConfig,
) -> dict[str, Any]:
    """Recreate a previously selected replicate row."""
    regime_config = regime_lookup[str(row["regime"])]
    observation_time = float(row["T"])
    simulation_index = int(row["simulation_index"])
    rep_seed = replicate_seed(
        fit_config.seed,
        regime_config["name"],
        regime_config["sampling"],
        observation_time,
        simulation_index,
    )

    ages = sample_age_vector(
        regime_config=regime_config,
        observation_time=observation_time,
        simulation_index=simulation_index,
        seed=fit_config.seed,
    )

    fit_result = fit_birth_death_processes(
        list_of_ages=[np.asarray(ages, dtype=np.float64)],
        times=np.asarray([float(observation_time)], dtype=np.float64),
        is_ordered=True,
        perturb_initial_params_std=0.0,
        random_seed=int(rep_seed),
        initial_birth_rates=np.asarray(
            [float(regime_config["birth_rate"])],
            dtype=np.float64,
        ),
        initial_death_rates=np.asarray(
            [float(regime_config["death_rate"])],
            dtype=np.float64,
        ),
        optimizers=fit_config.fit_optimizers,
        optimizer_kwargs=fit_config.fit_optimizer_kwargs,
        convergence_config=fit_config.convergence_config,
    )

    return {
        "regime": str(row["regime"]),
        "sampling": str(row["sampling"]),
        "T": observation_time,
        "simulation_index": simulation_index,
        "population_size": len(ages),
        "replicate_seed": int(rep_seed),
        "sample_seed": int(rep_seed),
        "fit_seed": int(rep_seed),
        "bootstrap_seed": int(bootstrap_seed(int(rep_seed))),
        "true_birth_rate": float(regime_config["birth_rate"]),
        "true_death_rate": float(regime_config["death_rate"]),
        "ages": np.asarray(ages, dtype=np.float64),
        "fit_result": fit_result,
    }
