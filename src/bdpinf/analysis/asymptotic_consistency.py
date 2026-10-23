"""Sampling and fitting helpers for asymptotic-consistency validation.

Sampling and fitting functions for the asymptotic-consistency notebook
(``07_asymptotic_consistency.qmd``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from bdpinf.analysis.spike_detection import bootstrap_seed, replicate_seed
from bdpinf.inference import AgeData, BirthDeathModel, FitResult, fit
from bdpinf.simulation import (
    simulate_bdp_ages_from_distributions,
    simulate_birth_death_process,
)


@dataclass(frozen=True)
class FittingRunConfig:
    """Shared fitting configuration for a validation run."""

    seed: int
    method: str = "trust-exact"
    tol: float = 1e-6
    max_iter: int = 200


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

    if regime_config["sampling"] == "distribution":
        sampled = simulate_bdp_ages_from_distributions(
            birth_rate=birth_rate,
            death_rate=death_rate,
            max_time=observation_time,
            rng=rng,
            ensure_non_extinct=True,
        )
        if sampled.spine_ages is None or sampled.non_spine_ages is None:
            return np.asarray([], dtype=np.float64)
        return np.concatenate([sampled.spine_ages, sampled.non_spine_ages]).astype(
            np.float64
        )

    simulated = simulate_birth_death_process(
        birth_rates=birth_rate,
        death_rates=death_rate,
        times=observation_time,
        random_seed=int(rep_seed),
        ensure_non_extinct=True,
    )
    survivors = simulated.surviving_records
    if survivors is None or len(survivors) == 0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(survivors["age"], dtype=np.float64)


def _replicate_record(
    regime_config: Mapping[str, Any],
    observation_time: float,
    simulation_index: int,
    rep_seed: int,
) -> dict[str, Any]:
    """Return the identifying fields shared by every replicate result row."""
    return {
        "regime": regime_config["name"],
        "sampling": regime_config["sampling"],
        "T": float(observation_time),
        "simulation_index": int(simulation_index),
        "replicate_seed": rep_seed,
        "sample_seed": rep_seed,
        "fit_seed": rep_seed,
        "bootstrap_seed": int(bootstrap_seed(rep_seed)),
    }


def _fit_ages(
    regime_config: Mapping[str, Any],
    observation_time: float,
    ages: np.ndarray,
    rep_seed: int,
    fit_config: FittingRunConfig,
    *,
    compute_covariance: bool = True,
) -> FitResult:
    """Fit one age vector starting from the regime's true rates."""
    horizon = float(observation_time)
    model = BirthDeathModel.piecewise_constant([horizon])
    return fit(
        AgeData.from_groups(
            [np.asarray(ages, dtype=np.float64)], horizon, is_ordered=True
        ),
        model,
        init={
            "birth.v0": float(regime_config["birth_rate"]),
            "death.v0": float(regime_config["death_rate"]),
        },
        seed=rep_seed,
        method=fit_config.method,
        tol=fit_config.tol,
        max_iter=fit_config.max_iter,
        compute_covariance=compute_covariance,
    )


def fit_replicate(
    regime_config: Mapping[str, Any],
    observation_time: float,
    ages: np.ndarray,
    simulation_index: int,
    *,
    fit_config: FittingRunConfig,
) -> dict[str, Any]:
    """Fit one replicate and return a serializable result record."""
    rep_seed = int(
        replicate_seed(
            fit_config.seed,
            regime_config["name"],
            regime_config["sampling"],
            observation_time,
            simulation_index,
        )
    )
    record = _replicate_record(
        regime_config, observation_time, simulation_index, rep_seed
    )
    record |= {
        "success": False,
        "population_size": int(np.size(ages)),
        "birth_estimate": np.nan,
        "death_estimate": np.nan,
    }
    if record["population_size"] == 0:
        return record

    fit_result = _fit_ages(
        regime_config,
        observation_time,
        ages,
        rep_seed,
        fit_config,
        compute_covariance=False,
    )
    if fit_result.success:
        record["success"] = True
        record["birth_estimate"] = float(fit_result.birth_values[0])
        record["death_estimate"] = float(fit_result.death_values[0])
    return record


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

    fit_result = _fit_ages(
        regime_config, observation_time, ages, int(rep_seed), fit_config
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
