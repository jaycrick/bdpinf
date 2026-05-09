"""Tests for Bayesian birth-death inference utilities."""

# ruff: noqa: S101

import numpy as np
import pandas as pd

from inf_bp_texts.inference import (
    fit_birth_death_processes_bayesian,
    fit_birth_death_processes_bayesian_from_data_frame,
)
from inf_bp_texts.inference.bayesian import BayesianMCMCConfig, BayesianModelConfig


def test_bayesian_model_config_defaults_to_unbias_true() -> None:
    assert BayesianModelConfig().unbias is True


def test_fit_birth_death_processes_bayesian_returns_expected_structure() -> None:
    ages = np.array([50.0, 120.0, 180.0, 220.0], dtype=np.float64)

    result = fit_birth_death_processes_bayesian(
        list_of_ages=[ages],
        times=300.0,
        model_config=BayesianModelConfig(rng_seed=7),
        mcmc_config=BayesianMCMCConfig(
            num_samples=20,
            warmup_steps=20,
            num_chains=1,
            target_accept_prob=0.8,
            disable_progressbar=True,
        ),
    )

    assert result["success"]
    assert "birth_rates" in result
    assert "death_rates" in result
    assert "birth_rates_90_ci" in result
    assert "death_rates_90_ci" in result
    assert result["birth_rates"].shape == (1,)
    assert result["death_rates"].shape == (1,)


def test_fit_birth_death_processes_bayesian_from_data_frame_auto_age_column() -> None:
    df = pd.DataFrame({"age": [40.0, 75.0, 110.0, 160.0]})

    result = fit_birth_death_processes_bayesian_from_data_frame(
        df=df,
        times=250.0,
        model_config=BayesianModelConfig(rng_seed=11),
        mcmc_config=BayesianMCMCConfig(
            num_samples=20,
            warmup_steps=20,
            num_chains=1,
            disable_progressbar=True,
        ),
    )

    assert result["success"]
    assert result["posterior_samples"]["birth_rates"].shape[1] == 1
    assert result["posterior_samples"]["death_rates"].shape[1] == 1
