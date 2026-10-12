"""Tests for Bayesian birth-death inference utilities."""

import numpy as np

from bdpinf.inference import (
    fit_birth_death_processes_bayesian,
)
from bdpinf.inference.bayesian import BayesianMCMCConfig, BayesianModelConfig


class TestBayesianFitting:
    """Test Bayesian fitting utilities."""

    def test_bayesian_model_config_defaults_to_unbias_true(self) -> None:
        assert BayesianModelConfig().condition_on_survival is True

    def test_fit_birth_death_processes_bayesian_returns_expected_structure(
        self,
    ) -> None:
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
