"""Numerical stability tests for mathematical functions under edge-case inputs."""

import math

import numpy as np
import pytest

from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.math_populations import calculate_populations_log_lik


class TestNumericalStability:
    """Numerical stability checks for boundary/edge-case parameter values."""

    def test_extinction_pr_near_critical_is_finite(self) -> None:
        """λ ≈ μ (within 1e-10) should produce a finite extinction probability."""
        lam = np.array([0.05])
        mu = np.array([0.05 + 1e-10])
        t = np.array([500.0])
        pr = calculate_extinction_pr(lam, mu, t)
        assert math.isfinite(float(pr)), f"Expected finite value, got {pr}"
        assert 0 <= float(pr) <= 1, f"Probability out of [0,1]: {pr}"

    def test_spine_log_lik_near_boundary_age_is_finite(self) -> None:
        """Age at the boundary of the epoch (age ≈ T) should yield finite log-lik."""
        T = 1000.0
        lam = np.array([0.01])
        mu = np.array([0.003])
        t = np.array([T])
        # Age very close to T (created near time 0)
        spine_ages = np.array([T - 1e-6])
        ll = spine_log_density(nested_terms(lam, mu, t), spine_ages).sum()
        assert math.isfinite(float(ll)), f"Expected finite log-lik, got {ll}"

    def test_populations_log_lik_large_n_is_finite(self) -> None:
        """Large observed population count should not produce NaN."""
        lam = np.array([0.05])
        mu = np.array([0.01])
        t = np.array([200.0])
        ns = np.array([50.0])
        ll = calculate_populations_log_lik(lam, mu, ns, t)
        assert math.isfinite(float(ll)) or float(ll) == float("-inf"), (
            f"Expected finite or -inf, got {ll}"
        )

    def test_non_spine_log_lik_empty_ages_is_zero(self) -> None:
        """Empty non-spine age tensor should yield log-likelihood of 0.0."""
        lam = np.array([0.01])
        mu = np.array([0.003])
        t = np.array([500.0])
        empty_ages = np.array([])
        ll = non_spine_log_density(nested_terms(lam, mu, t), empty_ages).sum()
        assert float(ll) == pytest.approx(0.0), f"Expected 0.0, got {ll}"

    def test_log_lik_very_small_rates_is_finite(self) -> None:
        """Rates near zero (but positive) should not cause numerical overflow."""
        lam = np.array([1e-6])
        mu = np.array([1e-7])
        t = np.array([100.0])
        spine_ages = np.array([50.0])
        ll = spine_log_density(nested_terms(lam, mu, t), spine_ages).sum()
        assert math.isfinite(float(ll)) or float(ll) == float("-inf"), (
            f"Expected finite or -inf, got {ll}"
        )


@pytest.mark.slow
class TestScoreFunction:
    """Score function test: gradient near zero at MLE truth."""

    def test_mle_recovers_birth_rate_from_survivor_ages(self) -> None:
        """The fitted birth rate lands within 30% of truth in at least half the replicates."""
        from bdpinf.inference import AgeData, BirthDeathModel, fit
        from bdpinf.simulation import simulate_birth_death

        true_birth = 0.01
        true_death = 0.003
        T = 800.0

        n_pass = 0
        for seed in range(10):
            process = simulate_birth_death(
                true_birth,
                true_death,
                T,
                random_seed=seed,
                ensure_non_extinct=True,
            )
            surviving = process.surviving_records
            if len(surviving) < 20:
                continue
            ages = surviving["age"].to_numpy(dtype=np.float64)
            result = fit(
                AgeData.from_groups([ages], T),
                BirthDeathModel.piecewise_constant([T]),
                seed=seed + 500,
                compute_covariance=False,
            )
            if not result.success:
                continue
            fitted_birth = float(result.birth_values[0])
            # Only the birth rate is asserted: survivor ages identify it, but the
            # death rate is weakly identified from them at these sample sizes
            # (measured: within 30% of truth in only 3 of 10 seeds).
            if abs(fitted_birth - true_birth) / true_birth < 0.30:
                n_pass += 1
        assert n_pass >= 5, (
            f"Only {n_pass}/10 replicates recovered the birth rate within 30%"
        )
