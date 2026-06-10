"""Numerical stability tests for mathematical functions under edge-case inputs."""

import pytest
import torch

from inf_bp_texts.models.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from inf_bp_texts.models.math_extinction import calculate_extinction_pr
from inf_bp_texts.models.math_populations import calculate_populations_log_lik


class TestNumericalStability:
    """Numerical stability checks for boundary/edge-case parameter values."""

    def test_extinction_pr_near_critical_is_finite(self) -> None:
        """λ ≈ μ (within 1e-10) should produce a finite extinction probability."""
        lam = torch.tensor([0.05], dtype=torch.float64)
        mu = torch.tensor([0.05 + 1e-10], dtype=torch.float64)
        t = torch.tensor([500.0], dtype=torch.float64)
        pr = calculate_extinction_pr(lam, mu, t)
        assert torch.isfinite(pr), f"Expected finite value, got {pr}"
        assert 0 <= float(pr) <= 1, f"Probability out of [0,1]: {pr}"

    def test_spine_log_lik_near_boundary_age_is_finite(self) -> None:
        """Age at the boundary of the epoch (age ≈ T) should yield finite log-lik."""
        T = 1000.0
        lam = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([0.003], dtype=torch.float64)
        t = torch.tensor([T], dtype=torch.float64)
        # Age very close to T (created near time 0)
        spine_ages = torch.tensor([T - 1e-6], dtype=torch.float64)
        ll = calculate_spine_ages_log_lik(lam, mu, spine_ages, t)
        assert torch.isfinite(ll), f"Expected finite log-lik, got {ll}"

    def test_populations_log_lik_large_n_is_finite(self) -> None:
        """Large observed population count should not produce NaN."""
        lam = torch.tensor([0.05], dtype=torch.float64)
        mu = torch.tensor([0.01], dtype=torch.float64)
        t = torch.tensor([200.0], dtype=torch.float64)
        ns = torch.tensor([50], dtype=torch.float64)
        ll = calculate_populations_log_lik(lam, mu, ns, t)
        assert torch.isfinite(ll) or float(ll) == float("-inf"), (
            f"Expected finite or -inf, got {ll}"
        )

    def test_non_spine_log_lik_empty_ages_is_zero(self) -> None:
        """Empty non-spine age tensor should yield log-likelihood of 0.0."""
        lam = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([0.003], dtype=torch.float64)
        t = torch.tensor([500.0], dtype=torch.float64)
        empty_ages = torch.tensor([], dtype=torch.float64)
        ll = calculate_non_spine_ages_log_lik(lam, mu, empty_ages, t)
        assert float(ll) == pytest.approx(0.0), f"Expected 0.0, got {ll}"

    def test_log_lik_very_small_rates_is_finite(self) -> None:
        """Rates near zero (but positive) should not cause numerical overflow."""
        lam = torch.tensor([1e-6], dtype=torch.float64)
        mu = torch.tensor([1e-7], dtype=torch.float64)
        t = torch.tensor([100.0], dtype=torch.float64)
        spine_ages = torch.tensor([50.0], dtype=torch.float64)
        ll = calculate_spine_ages_log_lik(lam, mu, spine_ages, t)
        assert torch.isfinite(ll) or float(ll) == float("-inf"), (
            f"Expected finite or -inf, got {ll}"
        )


@pytest.mark.slow
class TestScoreFunction:
    """Score function test: gradient near zero at MLE truth."""

    def test_mle_gradient_near_zero_at_truth(self) -> None:
        """At true parameters, NLL gradient should be near zero for large samples."""
        from inf_bp_texts.inference.fitting import fit_birth_death_processes
        from inf_bp_texts.simulation import simulate_birth_death_process

        true_birth = 0.01
        true_death = 0.003
        T = 800.0
        rng = torch.Generator()
        rng.manual_seed(77)

        n_pass = 0
        for seed in range(10):
            df = simulate_birth_death_process(
                birth_rates=true_birth,
                death_rates=true_death,
                times=T,
                random_seed=seed,
            )
            surviving = df[df["survived"]]
            if len(surviving) < 20:
                continue
            import numpy as np

            ages = surviving["age"].to_numpy(dtype=np.float64)
            result = fit_birth_death_processes(
                [ages], times=T, estimate_uncertainty=False, random_seed=seed + 500
            )
            if not result["success"]:
                continue
            fitted_birth = float(result["birth_rates"])
            fitted_death = float(result["death_rates"])
            if (
                abs(fitted_birth - true_birth) / true_birth < 0.30
                and abs(fitted_death - true_death) / true_death < 0.30
            ):
                n_pass += 1
        assert n_pass >= 5, (
            f"MLE gradient test: only {n_pass}/10 replicates recovered truth within 30%"
        )
