"""Regression tests: golden-value freeze for core likelihood functions.

These tests record the numerical output of key mathematical functions at
canonical parameter values. Any change to the output — including intentional
numerical-stability improvements — triggers a failure and forces the author
to update the golden value with an explanatory comment.

Usage
-----
Run with the rest of the fast test suite (included by default):

    uv run pytest tests/test_regression_likelihoods.py

To update a golden value after a deliberate change:
1. Re-run the function manually to obtain the new value.
2. Update the constant below and add a comment explaining the change.
"""

import numpy as np
import pytest
import torch

from inf_bp_texts.inference.likelihood import log_likelihood
from inf_bp_texts.models.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from inf_bp_texts.models.math_extinction import calculate_extinction_pr
from inf_bp_texts.models.math_populations import calculate_populations_log_lik

# ---------------------------------------------------------------------------
# Canonical parameter set (single epoch, supercritical)
# ---------------------------------------------------------------------------
_BIRTH = torch.tensor([0.05], dtype=torch.float64)
_DEATH = torch.tensor([0.03], dtype=torch.float64)
_TIMES = torch.tensor([1500.0], dtype=torch.float64)
_AGE_750 = torch.tensor([750.0], dtype=torch.float64)
_NS_5 = torch.tensor([5], dtype=torch.int64)
_AGES_10 = np.array(
    [100.0, 200.0, 300.0, 450.0, 600.0, 750.0, 900.0, 1050.0, 1200.0, 1350.0]
)

# ---------------------------------------------------------------------------
# Golden values — computed 2026-06-09 from the initial implementation
# ---------------------------------------------------------------------------
# Supercritical (λ=0.05, μ=0.03, T=1500): Pr(ext) converges to μ/λ = 0.6 for large T
_GOLDEN_EXTINCTION_SUPERCRITICAL = 0.6000000000
# Subcritical (λ=0.03, μ=0.05, T=1500): near-certain extinction
_GOLDEN_EXTINCTION_SUBCRITICAL = 1.0000000000
# Critical (λ=μ=0.05, T=1500): μT/(1 + μT) = 75/76
_GOLDEN_EXTINCTION_CRITICAL = 0.9868421053
# Spine age log-lik at age=750
_GOLDEN_SPINE_LL_750 = -25.0902674713
# Non-spine age log-lik at age=750
_GOLDEN_NON_SPINE_LL_750 = -40.4957324571
# Population log-lik at n=5
_GOLDEN_POPULATIONS_LL_N5 = -31.8330083712
# Combined log-likelihood for 10 fixed ages
_GOLDEN_COMBINED_LL_10AGES = -381.6393679271

_ATOL = 1e-4

# ---------------------------------------------------------------------------
# Canonical parameter set (two-epoch, supercritical → subcritical)
# Epoch 1: birth 0.05, death 0.03, T=750  (supercritical)
# Epoch 2: birth 0.01, death 0.02, T=1500 (subcritical)
# ---------------------------------------------------------------------------
_BIRTH_2 = torch.tensor([0.05, 0.01], dtype=torch.float64)
_DEATH_2 = torch.tensor([0.03, 0.02], dtype=torch.float64)
_TIMES_2 = torch.tensor([750.0, 1500.0], dtype=torch.float64)
_AGE_300 = torch.tensor([300.0], dtype=torch.float64)
_NS_2 = torch.tensor([3, 5], dtype=torch.int64)
_AGES_10_MULTI = np.array(
    [80.0, 200.0, 350.0, 500.0, 650.0, 720.0, 900.0, 1050.0, 1200.0, 1400.0]
)

# ---------------------------------------------------------------------------
# Multi-epoch golden values — computed 2026-06-15 from the initial implementation
# ---------------------------------------------------------------------------
# Spine age log-lik at age=300, 2-epoch
_GOLDEN_SPINE_LL_300_MULTI = -6.9630921864
# Non-spine age log-lik at age=300, 2-epoch
_GOLDEN_NON_SPINE_LL_300_MULTI = -7.6053693664
# Population log-lik at ns=[3,5], 2-epoch
_GOLDEN_POPULATIONS_LL_NS_2_MULTI = -18.6682576565
# Combined log-likelihood (unknown-spine marginalisation) for 10 ages, 2-epoch.
# Updated 2026-06-18: fixed calculate_log_lik_initial_lifetimes_given_extinction
# which previously called calculate_log_pr_extinction_given_initial_lifetimes
# without as_array=True, so the return_array=True path (used inside
# log_likelihood) received a scalar SUM instead of per-lifetime values.
_GOLDEN_COMBINED_LL_10AGES_MULTI = -177.7867323299


@pytest.mark.regression
class TestRegressionLikelihoods:
    """Golden-value regression tests for core likelihood functions."""

    def test_extinction_pr_golden_supercritical(self) -> None:
        result = calculate_extinction_pr(_BIRTH, _DEATH, _TIMES)
        np.testing.assert_allclose(
            float(result), _GOLDEN_EXTINCTION_SUPERCRITICAL, atol=_ATOL, rtol=0.0
        )

    def test_extinction_pr_golden_subcritical(self) -> None:
        lam = torch.tensor([0.03], dtype=torch.float64)
        mu = torch.tensor([0.05], dtype=torch.float64)
        result = calculate_extinction_pr(lam, mu, _TIMES)
        np.testing.assert_allclose(
            float(result), _GOLDEN_EXTINCTION_SUBCRITICAL, atol=_ATOL, rtol=0.0
        )

    def test_extinction_pr_golden_critical(self) -> None:
        lam = torch.tensor([0.05], dtype=torch.float64)
        mu = torch.tensor([0.05], dtype=torch.float64)
        result = calculate_extinction_pr(lam, mu, _TIMES)
        np.testing.assert_allclose(
            float(result), _GOLDEN_EXTINCTION_CRITICAL, atol=_ATOL, rtol=0.0
        )

    def test_spine_ages_log_lik_golden_single_age(self) -> None:
        result = calculate_spine_ages_log_lik(_BIRTH, _DEATH, _AGE_750, _TIMES)
        np.testing.assert_allclose(
            float(result), _GOLDEN_SPINE_LL_750, atol=_ATOL, rtol=0.0
        )

    def test_non_spine_ages_log_lik_golden_single_age(self) -> None:
        result = calculate_non_spine_ages_log_lik(_BIRTH, _DEATH, _AGE_750, _TIMES)
        np.testing.assert_allclose(
            float(result), _GOLDEN_NON_SPINE_LL_750, atol=_ATOL, rtol=0.0
        )

    def test_populations_log_lik_golden_n_equals_5(self) -> None:
        result = calculate_populations_log_lik(
            _BIRTH, _DEATH, _NS_5, _TIMES, initial_population=1, unbias=False
        )
        np.testing.assert_allclose(
            float(result), _GOLDEN_POPULATIONS_LL_N5, atol=_ATOL, rtol=0.0
        )

    def test_combined_log_likelihood_golden_ten_ages(self) -> None:
        result = log_likelihood(
            birth_rates=_BIRTH,
            death_rates=_DEATH,
            ages=_AGES_10,
            times=1500.0,
        )
        assert abs(float(result) - _GOLDEN_COMBINED_LL_10AGES) < _ATOL

    # ------------------------------------------------------------------
    # Multi-epoch golden tests (two-epoch parameter set)
    # These are the reference numbers that Phase 2 (NumPyro/JAX) must
    # match within _ATOL before that backend is trusted.
    # ------------------------------------------------------------------

    def test_spine_ages_log_lik_golden_single_age_multi_epoch(self) -> None:
        result = calculate_spine_ages_log_lik(_BIRTH_2, _DEATH_2, _AGE_300, _TIMES_2)
        np.testing.assert_allclose(
            float(result), _GOLDEN_SPINE_LL_300_MULTI, atol=_ATOL, rtol=0.0
        )

    def test_non_spine_ages_log_lik_golden_single_age_multi_epoch(self) -> None:
        result = calculate_non_spine_ages_log_lik(
            _BIRTH_2, _DEATH_2, _AGE_300, _TIMES_2
        )
        np.testing.assert_allclose(
            float(result), _GOLDEN_NON_SPINE_LL_300_MULTI, atol=_ATOL, rtol=0.0
        )

    def test_populations_log_lik_golden_ns_2_multi_epoch(self) -> None:
        result = calculate_populations_log_lik(
            _BIRTH_2, _DEATH_2, _NS_2, _TIMES_2, initial_population=1, unbias=False
        )
        np.testing.assert_allclose(
            float(result), _GOLDEN_POPULATIONS_LL_NS_2_MULTI, atol=_ATOL, rtol=0.0
        )

    def test_combined_log_likelihood_golden_ten_ages_multi_epoch(self) -> None:
        result = log_likelihood(
            birth_rates=_BIRTH_2,
            death_rates=_DEATH_2,
            ages=_AGES_10_MULTI,
            times=_TIMES_2,
        )
        assert abs(float(result) - _GOLDEN_COMBINED_LL_10AGES_MULTI) < _ATOL
