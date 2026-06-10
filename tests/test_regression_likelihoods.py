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
from torch.testing import assert_close

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


@pytest.mark.regression
class TestRegressionLikelihoods:
    """Golden-value regression tests for core likelihood functions."""

    def test_extinction_pr_golden_supercritical(self) -> None:
        result = calculate_extinction_pr(_BIRTH, _DEATH, _TIMES)
        assert_close(
            result,
            torch.tensor(_GOLDEN_EXTINCTION_SUPERCRITICAL, dtype=torch.float64),
            atol=_ATOL,
            rtol=0.0,
        )

    def test_extinction_pr_golden_subcritical(self) -> None:
        lam = torch.tensor([0.03], dtype=torch.float64)
        mu = torch.tensor([0.05], dtype=torch.float64)
        result = calculate_extinction_pr(lam, mu, _TIMES)
        assert_close(
            result,
            torch.tensor(_GOLDEN_EXTINCTION_SUBCRITICAL, dtype=torch.float64),
            atol=_ATOL,
            rtol=0.0,
        )

    def test_extinction_pr_golden_critical(self) -> None:
        lam = torch.tensor([0.05], dtype=torch.float64)
        mu = torch.tensor([0.05], dtype=torch.float64)
        result = calculate_extinction_pr(lam, mu, _TIMES)
        assert_close(
            result,
            torch.tensor(_GOLDEN_EXTINCTION_CRITICAL, dtype=torch.float64),
            atol=_ATOL,
            rtol=0.0,
        )

    def test_spine_ages_log_lik_golden_single_age(self) -> None:
        result = calculate_spine_ages_log_lik(_BIRTH, _DEATH, _AGE_750, _TIMES)
        assert_close(
            result,
            torch.tensor(_GOLDEN_SPINE_LL_750, dtype=torch.float64),
            atol=_ATOL,
            rtol=0.0,
        )

    def test_non_spine_ages_log_lik_golden_single_age(self) -> None:
        result = calculate_non_spine_ages_log_lik(_BIRTH, _DEATH, _AGE_750, _TIMES)
        assert_close(
            result,
            torch.tensor(_GOLDEN_NON_SPINE_LL_750, dtype=torch.float64),
            atol=_ATOL,
            rtol=0.0,
        )

    def test_populations_log_lik_golden_n_equals_5(self) -> None:
        result = calculate_populations_log_lik(
            _BIRTH, _DEATH, _NS_5, _TIMES, initial_population=1, unbias=False
        )
        assert_close(
            result,
            torch.tensor(_GOLDEN_POPULATIONS_LL_N5, dtype=torch.float64),
            atol=_ATOL,
            rtol=0.0,
        )

    def test_combined_log_likelihood_golden_ten_ages(self) -> None:
        result = log_likelihood(
            birth_rates=_BIRTH,
            death_rates=_DEATH,
            ages=_AGES_10,
            times=1500.0,
        )
        assert abs(float(result) - _GOLDEN_COMBINED_LL_10AGES) < _ATOL
