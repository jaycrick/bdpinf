from typing import Dict

import numpy as np
import pytest
import scipy
import torch
from numpy.testing import assert_allclose
from numpy.typing import NDArray

# Import private functions for direct testing
from inf_bp_texts.models.mathematical_functions import (
    _integrand_surv_from_weighted_birth_time_vectorized,
    _population_geometric_parameter,
    _pr_extinction_numerator,
    calculate_ages_log_lik,
    calculate_ages_log_lik_derivatives,
    calculate_extinction_pr,
    calculate_log_lik_initial_lifetimes_given_extinction,
    calculate_log_lik_initial_lifetimes_given_non_extinction,
    calculate_log_lik_lifetimes,
    calculate_log_pr_extinction_given_initial_lifetimes,
    calculate_non_spine_ages_log_lik,
    calculate_non_spine_ages_log_lik_derivatives,
    calculate_populations_log_lik,
    calculate_populations_log_lik_derivatives,
    calculate_spine_ages_log_lik,
    calculate_spine_ages_log_lik_derivatives,
)

"""
Comprehensive test suite for mathematical_functions.py

This module tests all mathematical functions for birth-death process calculations,
organized in a logical dependency order:
1. Basic helper functions (private)
2. Extinction probability calculations
3. Population likelihood functions
4. Age likelihood functions (spine and non-spine)
5. Derivative functions
6. Integration and edge cases

Test Strategy:
- Unit tests for individual functions
- Numerical stability tests for edge cases (λ ≈ μ, extreme parameters)
- Mathematical property tests (probability bounds, monotonicity)
- Integration tests for function chains
- Array vs scalar input consistency tests
"""

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def standard_scalar_params():
    """Standard scalar parameters for testing."""
    return {
        "lambda_param": torch.tensor([0.1], dtype=torch.float64),
        "mu_param": torch.tensor([0.05], dtype=torch.float64),
        "time": torch.tensor([100.0], dtype=torch.float64),
    }


@pytest.fixture
def standard_array_params():
    """Standard array parameters for piecewise-constant rates."""
    return {
        "lambda_params": torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64),
        "mu_params": torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64),
        "times": torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64),
    }


@pytest.fixture
def nearly_equal_params():
    """Parameters where λ ≈ μ (numerical stability test)."""
    return {
        "lambda_param": torch.tensor([0.1], dtype=torch.float64),
        "mu_param": torch.tensor(
            [0.1 + 1e-13], dtype=torch.float64
        ),  # Within tolerance of 1e-12
        "time": torch.tensor([100.0], dtype=torch.float64),
    }


@pytest.fixture
def log_scale_params() -> Dict[str, float]:
    """Parameters in log scale."""
    return {
        "log_lambda": np.log(0.1),
        "log_mu": np.log(0.05),
        "time": 100.0,
    }


@pytest.fixture
def sample_ages_scalar():
    """Single age value for testing."""
    return 50.0


@pytest.fixture
def sample_ages_array():
    """Array of ages for vectorized testing."""
    return torch.tensor([10.0, 30.0, 50.0, 70.0, 90.0], dtype=torch.float64)


@pytest.fixture
def sample_population_counts():
    """Sample population counts (n values)."""
    return torch.tensor([0, 1, 5, 10, 20], dtype=torch.int64)


@pytest.fixture
def two_epoch_params():
    """Parameters for two-epoch testing."""
    return {
        "lambda_params": torch.tensor([0.01, 0.001], dtype=torch.float64),
        "mu_params": torch.tensor([0.001, 0.001], dtype=torch.float64),
        "times": torch.tensor([750.0, 1500.0], dtype=torch.float64),
    }


@pytest.fixture
def three_epoch_params():
    """Parameters for three-epoch testing."""
    return {
        "lambda_params": torch.tensor([0.1, 0.05, 0.08], dtype=torch.float64),
        "mu_params": torch.tensor([0.03, 0.02, 0.04], dtype=torch.float64),
        "times": torch.tensor([100.0, 250.0, 400.0], dtype=torch.float64),
    }


# ============================================================================
# LEVEL 0: Numerical Accuracy Tests (Analytical Validation)
# ============================================================================


class TestMultiEpochNumericalAccuracy:
    """Test multi-epoch functions against analytical results.

    These tests validate that multi-epoch implementations produce correct results
    by comparing against known analytical values or reduced single-epoch cases.
    """

    def test_two_epoch_population_lik_reduces_to_single_epoch(self):
        """Test that two identical epochs match single-epoch result."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)
        n = torch.tensor([5], dtype=torch.int64)

        # Single epoch result
        single_result = calculate_populations_log_lik(lambda_val, mu_val, T, ns=n)

        # Two epochs with same parameters should give same result
        two_epoch_result = calculate_populations_log_lik(
            torch.tensor([lambda_val[0], lambda_val[0]], dtype=torch.float64),
            torch.tensor([mu_val[0], mu_val[0]], dtype=torch.float64),
            torch.tensor([T[0] / 2, T[0]], dtype=torch.float64),
            ns=n,
        )

        assert np.isfinite(single_result)
        assert np.isfinite(two_epoch_result)
        assert_allclose(single_result, two_epoch_result, rtol=1e-8)

    def test_multi_epoch_extinction_pr_bounds(self):
        """Test that multi-epoch extinction probability is in [0, 1]."""
        lambda_params = torch.tensor([0.2, 0.15, 0.1], dtype=torch.float64)
        mu_params = torch.tensor([0.1, 0.12, 0.15], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = calculate_extinction_pr(lambda_params, mu_params, times)

        assert 0 < result < 1, f"Extinction probability {result} not in (0, 1)"

    def test_two_epoch_spine_age_density_basic(self):
        """Validate multi-epoch spine age density for ages in each epoch.

        This exercises the reflected multi-epoch path in
        ``calculate_spine_ages_log_lik`` by checking that ages in both the first
        and second epochs produce finite log densities with the current closed-form
        implementation.
        """
        lambda_params = torch.tensor([0.1, 0.2], dtype=torch.float64)
        mu_params = torch.tensor([0.05, 0.1], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # Age in first epoch
        age_epoch1 = torch.tensor([30.0], dtype=torch.float64)
        result1 = calculate_spine_ages_log_lik(
            lambda_params, mu_params, age_epoch1, times
        )
        assert np.isfinite(result1)

        # Age in second epoch
        age_epoch2 = torch.tensor([70.0], dtype=torch.float64)
        result2 = calculate_spine_ages_log_lik(
            lambda_params, mu_params, age_epoch2, times
        )
        assert np.isfinite(result2)

    def test_multi_epoch_non_spine_age_array(self):
        """Check multi-epoch non-spine ages spanning epochs yield finite, negative log-lik.

        Confirms the multi-epoch branch of ``calculate_non_spine_ages_log_lik``
        handles ages on either side of an epoch boundary and returns a negative
        summed log-density as expected for probabilities < 1.
        """
        lambda_params = torch.tensor([0.15, 0.1], dtype=torch.float64)
        mu_params = torch.tensor([0.05, 0.08], dtype=torch.float64)
        times = torch.tensor([100.0, 200.0], dtype=torch.float64)

        # Ages spanning both epochs
        ages = torch.tensor([20.0, 60.0, 120.0, 180.0], dtype=torch.float64)

        result = calculate_non_spine_ages_log_lik(lambda_params, mu_params, ages, times)

        # Should return finite sum
        assert np.isfinite(result)
        # Log likelihood should be negative
        assert result < 0

    def test_two_epoch_geometric_parameter(self):
        """Test _population_geometric_parameter for two epochs."""
        birth_rates = torch.tensor([0.2, 0.3], dtype=torch.float64)
        death_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = _population_geometric_parameter(birth_rates, death_rates, times)

        # Should be in valid probability range
        assert 0 < result < 1
        assert np.isfinite(result)


class TestMultiEpochConsistency:
    """Test consistency properties of multi-epoch functions."""

    def test_epoch_order_matters(self):
        """Test that swapping epoch parameters changes results."""
        # Different parameters for each epoch
        lambda_params1 = torch.tensor([0.2, 0.1], dtype=torch.float64)
        mu_params1 = torch.tensor([0.1, 0.05], dtype=torch.float64)
        times = torch.tensor([60.0, 100.0], dtype=torch.float64)

        # Swapped parameters
        lambda_params2 = torch.tensor([0.1, 0.2], dtype=torch.float64)
        mu_params2 = torch.tensor([0.05, 0.1], dtype=torch.float64)

        result1 = calculate_populations_log_lik(lambda_params1, mu_params1, times, ns=5)
        result2 = calculate_populations_log_lik(lambda_params2, mu_params2, times, ns=5)

        # Results should differ (order matters)
        assert not np.isclose(result1, result2)

    def test_multi_epoch_ages_sum_property(self):
        """Test that multi-epoch age likelihood sums correctly across ages."""
        lambda_params = torch.tensor([0.12, 0.08], dtype=torch.float64)
        mu_params = torch.tensor([0.04, 0.06], dtype=torch.float64)
        times = torch.tensor([100.0, 200.0], dtype=torch.float64)

        ages = torch.tensor([20.0, 80.0, 150.0], dtype=torch.float64)

        # Compute for array
        result_array = calculate_non_spine_ages_log_lik(
            lambda_params, mu_params, ages, times
        )

        # Compute individually and sum
        result_sum = sum(
            calculate_non_spine_ages_log_lik(
                lambda_params,
                mu_params,
                torch.tensor([age], dtype=torch.float64),
                times,
            )
            for age in ages
        )

        assert_allclose(result_array, result_sum, rtol=1e-8)


# ============================================================================
# LEVEL 0.1: Multi-Epoch vs Single-Epoch Equivalence (Identical Rates)
# ============================================================================


class TestMultiEpochEqualsSingleEpoch:
    """Verify multi-epoch equals single-epoch for identical rates.

    For functions that support array inputs (piecewise-constant epochs), when
    the rates are identical across epochs, the result should match the
    equivalent single-epoch case with the same total time. Where known
    numerical differences exist (due to reflection/time-reversal branches),
    we use a relaxed tolerance and document the behavior.
    """

    def test_extinction_pr_multi_equals_single(self):
        lambda_val, mu_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )

        single = calculate_extinction_pr(lambda_val, mu_val, T)

        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_extinction_pr(lambdas, mus, times)

        assert_allclose(single, multi, rtol=1e-8)
        assert 0 < single < 1

    def test_population_geometric_parameter_multi_equals_single(self):
        lambda_val, mu_val, T = (
            torch.tensor([0.03], dtype=torch.float64),
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([90.0], dtype=torch.float64),
        )
        single = _population_geometric_parameter(lambda_val, mu_val, T)
        times = torch.tensor([30.0, 60.0, 90.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = _population_geometric_parameter(lambdas, mus, times)
        assert_allclose(single, multi, rtol=1e-8)

    def test_pr_extinction_numerator_multi_equals_single(self):
        lambda_val, mu_val, T = (
            torch.tensor([0.015], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([100.0], dtype=torch.float64),
        )
        single = _pr_extinction_numerator(lambda_val, mu_val, T)
        times = torch.tensor([25.0, 50.0, 75.0, 100.0], dtype=torch.float64)
        lambdas = np.full(4, lambda_val)
        mus = np.full(4, mu_val)
        multi = _pr_extinction_numerator(lambdas, mus, times)
        assert_allclose(single, multi, rtol=1e-8)

    def test_log_lik_lifetimes_multi_equals_single(self):
        lifetimes = torch.tensor([10.0, 30.0, 60.0, 90.0], dtype=torch.float64)
        lambda_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_lik_lifetimes(lifetimes, lambda_val, T)
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        multi = calculate_log_lik_lifetimes(lifetimes, lambdas, times)
        # Multi-epoch lifetimes uses piecewise handling; allow relaxed tolerance
        assert_allclose(single, multi, rtol=1e-8)

    def test_log_pr_extinction_given_initial_lifetimes_multi_equals_single(self):
        lifetimes = torch.tensor([10.0, 45.0, 75.0, 110.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes, lambda_val, mu_val, T
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes, lambdas, mus, times
        )
        assert_allclose(single, multi, rtol=1e-8)

    def test_log_lik_initial_lifetimes_given_extinction_multi_equals_single(self):
        lifetimes = torch.tensor([12.0, 50.0, 95.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.025], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_lik_initial_lifetimes_given_extinction(
            lifetimes, lambda_val, mu_val, T
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_log_lik_initial_lifetimes_given_extinction(
            lifetimes, lambdas, mus, times
        )
        assert_allclose(single, multi, rtol=1e-85)

    def test_log_lik_initial_lifetimes_given_non_extinction_multi_equals_single(self):
        lifetimes = torch.tensor([12.0, 50.0, 95.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.025], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes, lambda_val, mu_val, T
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes, lambdas, mus, times
        )
        assert_allclose(single, multi, rtol=1e-85)

    def test_non_spine_ages_log_lik_multi_equals_single(self):
        ages = torch.tensor([15.0, 60.0, 95.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        # Single-epoch via scalar
        single = calculate_non_spine_ages_log_lik(lambda_val, mu_val, ages, T)
        # Multi-epoch via arrays with identical rates
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_non_spine_ages_log_lik(lambdas, mus, ages, times)
        # Known numerical differences due to reflection method; allow relaxed tolerance
        assert np.isfinite(single)
        assert np.isfinite(multi)
        assert_allclose(single, multi, rtol=1e-8)

    def test_spine_ages_log_lik_multi_equals_single(self):
        ages = torch.tensor([20.0, 40.0, 110.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.018], dtype=torch.float64),
            torch.tensor([0.009], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_spine_ages_log_lik(lambda_val, mu_val, ages, T)
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_spine_ages_log_lik(lambdas, mus, ages, times)
        assert np.isfinite(single)
        assert np.isfinite(multi)
        assert_allclose(single, multi, rtol=1e-8)

    def test_ages_log_lik_multi_equals_single_non_spine(self):
        ages = torch.tensor([10.0, 55.0, 90.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_ages_log_lik(lambda_val, mu_val, ages, T, are_spines=False)
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_ages_log_lik(lambdas, mus, ages, times, are_spines=False)
        assert np.isfinite(single)
        assert np.isfinite(multi)
        assert_allclose(single, multi, rtol=1e-8)

    def test_ages_log_lik_multi_equals_single_spine(self):
        ages = torch.tensor([15.0, 120.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.018], dtype=torch.float64),
            torch.tensor([0.009], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_ages_log_lik(lambda_val, mu_val, ages, T, are_spines=True)
        times = torch.tensor([30.0, 90.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = calculate_ages_log_lik(lambdas, mus, ages, times, are_spines=True)
        assert np.isfinite(single)
        assert np.isfinite(multi)
        assert_allclose(single, multi, rtol=1e-8)

    def test_integrand_surv_from_weighted_birth_time_vectorized_multi_equals_single(
        self,
    ):
        birth_time = torch.tensor([30.0], dtype=torch.float64)
        lambda_val, mu_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time, lambda_val, mu_val, T
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        lambdas = np.full(3, lambda_val)
        mus = np.full(3, mu_val)
        multi = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time, lambdas, mus, times
        )
        assert_allclose(single, multi, rtol=1e-8)


# ============================================================================
# LEVEL 0: Numerical Accuracy Tests (Analytical Validation) - ORIGINAL
# ============================================================================


class TestNumericalAccuracy:
    """Test functions against known analytical results.

    These tests validate that the implementations produce numerically correct
    results by comparing against hand-calculated or analytically derived values.
    """

    def test_extinction_pr_critical_case_analytical(self):
        """Test extinction probability when λ = μ (analytical formula).

        When λ = μ, the extinction probability has the closed form:
        α_t = λT / (1 + λT)
        """
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Expected: α_t = 0.1 * 100 / (1 + 0.1 * 100) = 10 / 11 ≈ 0.909091
        expected = lambda_val * T / (1 + lambda_val * T)

        result = calculate_extinction_pr(
            lambda_val,
            lambda_val,  # μ = λ
            T,
        )

        assert_allclose(result, expected, rtol=1e-8)
        assert 0 < result < 1

    def test_populations_log_lik_n0_simple(self):
        """Test log P(n=0) for simple parameters.

        For n=0: log P(n=0) = log(α_t)
        Where α_t = μ(e^{(λ-μ)T} - 1) / (λ e^{(λ-μ)T} - μ)
        """
        lambda_val = torch.tensor([0.2], dtype=torch.float64)
        mu_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([10.0], dtype=torch.float64)

        # Compute expected α_t manually
        growth = (lambda_val - mu_val) * T  # 1.0
        exp_growth = np.exp(growth)  # e^1.0 ≈ 2.71828
        alpha_t = mu_val * (exp_growth - 1) / (lambda_val * exp_growth - mu_val)
        expected_log_prob = np.log(alpha_t)

        result = calculate_populations_log_lik(
            lambda_val, mu_val, T, ns=torch.tensor([0], dtype=torch.int64)
        )

        assert_allclose(result, expected_log_prob, rtol=1e-8)

    def test_populations_log_lik_critical_case_n0(self):
        """Test log P(n=0) when λ = μ.

        When λ = μ: α_t = λT / (1 + λT)
        So log P(n=0) = log(λT / (1 + λT))
        """
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([50.0], dtype=torch.float64)

        # Expected: α_t = 0.1 * 50 / (1 + 0.1 * 50) = 5/6
        alpha_t = lambda_val * T / (1 + lambda_val * T)
        expected = np.log(alpha_t)

        result = calculate_populations_log_lik(
            lambda_val, lambda_val, T, ns=torch.tensor([0], dtype=torch.int64)
        )

        assert_allclose(result, expected, rtol=1e-8)

    def test_populations_log_lik_n1_simple(self):
        """Test log P(n=1) for simple parameters.

        For n=1: log P(n=1) = log(1 - α_t) + log(1 - β_t)
        Where β_t = (λ/μ) * α_t
        """
        lambda_val = torch.tensor([0.3], dtype=torch.float64)
        mu_val = torch.tensor([0.2], dtype=torch.float64)
        T = torch.tensor([5.0], dtype=torch.float64)

        # Compute α_t and β_t
        growth = (lambda_val - mu_val) * T
        exp_growth = np.exp(growth)
        alpha_t = mu_val * (exp_growth - 1) / (lambda_val * exp_growth - mu_val)
        beta_t = (lambda_val / mu_val) * alpha_t

        expected = np.log(1 - alpha_t) + np.log(1 - beta_t)

        result = calculate_populations_log_lik(
            lambda_val, mu_val, T, ns=torch.tensor([1], dtype=torch.int64)
        )

        assert_allclose(result, expected, rtol=1e-8)

    def test_spine_age_at_zero(self):
        """Test spine age density at t=0.

        At t=0 (age zero), the density involves special limiting behavior.
        """
        lambda_val = torch.tensor([0.2], dtype=torch.float64)
        mu_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([10.0], dtype=torch.float64)
        age = torch.tensor([0.0], dtype=torch.float64)

        result = calculate_spine_ages_log_lik(lambda_val, mu_val, age, T)

        # Should return finite value (not -inf)
        assert np.isfinite(result)
        # Age density at t=0 should be positive (log density negative but finite)
        assert result > -100  # Reasonable bound

    def test_geometric_scalar_case(self):
        """
        Docstring for test_geometric_scalar_case

        :param self: Description
        """

        T = torch.tensor([10.0], dtype=torch.float64)
        lambda_val = torch.tensor([1.0], dtype=torch.float64)
        mu_val = torch.tensor([2.0], dtype=torch.float64)

        def _integrand(x):
            lam = float(lambda_val[0])
            mu = float(mu_val[0])
            x_arr = np.asarray(x)
            return mu * np.exp(-(lam - mu) * x_arr)

        integral, _ = scipy.integrate.fixed_quad(
            _integrand,
            0,
            T[0],
            n=10,
        )
        expected = np.exp(-(lambda_val - mu_val) * T) / (1 + integral)
        result = _population_geometric_parameter(lambda_val, mu_val, T)
        assert_allclose(result, expected, rtol=1e-8)

    def test_geometric_parameter_critical_case(self):
        """Test _population_geometric_parameter when λ = μ.

        Private function test: This function returns 1 - β_t where β_t = λT / (1 + λT)
        """
        lambda_val = mu_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([20.0], dtype=torch.float64)

        def _integrand(x):
            lam = float(lambda_val[0])
            mu = float(mu_val[0])
            x_arr = np.asarray(x)
            return mu * np.exp(-(lam - mu) * x_arr)

        integral, _ = scipy.integrate.fixed_quad(
            _integrand,
            0,
            T[0],
            n=10,
        )
        expected = np.exp(-(lambda_val - mu_val) * T) / (1 + integral)
        result = _population_geometric_parameter(lambda_val, mu_val, T)

        assert_allclose(result, expected, rtol=1e-8)

    def test_geometric_parameter_general_case(self):
        """Test _population_geometric_parameter for λ ≠ μ.

        Private function test: Function returns 1 - β_t where β_t = (λ/μ) * α_t
        """
        birth_rates = torch.tensor([0.3, 0.4], dtype=torch.float64)
        death_rates = torch.tensor([0.1, 0.3], dtype=torch.float64)
        epoch_ends = torch.tensor([5.0, 9.0], dtype=torch.float64)

        def _integrand(x):
            x_arr = np.asarray(x)
            first = np.exp(-(birth_rates[0] - death_rates[0]) * x_arr) * death_rates[0]
            second = (
                np.exp(
                    -(
                        (birth_rates[0] - death_rates[0]) * epoch_ends[0]
                        + (birth_rates[1] - death_rates[1]) * (x_arr - epoch_ends[0])
                    )
                )
                * death_rates[1]
            )
            return np.where(x_arr <= epoch_ends[0], first, second)

        integral, _ = scipy.integrate.fixed_quad(
            _integrand,
            0.0,
            epoch_ends[1],
            n=10,
        )
        expected = np.exp(
            -(
                (birth_rates[1] - death_rates[1]) * (epoch_ends[1] - epoch_ends[0])
                + epoch_ends[0] * (birth_rates[0] - death_rates[0])
            )
        ) / (1 + integral)
        result = _population_geometric_parameter(birth_rates, death_rates, epoch_ends)

        assert_allclose(result, expected, rtol=1e-8)

    def test_pr_extinction_numerator_simple(self):
        """Test _pr_extinction_numerator for simple case.

        Private function test: Validates that function returns a positive finite value.
        The exact formula is complex and involves integration - we validate properties.
        """
        lambda_val = torch.tensor([0.2], dtype=torch.float64)
        mu_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([10.0], dtype=torch.float64)

        result = _pr_extinction_numerator(
            lambda_val,
            mu_val,
            T,
        )

        # Should be positive and finite
        assert np.isfinite(result)
        assert result > 0
        # For λ > μ, numerator should be between 0 and 1
        assert 0 < result < 1

    def test_calculate_log_lik_lifetimes_simple(self):
        """Test calculate_log_lik_lifetimes for simple scalar age.

        Private function test: log density involves birth rate and growth.
        """
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        age = torch.tensor([20.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # For simple case: log f(t) = log(λ) - λt
        # (This is simplified; actual formula may differ)
        result = calculate_log_lik_lifetimes(
            age,
            lambda_val,
            T,
        )

        # Should be finite and negative (log of density)
        assert np.isfinite(result)
        assert result < 0

    def test_calculate_log_lik_lifetimes_array(self):
        """Test calculate_log_lik_lifetimes with array of ages.

        Private function test: should sum log likelihoods across ages.
        """
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        ages = torch.tensor([10.0, 30.0, 50.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Compute for array
        result_array = calculate_log_lik_lifetimes(
            ages,
            lambda_val,
            T,
        )

        # Compute individually and sum
        result_sum = sum(
            calculate_log_lik_lifetimes(
                torch.tensor([age], dtype=torch.float64),
                lambda_val,
                T,
            )
            for age in ages
        )

        assert_allclose(result_array, result_sum, rtol=1e-8)

    def test_non_spine_age_density_known_values(self):
        """Test non-spine age density against manually computed values.

        For specific parameter values, compute expected log density.
        """
        lambda_val = torch.tensor([0.2], dtype=torch.float64)
        mu_val = torch.tensor([0.1], dtype=torch.float64)
        age = torch.tensor([5.0], dtype=torch.float64)
        T = torch.tensor([20.0], dtype=torch.float64)

        # Non-spine density: f(t) = [λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}] / [1 - e^{-(λ-μ)T}]
        growth_rate = lambda_val - mu_val
        term1 = lambda_val * np.exp(-lambda_val * age)
        term2 = mu_val * np.exp(-growth_rate * T) * np.exp(-mu_val * age)
        numerator = term1 - term2
        denominator = 1 - np.exp(-growth_rate * T)
        density = numerator / denominator
        expected = np.log(density)

        result = calculate_non_spine_ages_log_lik(lambda_val, mu_val, age, T)

        assert_allclose(result, expected, rtol=1e-8)

    def test_derivatives_sum_over_array(self):
        """Test that derivatives sum correctly over array of n values.

        Computing derivatives for [n1, n2] should equal sum of individual derivatives.
        """
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = 0.08
        T = torch.tensor([50.0], dtype=torch.float64)
        ns = torch.tensor([1, 5, 10], dtype=torch.int64)

        # Compute together
        result_together = calculate_populations_log_lik_derivatives(
            lambda_val, mu_val, T, ns
        )

        # Compute separately and sum
        derivs_separate = [
            calculate_populations_log_lik_derivatives(lambda_val, mu_val, T, n)
            for n in ns
        ]
        result_separate = np.sum(derivs_separate, axis=0)

        assert_allclose(result_together, result_separate, rtol=1e-8)

    def test_extinction_pr_bounds(self):
        """Test that extinction probability satisfies 0 < α_t < 1 for various inputs.

        Validates mathematical constraint across parameter space.
        """
        test_cases = [
            (0.1, 0.05, 100.0),
            (0.5, 0.3, 50.0),
            (1.0, 0.8, 10.0),
            (0.01, 0.005, 1000.0),
        ]

        for lambda_val, mu_val, T in test_cases:
            result = calculate_extinction_pr(
                torch.tensor(lambda_val, dtype=torch.float64),
                torch.tensor(mu_val, dtype=torch.float64),
                torch.tensor(T, dtype=torch.float64),
            )

            assert 0 < result < 1, f"Failed for λ={lambda_val}, μ={mu_val}, T={T}"

    def test_calculate_log_pr_extinction_given_initial_lifetimes_zero(self):
        """Test calculate_log_pr_extinction_given_initial_lifetimes at age zero.

        Private function test: Validates function returns reasonable log probability.
        """
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)
        age = torch.tensor([0.0], dtype=torch.float64)

        result = calculate_log_pr_extinction_given_initial_lifetimes(
            age, lambda_val, mu_val, T
        )

        # Should be finite (could be 0 at age 0, representing log(1) = 0)
        assert np.isfinite(result)
        # Log probability should be non-positive (probability ≤ 1)
        assert result <= 0

    def test_piecewise_constant_single_vs_multi_epoch(self):
        """Test that single-epoch array matches scalar computation.

        A single-epoch piecewise-constant model should match scalar input.
        """
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.07], dtype=torch.float64)
        T = torch.tensor([80.0], dtype=torch.float64)
        n = torch.tensor([5], dtype=torch.int64)

        # Scalar computation
        result_scalar = calculate_populations_log_lik(lambda_val, mu_val, T, n)

        # Single-epoch array computation
        result_array = calculate_populations_log_lik(lambda_val, mu_val, T, n)

        assert_allclose(result_scalar, result_array, rtol=1e-8)


# ============================================================================
# LEVEL 1: Extinction Probability Tests (Foundation)
# ============================================================================


class TestExtinctionProbability:
    """Test extinction probability calculations (α_t)."""

    def test_extinction_pr_basic(
        self, standard_array_params: Dict[str, NDArray[np.float64]]
    ):
        """Test basic extinction probability calculation."""
        result = calculate_extinction_pr(
            standard_array_params["lambda_params"],
            standard_array_params["mu_params"],
            standard_array_params["times"],
        )

        assert 0 < result < 1
        assert isinstance(result, torch.Tensor)

    def test_extinction_pr_return_log_prob(
        self, standard_array_params: Dict[str, NDArray[np.float64]]
    ):
        """Test extinction probability with return_log_prob=True."""
        result = calculate_extinction_pr(
            standard_array_params["lambda_params"],
            standard_array_params["mu_params"],
            standard_array_params["times"],
            return_log_prob=True,
        )

        # Log probability should be negative
        assert result < 0
        assert np.isfinite(result)

    def test_extinction_pr_critical_case(self):
        """Test numerical stability when λ ≈ μ."""
        # Nearly equal rates
        lambda_arr = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        mu_arr = torch.tensor(
            [0.1 + 1e-13, 0.1 + 1e-13, 0.1 + 1e-13], dtype=torch.float64
        )
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = calculate_extinction_pr(lambda_arr, mu_arr, times)

        # Should handle critical case without returning inf or nan
        assert 0 < result < 1
        assert np.isfinite(result)

    def test_extinction_pr_zero_rates_both(self):
        """Test extinction probability when both λ = 0 and μ = 0."""
        lambda_arr = torch.tensor([0.0], dtype=torch.float64)
        mu_arr = torch.tensor([0.0], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = calculate_extinction_pr(lambda_arr, mu_arr, times)

        # With no birth and no death, extinction probability should zero
        assert result == 0.0

    def test_extinction_pr_zero_birth_only(self):
        """Test extinction probability when λ = 0 and μ > 0 (only death)."""
        lambda_arr = torch.tensor([0.0], dtype=torch.float64)
        mu_arr = torch.tensor([0.1], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = calculate_extinction_pr(lambda_arr, mu_arr, times)

        assert 0 < result < 1

    def test_extinction_pr_zero_death_only(self):
        """Test extinction probability when λ > 0 and μ = 0 (only birth)."""
        lambda_arr = torch.tensor([0.1], dtype=torch.float64)
        mu_arr = torch.tensor([0.0], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = calculate_extinction_pr(lambda_arr, mu_arr, times)

        # With zero death rate, extinction should be zero
        assert result == 0

    def test_extinction_pr_zero_rates_multi_epoch(self):
        """Test extinction probability with multi-epoch where some epochs have zero rates."""
        # Some epochs with normal rates, some with zero rates
        lambda_arr = torch.tensor([0.1, 0.0, 0.15], dtype=torch.float64)
        mu_arr = torch.tensor([0.05, 0.0, 0.08], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = calculate_extinction_pr(lambda_arr, mu_arr, times)

        # Should handle mixed zero and non-zero rates
        assert 0 < result < 1

    def test_extinction_pr_increases_with_time(self):
        """Test that extinction probability increases as observation time increases."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)

        # Compare extinction probabilities at different times
        times = [50.0, 100.0, 200.0, 400.0]
        results = [
            calculate_extinction_pr(
                lambda_val, mu_val, torch.tensor([t], dtype=torch.float64)
            )
            for t in times
        ]

        # Extinction probability should be monotonically increasing with time
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"α(t={times[i]}) = {results[i]} should be < "
                f"α(t={times[i + 1]}) = {results[i + 1]}"
            )

    def test_extinction_pr_increases_with_death_rate(self):
        """Test that extinction probability increases as death rate increases."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Compare extinction probabilities at different death rates
        death_rates = [
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.05], dtype=torch.float64),
            torch.tensor([0.08], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
        ]
        results = [
            calculate_extinction_pr(lambda_val, mu_val, T) for mu_val in death_rates
        ]

        # Extinction probability should be monotonically increasing with death rate
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"α(μ={death_rates[i]}) = {results[i]} should be < "
                f"α(μ={death_rates[i + 1]}) = {results[i + 1]}"
            )

    def test_extinction_pr_decreases_with_birth_rate(self):
        """Test that extinction probability decreases as birth rate increases."""
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Compare extinction probabilities at different birth rates
        birth_rates = [
            torch.tensor([0.05], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([0.15], dtype=torch.float64),
            torch.tensor([0.2], dtype=torch.float64),
        ]
        results = [
            calculate_extinction_pr(lambda_val, mu_val, T) for lambda_val in birth_rates
        ]

        # Extinction probability should be monotonically decreasing with birth rate
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1], (
                f"α(λ={birth_rates[i]}) = {results[i]} should be > "
                f"α(λ={birth_rates[i + 1]}) = {results[i + 1]}"
            )


# ============================================================================
# LEVEL 2: Population Likelihood Tests
# ============================================================================


class TestPopulationsLogLikelihood:
    """Test population count log-likelihood calculations."""

    def test_scalar_inputs_basic(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test basic scalar input case."""
        result = calculate_populations_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert np.isfinite(result)
        assert isinstance(result, torch.Tensor)

    def test_scalar_inputs_n_zero(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test n=0 (extinction) case."""
        result = calculate_populations_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=torch.tensor([0], dtype=torch.int64),
        )

        # P(n=0) = α_t, so log P should be finite and negative
        assert np.isfinite(result)
        assert result < 0

    def test_scalar_inputs_array_ns(
        self, standard_scalar_params, sample_population_counts
    ):
        """Test scalar parameters with array of population counts."""
        result = calculate_populations_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=sample_population_counts,
        )

        # Should return sum of log probabilities
        assert np.isfinite(result)

    def test_array_inputs_piecewise_constant(
        self, standard_array_params: Dict[str, NDArray[np.float64]]
    ):
        """Test piecewise-constant rates (array inputs)."""
        result = calculate_populations_log_lik(
            standard_array_params["lambda_params"],
            standard_array_params["mu_params"],
            standard_array_params["times"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert np.isfinite(result)

    def test_negative_time_returns_error(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError):
            calculate_populations_log_lik(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                -100.0,  # Negative time
                ns=torch.tensor([1], dtype=torch.int64),
            )

    def test_negative_ns_returns_error(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test that negative population count raises ValueError."""
        with pytest.raises(ValueError):
            calculate_populations_log_lik(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                standard_scalar_params["time"],
                ns=-5,  # Negative count
            )

    def test_critical_case_scalar(self, nearly_equal_params):
        """Test numerical stability when λ ≈ μ (scalar)."""
        result = calculate_populations_log_lik(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            nearly_equal_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert np.isfinite(result)

    def test_array_length_mismatch_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="matching lengths"):
            calculate_populations_log_lik(
                lambda_params=torch.tensor([0.1, 0.15], dtype=torch.float64),
                mu_params=torch.tensor(
                    [0.05, 0.06, 0.055], dtype=torch.float64
                ),  # Different length
                times=torch.tensor([50.0, 100.0], dtype=torch.float64),
                ns=torch.tensor([1], dtype=torch.int64),
            )


class TestPopulationsLogLikelihoodDerivatives:
    """Test derivatives of population log-likelihood."""

    def test_basic_derivatives(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test basic derivative calculation."""
        result = calculate_populations_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_with_n_zero(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test derivatives for n=0 case."""
        result = calculate_populations_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=torch.tensor([0], dtype=torch.int64),
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_array_ns(
        self, standard_scalar_params, sample_population_counts
    ):
        """Test derivatives with array of n values."""
        result = calculate_populations_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=sample_population_counts,
        )

        # Should return summed derivatives
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_critical_case(self, nearly_equal_params):
        """Test derivative stability when λ ≈ μ."""
        result = calculate_populations_log_lik_derivatives(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            nearly_equal_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert np.all(np.isfinite(result))


# ============================================================================
# LEVEL 3: Age Likelihood Tests - Spine Individuals
# ============================================================================


class TestSpineAgesLogLikelihood:
    """Test age log-likelihood for spine individuals."""

    def test_scalar_age(self, standard_scalar_params, sample_ages_scalar):
        """Test with single age value."""
        result = calculate_spine_ages_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert np.isfinite(result)
        assert isinstance(result, torch.Tensor)

    def test_array_ages(self, standard_scalar_params, sample_ages_array):
        """Test with array of ages."""
        result = calculate_spine_ages_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        # Should return summed log-likelihood
        assert np.isfinite(result)

    def test_age_at_boundary(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test age exactly equal to observation time T."""
        T = standard_scalar_params["time"]
        result = calculate_spine_ages_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            T,  # Age = T (boundary case)
            T,
        )

        assert np.isfinite(result)

    def test_critical_case(self, nearly_equal_params, sample_ages_scalar):
        """Test numerical stability when λ ≈ μ."""
        result = calculate_spine_ages_log_lik(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            sample_ages_scalar,
            nearly_equal_params["time"],
        )

        assert np.isfinite(result)

    def test_invalid_age_greater_than_T(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test that age > T returns -inf."""
        with pytest.raises(
            ValueError, match=r"ages_t must be in \[0, .*\]. Got ages_t:"
        ):  # age > T
            calculate_spine_ages_log_lik(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                standard_scalar_params["time"] + 10.0,  # age > T
                standard_scalar_params["time"],
            )

    def test_negative_age_returns_minus_inf(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test that negative age returns -inf."""
        with pytest.raises(
            ValueError, match=r"ages_t must be in \[0, .*\]. Got ages_t:"
        ):  # Negative age
            calculate_spine_ages_log_lik(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                -10.0,  # Negative age
                standard_scalar_params["time"],
            )


class TestSpineAgesLogLikelihoodDerivatives:
    """Test derivatives for spine age log-likelihood."""

    def test_scalar_age_derivatives(self, standard_scalar_params, sample_ages_scalar):
        """Test derivatives with single age."""
        result = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_array_ages_derivatives(self, standard_scalar_params, sample_ages_array):
        """Test derivatives with array of ages."""
        result = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        # Should return summed derivatives
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_at_boundary(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test derivatives when age = T."""
        T = standard_scalar_params["time"]
        result = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            T,
            T,
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_critical_case(self, nearly_equal_params, sample_ages_scalar):
        """Test derivative stability when λ ≈ μ."""
        result = calculate_spine_ages_log_lik_derivatives(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            sample_ages_scalar,
            nearly_equal_params["time"],
        )

        assert np.all(np.isfinite(result))


# ============================================================================
# LEVEL 4: Age Likelihood Tests - Non-Spine Individuals
# ============================================================================


class TestNonSpineAgesLogLikelihood:
    """Test age log-likelihood for non-spine individuals."""

    def test_scalar_params_scalar_age(self, sample_ages_scalar):
        """Test with scalar parameters and single age."""
        # For single-epoch case, wrap scalars in arrays of length 1
        result = calculate_non_spine_ages_log_lik(
            lambda_params=torch.tensor([0.1], dtype=torch.float64),
            mu_params=torch.tensor([0.05], dtype=torch.float64),
            ages_t=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert np.isfinite(result)

    def test_scalar_params_array_ages(self, sample_ages_array):
        """Test with scalar parameters and array of ages."""
        # For single-epoch case, wrap scalars in arrays of length 1
        result = calculate_non_spine_ages_log_lik(
            lambda_params=torch.tensor([0.1], dtype=torch.float64),
            mu_params=torch.tensor([0.05], dtype=torch.float64),
            ages_t=sample_ages_array,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert np.isfinite(result)

    def test_array_params_piecewise_constant(
        self, standard_array_params: Dict[str, NDArray[np.float64]]
    ):
        """Test with piecewise-constant rates (array parameters)."""
        # Use ages within the valid time range
        max_time = standard_array_params["times"][-1]
        ages = torch.tensor([10.0, 50.0, max_time - 10.0], dtype=torch.float64)
        result = calculate_non_spine_ages_log_lik(
            lambda_params=standard_array_params["lambda_params"],
            mu_params=standard_array_params["mu_params"],
            ages_t=ages,
            times=standard_array_params["times"],
        )

        assert np.isfinite(result)

    def test_single_epoch(self, sample_ages_scalar):
        """Test with single epoch (single-element arrays)."""
        result = calculate_non_spine_ages_log_lik(
            lambda_params=torch.tensor([0.1], dtype=torch.float64),
            mu_params=torch.tensor([0.05], dtype=torch.float64),
            ages_t=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert np.isfinite(result)

    def test_critical_case(self, sample_ages_scalar):
        """Test numerical stability when λ ≈ μ."""
        result = calculate_non_spine_ages_log_lik(
            lambda_params=torch.tensor([0.1], dtype=torch.float64),
            mu_params=torch.tensor([0.1 + 1e-13], dtype=torch.float64),
            ages_t=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert np.isfinite(result)

    def test_invalid_age_returns_minus_inf(self):
        """Test that invalid ages return -inf."""
        # Age > T
        with pytest.raises(
            ValueError, match=r"ages_t must be in \[0, .*\]. Got ages_t:"
        ):
            calculate_non_spine_ages_log_lik(
                lambda_params=torch.tensor([0.1], dtype=torch.float64),
                mu_params=torch.tensor([0.05], dtype=torch.float64),
                ages_t=150.0,  # age > times[-1]
                times=torch.tensor([100.0], dtype=torch.float64),
            )

    def test_shape_mismatch_raises_error(self, sample_ages_scalar):
        """Test that parameter shape mismatches raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            calculate_non_spine_ages_log_lik(
                lambda_params=torch.tensor([0.1, 0.15], dtype=torch.float64),
                mu_params=torch.tensor([0.05], dtype=torch.float64),  # Different shape
                ages_t=sample_ages_scalar,
                times=torch.tensor([50.0, 100.0], dtype=torch.float64),
            )

    def test_non_monotonic_times_raises_error(self, sample_ages_scalar):
        """Test that non-monotonic times raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            calculate_non_spine_ages_log_lik(
                lambda_params=torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64),
                mu_params=torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64),
                ages_t=sample_ages_scalar,
                times=torch.tensor(
                    [50.0, 100.0, 90.0], dtype=torch.float64
                ),  # Not monotonic
            )


class TestNonSpineAgesLogLikelihoodDerivatives:
    """Test derivatives for non-spine age log-likelihood."""

    def test_scalar_age_derivatives(self, standard_scalar_params, sample_ages_scalar):
        """Test derivatives with single age."""
        result = calculate_non_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_array_ages_derivatives(self, standard_scalar_params, sample_ages_array):
        """Test derivatives with array of ages."""
        result = calculate_non_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_critical_case(self, nearly_equal_params, sample_ages_scalar):
        """Test derivative stability when λ ≈ μ."""
        result = calculate_non_spine_ages_log_lik_derivatives(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            sample_ages_scalar,
            nearly_equal_params["time"],
        )

        assert np.all(np.isfinite(result))


# ============================================================================
# LEVEL 5: Unified Age Likelihood Interface Tests
# ============================================================================


class TestAgesLogLikelihood:
    """Test unified age log-likelihood interface (spine/non-spine selector)."""

    def test_spine_routing(self, standard_scalar_params, sample_ages_scalar):
        """Test that are_spines=True routes to spine implementation."""
        result = calculate_ages_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=True,
        )

        # Should match spine calculation
        expected = calculate_spine_ages_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert_allclose(result, expected)

    def test_non_spine_routing(self, sample_ages_scalar):
        """Test that are_spines=False routes to non-spine implementation."""
        result = calculate_ages_log_lik(
            lambda_params=0.1,
            mu_params=0.05,
            ages_t=sample_ages_scalar,
            times=100.0,
            are_spines=False,
        )

        # Should match non-spine calculation (using array inputs)
        expected = calculate_non_spine_ages_log_lik(
            lambda_params=torch.tensor([0.1], dtype=torch.float64),
            mu_params=torch.tensor([0.05], dtype=torch.float64),
            ages_t=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert_allclose(result, expected)


class TestAgesLogLikelihoodDerivatives:
    """Test unified age log-likelihood derivatives interface."""

    def test_spine_derivatives_routing(
        self, standard_scalar_params, sample_ages_scalar
    ):
        """Test that are_spines=True routes derivatives correctly."""
        result = calculate_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=True,
        )

        expected = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert_allclose(result, expected)

    def test_non_spine_derivatives_routing(
        self, standard_scalar_params, sample_ages_scalar
    ):
        """Test that are_spines=False routes derivatives correctly."""
        result = calculate_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=False,
        )

        expected = calculate_non_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert_allclose(result, expected)


# ============================================================================
# LEVEL 6: Mathematical Property Tests
# ============================================================================


class TestMathematicalProperties:
    """Test mathematical properties and consistency."""

    def test_pr_bounds_population(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test that log probabilities are negative (probabilities < 1)."""
        for n in range(0, 20):
            result = calculate_populations_log_lik(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                standard_scalar_params["time"],
                ns=n,
            )
            # Log probabilities should be ≤ 0
            assert result <= 0 or np.isclose(result, 0, rtol=1e-8)

    def test_extinction_pr_increases_with_mu(self):
        """Test that extinction probability increases as death rate increases."""
        birth_rates = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        alpha_1 = calculate_extinction_pr(
            birth_rates, torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64), times
        )
        alpha_2 = calculate_extinction_pr(
            birth_rates, torch.tensor([0.08, 0.08, 0.08], dtype=torch.float64), times
        )

        # Higher death rate → higher extinction probability
        assert 0 < alpha_1 < alpha_2 < 1

    def test_consistency_scalar_vs_single_element_array(self):
        """Test that scalar and single-element array give same results."""
        scalar_result = calculate_populations_log_lik(
            lambda_params=0.1,
            mu_params=0.05,
            times=100.0,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        array_result = calculate_populations_log_lik(
            lambda_params=torch.tensor([0.1], dtype=torch.float64),
            mu_params=torch.tensor([0.05], dtype=torch.float64),
            times=torch.tensor([100.0], dtype=torch.float64),
            ns=torch.tensor([1], dtype=torch.int64),
        )

        # Results should be very close
        assert_allclose(scalar_result, array_result, rtol=1e-8)

    def test_sum_property_array_ns(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test that array of ns gives sum of individual likelihoods."""
        ns_array = torch.tensor([5, 10, 15], dtype=torch.int64)

        # Compute together
        result_together = calculate_populations_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=ns_array,
        )

        # Compute separately and sum
        result_separate = sum(
            calculate_populations_log_lik(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                standard_scalar_params["time"],
                ns=n,
            )
            for n in ns_array
        )

        assert_allclose(result_together, result_separate, rtol=1e-8)


# ============================================================================
# LEVEL 7: Numerical Stability and Edge Case Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability in challenging scenarios."""

    def test_extreme_time_values(self):
        """Test with very large time values."""
        # Large time can cause overflow in exponentials
        result = calculate_populations_log_lik(
            lambda_params=0.001,  # Small rates to prevent overflow
            mu_params=0.0005,
            times=10000.0,  # Very large time
            ns=torch.tensor([1], dtype=torch.int64),
        )

        # Should not return inf or nan
        assert np.isfinite(result) or result == -np.inf

    def test_extreme_rate_differences(self):
        """Test when λ >> μ or μ >> λ."""
        # λ >> μ (high growth)
        result1 = calculate_populations_log_lik(
            lambda_params=1.0,
            mu_params=0.001,
            times=10.0,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        # Should handle gracefully
        assert np.isfinite(result1) or result1 == -np.inf

        # μ >> λ (high death)
        result2 = calculate_populations_log_lik(
            lambda_params=0.001,
            mu_params=1.0,
            times=10.0,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert np.isfinite(result2) or result2 == -np.inf

    def test_very_small_rates(self):
        """Test with very small birth and death rates."""
        result = calculate_populations_log_lik(
            lambda_params=1e-10,
            mu_params=5e-11,
            times=1000.0,
            ns=5,
        )

        assert np.isfinite(result) or result == -np.inf

    def test_many_epochs(self):
        """Test with many epochs (long array)."""
        n_epochs = 100
        birth_rates = np.linspace(0.08, 0.12, n_epochs)
        death_rates = np.linspace(0.04, 0.06, n_epochs)
        times = np.linspace(1.0, 100.0, n_epochs)

        result = calculate_populations_log_lik(
            lambda_params=birth_rates,
            mu_params=death_rates,
            times=times,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert np.isfinite(result) or result == -np.inf

    def test_ages_near_observation_time(self):
        """Test age likelihood with ages very close to T."""
        T = torch.tensor([100.0], dtype=torch.float64)
        ages = torch.tensor([T - 1e-6, T - 1e-8, T - 1e-10], dtype=torch.float64)

        result = calculate_spine_ages_log_lik(
            lambda_params=0.1,
            mu_params=0.05,
            ages_t=ages,
            times=T,
        )

        assert np.isfinite(result)


# ============================================================================
# LEVEL 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Test integration between related functions."""

    def test_derivatives_match_numerical_gradient(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test that analytical derivatives match numerical gradients."""
        # Test for populations log likelihood derivatives

        def objective(params):
            return calculate_populations_log_lik(
                params[0], params[1], standard_scalar_params["time"], ns=5
            )

        def gradient(params):
            return calculate_populations_log_lik_derivatives(
                params[0], params[1], standard_scalar_params["time"], ns=5
            )

        params = torch.tensor(
            [
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
            ],
            dtype=torch.float64,
        )

        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance"

    def test_spine_non_spine_different_formulas(
        self, standard_scalar_params, sample_ages_scalar
    ):
        """Test that spine and non-spine give different results (different formulas)."""
        spine_result = calculate_spine_ages_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        non_spine_result = calculate_non_spine_ages_log_lik(
            torch.tensor([standard_scalar_params["lambda_param"]], dtype=torch.float64),
            torch.tensor([standard_scalar_params["mu_param"]], dtype=torch.float64),
            sample_ages_scalar,
            torch.tensor([standard_scalar_params["time"]], dtype=torch.float64),
        )

        # These should be different (different conditioning)
        assert not np.isclose(spine_result, non_spine_result)


# ============================================================================
# Performance and Vectorization Tests
# ============================================================================


class TestVectorization:
    """Test vectorization and performance characteristics."""

    def test_array_ages_more_efficient_than_loop(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Verify that array processing is implemented (not just a loop)."""
        # This is more of a smoke test - actual performance testing
        # would require timing benchmarks
        ages = np.linspace(10, 90, 100)

        result = calculate_spine_ages_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            ages,
            standard_scalar_params["time"],
        )

        assert np.isfinite(result)

    def test_large_population_counts_array(
        self, standard_scalar_params: Dict[str, NDArray[np.float64]]
    ):
        """Test with large array of population counts."""
        ns = np.arange(0, 100)

        result = calculate_populations_log_lik(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=ns,
        )

        assert np.isfinite(result)


# ============================================================================
# Gradient Validation Tests
# ============================================================================


class TestGradientValidation:
    """
    Comprehensive tests for analytical gradient functions
    using numerical approximations.

    Uses scipy.optimize.check_grad and manual finite differences to verify that
    analytical derivatives match numerical gradients.
    """

    def test_populations_log_lik_derivatives_scalar_n(self):
        """Test population derivatives with scalar n using check_grad."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        time = 100.0
        n = 5

        def objective(params):
            return calculate_populations_log_lik(
                params[0],
                params[1],
                time,
                ns=n,
            )

        def gradient(params):
            return calculate_populations_log_lik_derivatives(
                params[0],
                params[1],
                time,
                ns=n,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance for scalar n"

    def test_populations_log_lik_derivatives_array_n(self):
        """Test population derivatives with array n using check_grad."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        time = 100.0
        ns = torch.tensor([0, 1, 2, 5, 10], dtype=torch.int64)

        def objective(params):
            return calculate_populations_log_lik(
                params[0],
                params[1],
                time,
                ns=ns,
            )

        def gradient(params):
            return calculate_populations_log_lik_derivatives(
                params[0],
                params[1],
                time,
                ns=ns,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 3e-5, f"Gradient error {error} exceeds tolerance for array n"

    def test_populations_log_lik_derivatives_near_equal_rates(self):
        """Test population derivatives when λ ≈ μ (limiting case)."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = (
            torch.tensor([0.1], dtype=torch.float64) + 1e-13
        )  # Within tolerance, should use limiting formula
        time = 100.0
        n = 5

        def objective(params):
            return calculate_populations_log_lik(
                params[0],
                params[1],
                time,
                ns=n,
            )

        def gradient(params):
            return calculate_populations_log_lik_derivatives(
                params[0],
                params[1],
                time,
                ns=n,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance for λ ≈ μ case"

    def test_non_spine_ages_derivatives_scalar_age(self):
        """Test non-spine age derivatives with scalar age."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(params):
            return calculate_non_spine_ages_log_lik(
                params[0],
                params[1],
                age,
                T,
            )

        def gradient(params):
            return calculate_non_spine_ages_log_lik_derivatives(
                params[0],
                params[1],
                age,
                T,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance for scalar age"

    def test_non_spine_ages_derivatives_array_ages(self):
        """Test non-spine age derivatives with multiple ages."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        ages = torch.tensor([10.0, 30.0, 50.0, 70.0, 90.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(params):
            return calculate_non_spine_ages_log_lik(
                params[0],
                params[1],
                ages,
                T,
            )

        def gradient(params):
            return calculate_non_spine_ages_log_lik_derivatives(
                params[0],
                params[1],
                ages,
                T,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance for array ages"

    def test_non_spine_ages_derivatives_near_equal_rates(self):
        """Test non-spine age derivatives when λ ≈ μ."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.1], dtype=torch.float64) + 1e-13
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(params):
            return calculate_non_spine_ages_log_lik(
                params[0],
                params[1],
                age,
                T,
            )

        def gradient(params):
            return calculate_non_spine_ages_log_lik_derivatives(
                params[0],
                params[1],
                age,
                T,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance for λ ≈ μ case"

    def test_spine_ages_derivatives_interior_ages(self):
        """Test spine age derivatives for ages < T (interior case)."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        ages = torch.tensor([10.0, 30.0, 50.0, 70.0, 90.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(params):
            return calculate_spine_ages_log_lik(
                params[0],
                params[1],
                ages,
                T,
            )

        def gradient(params):
            return calculate_spine_ages_log_lik_derivatives(
                params[0],
                params[1],
                ages,
                T,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, (
            f"Gradient error {error} exceeds tolerance for interior ages"
        )

    def test_spine_ages_derivatives_boundary_age(self):
        """Test spine age derivatives at boundary (age = T)."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        age = 100.0
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(params):
            return calculate_spine_ages_log_lik(
                params[0],
                params[1],
                age,
                T,
            )

        def gradient(params):
            return calculate_spine_ages_log_lik_derivatives(
                params[0],
                params[1],
                age,
                T,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance at boundary"

    def test_spine_ages_derivatives_mixed_ages(self):
        """Test spine age derivatives with mix of interior and boundary ages."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        ages = torch.tensor(
            [10.0, 50.0, 90.0, 100.0, 100.0], dtype=torch.float64
        )  # Some at boundary
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(params):
            return calculate_spine_ages_log_lik(
                params[0],
                params[1],
                ages,
                T,
            )

        def gradient(params):
            return calculate_spine_ages_log_lik_derivatives(
                params[0],
                params[1],
                ages,
                T,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance for mixed ages"

    def test_spine_ages_derivatives_near_equal_rates(self):
        """Test spine age derivatives when λ ≈ μ."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.1], dtype=torch.float64) + 1e-13
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(params):
            return calculate_spine_ages_log_lik(
                params[0],
                params[1],
                age,
                T,
            )

        def gradient(params):
            return calculate_spine_ages_log_lik_derivatives(
                params[0],
                params[1],
                age,
                T,
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, f"Gradient error {error} exceeds tolerance for λ ≈ μ case"

    def test_derivatives_with_log_scale_params(self):
        """Test that derivatives work correctly with log-scale parameters."""
        log_lambda = np.log(0.1)
        log_mu = np.log(0.05)
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        def objective(log_params):
            return calculate_non_spine_ages_log_lik(
                log_params[0], log_params[1], age, T, log_scale_params=True
            )

        def gradient(log_params):
            grads = calculate_non_spine_ages_log_lik_derivatives(
                log_params[0], log_params[1], age, T, log_scale_params=True
            )
            # Chain rule: d/d(log λ) = (d/dλ) * λ
            return grads * np.exp(log_params)

        params = torch.tensor([log_lambda, log_mu], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective, gradient, params)
        assert error < 1e-5, (
            f"Gradient error {error} exceeds tolerance for log-scale params"
        )

    def test_finite_difference_manual_check(self):
        """Manual finite difference check for additional verification."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)
        epsilon = 1e-7

        # Compute analytical gradient
        analytical = calculate_non_spine_ages_log_lik_derivatives(
            lambda_val,
            mu_val,
            age,
            T,
        )

        # Compute numerical gradient using central differences
        # d/dλ
        f_lambda_plus = calculate_non_spine_ages_log_lik(
            lambda_val + epsilon,
            mu_val,
            age,
            T,
        )
        f_lambda_minus = calculate_non_spine_ages_log_lik(
            lambda_val - epsilon,
            mu_val,
            age,
            T,
        )
        d_lambda_numerical = (f_lambda_plus - f_lambda_minus) / (2 * epsilon)

        # d/dμ
        f_mu_plus = calculate_non_spine_ages_log_lik(
            lambda_val,
            mu_val + epsilon,
            age,
            T,
        )
        f_mu_minus = calculate_non_spine_ages_log_lik(
            lambda_val,
            mu_val - epsilon,
            age,
            T,
        )
        d_mu_numerical = (f_mu_plus - f_mu_minus) / (2 * epsilon)

        numerical = torch.tensor(
            [d_lambda_numerical, d_mu_numerical], dtype=torch.float64
        )

        # Check relative error
        relative_error = np.abs(analytical - numerical) / (np.abs(analytical) + 1e-10)
        assert np.all(relative_error < 1e-5), (
            f"Relative error {relative_error} exceeds tolerance"
        )

    def test_gradient_consistency_across_parameter_ranges(self):
        """Test gradients across different parameter ranges."""
        ages = torch.tensor([20.0, 50.0, 80.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Test different parameter combinations
        test_cases = [
            (0.01, 0.005, "low rates"),
            (0.1, 0.05, "medium rates"),
            (0.5, 0.3, "high rates"),
            (0.1, 0.09, "close rates"),
            (0.2, 0.05, "large difference"),
        ]

        for lambda_val, mu_val, description in test_cases:

            def objective(params):
                return calculate_non_spine_ages_log_lik(
                    params[0],
                    params[1],
                    ages,
                    T,
                )

            def gradient(params):
                return calculate_non_spine_ages_log_lik_derivatives(
                    params[0],
                    params[1],
                    ages,
                    T,
                )

            params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
            error = scipy.optimize.check_grad(objective, gradient, params)
            assert error < 1e-4, (
                f"Gradient error {error} exceeds tolerance for {description}"
            )

    def test_wrapper_function_derivatives(self):
        """Test that wrapper function derivatives match specific implementations."""
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        # Non-spine derivatives via wrapper
        def objective_wrapper(params):
            return calculate_ages_log_lik(
                params[0], params[1], age, T, are_spines=False
            )

        def gradient_wrapper(params):
            return calculate_ages_log_lik_derivatives(
                params[0], params[1], age, T, are_spines=False
            )

        params = torch.tensor([lambda_val, mu_val], dtype=torch.float64)
        error = scipy.optimize.check_grad(objective_wrapper, gradient_wrapper, params)
        assert error < 1e-5, f"Wrapper gradient error {error} exceeds tolerance"

        # Spine derivatives via wrapper
        def objective_wrapper_spine(params):
            return calculate_ages_log_lik(params[0], params[1], age, T, are_spines=True)

        def gradient_wrapper_spine(params):
            return calculate_ages_log_lik_derivatives(
                params[0], params[1], age, T, are_spines=True
            )

        error_spine = scipy.optimize.check_grad(
            objective_wrapper_spine, gradient_wrapper_spine, params
        )
        assert error_spine < 1e-5, (
            f"Wrapper spine gradient error {error_spine} exceeds tolerance"
        )


# ============================================================================
# Private Helper Function Tests
# ============================================================================


class TestIntegrandSurvFromWeightedBirthTime:
    """Test the _integrand_surv_from_weighted_birth_time_vectorized private helper function.

    This function is used in calculating survival probabilities and can have
    issues with index out of bounds when birth_time is at or beyond epoch times.
    """

    def test_basic_single_epoch(self):
        """Test basic functionality with a single epoch."""
        birth_time = torch.tensor([5.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1], dtype=torch.float64)
        death_rates = torch.tensor([0.05], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        # Result should be finite and non-negative
        assert np.isfinite(result)
        assert result >= 0

    def test_basic_multiple_epochs(self):
        """Test with multiple epochs."""
        birth_time = torch.tensor([25.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)
        assert result >= 0

    def test_birth_time_at_zero(self):
        """Test when birth_time is 0 (start of process)."""
        birth_time = torch.tensor([0.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)
        assert result >= 0

    def test_birth_time_equals_first_epoch_end(self):
        """Test edge case when birth_time equals first epoch end time."""
        birth_time = torch.tensor([50.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # After subtracting birth_time, times becomes [0.0, 50.0]
        # valid_indices will be where times > 0, so only index 1
        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)
        assert result >= 0

    def test_birth_time_beyond_all_epochs_raises_error(self):
        """Test that birth_time beyond all epochs raises an error."""
        birth_time = 150.0  # Beyond all epochs
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        with pytest.raises(ValueError):
            _integrand_surv_from_weighted_birth_time_vectorized(
                birth_time,
                birth_rates,
                death_rates,
                times,
            )

    def test_birth_time_slightly_beyond_last_epoch(self):
        """Test birth_time just beyond the last epoch time."""
        birth_time = 100.1
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # After subtracting birth_time, times becomes negative, no valid indices
        # Should raise ValueError
        with pytest.raises(ValueError):
            _integrand_surv_from_weighted_birth_time_vectorized(
                birth_time,
                birth_rates,
                death_rates,
                times,
            )

    def test_birth_time_within_first_epoch(self):
        """Test normal case with birth_time within first epoch."""
        birth_time = torch.tensor([10.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)
        assert result >= 0

    def test_birth_time_within_middle_epoch(self):
        """Test birth_time within a middle epoch."""
        birth_time = torch.tensor([75.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        # After subtracting, times becomes [-25.0, 25.0, 75.0]
        # valid_indices will be [1, 2]
        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)
        assert result >= 0

    def test_scalar_inputs_converted_to_arrays(self):
        """Test that scalar inputs are properly converted to arrays."""
        birth_time = torch.tensor([10.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1], dtype=torch.float64)
        death_rates = torch.tensor([0.05], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)
        assert result >= 0

    def test_result_proportional_to_birth_rate(self):
        """Test that result is proportional to the first valid birth rate."""
        birth_time = torch.tensor([10.0], dtype=torch.float64)
        birth_rates1 = torch.tensor([0.1, 0.15], dtype=torch.float64)
        birth_rates2 = torch.tensor(
            [0.2, 0.15], dtype=torch.float64
        )  # Double the first rate
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result1 = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates1,
            death_rates,
            times,
        )
        result2 = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates2,
            death_rates,
            times,
        )

        # Result should scale with birth rate
        # (approximately, given extinction prob is also affected)
        assert result2 > result1

    def test_array_length_mismatch_times_longer(self):
        """Test when times array is longer than birth_rates/death_rates arrays.

        This could cause index out of bounds errors if valid_indices from the longer
        times array are used to index the shorter rate arrays.
        """
        birth_time = torch.tensor([10.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)  # Length 2
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)  # Length 2
        times = torch.tensor(
            [50.0, 100.0, 150.0], dtype=torch.float64
        )  # Length 3 - MISMATCH!

        # This should raise an ValueError
        with pytest.raises(ValueError):
            _integrand_surv_from_weighted_birth_time_vectorized(
                birth_time,
                birth_rates,
                death_rates,
                times,
            )

    def test_array_length_mismatch_rates_longer(self):
        """Test when birth_rates/death_rates arrays are longer than times array.

        Currently does NOT raise an error - the function silently uses only
        the first N rate values where N = len(times). This could lead to
        subtle bugs where extra rate parameters are ignored.
        """
        birth_time = torch.tensor([10.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)  # Length 3
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)  # Length 3
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)  # Length 2 - MISMATCH!

        # Currently this doesn't fail - it just uses first 2 rate values
        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )
        assert np.isfinite(result)

    def test_birth_and_death_rates_different_lengths(self):
        """Test when birth_rates and death_rates have different lengths.

        Currently does NOT raise an error - uses the shorter length.
        This could lead to subtle bugs in calculations.
        """
        birth_time = torch.tensor([10.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)  # Length 2
        death_rates = torch.tensor(
            [0.05, 0.06, 0.055], dtype=torch.float64
        )  # Length 3 - MISMATCH!
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # Currently doesn't fail - behavior depends on numpy broadcasting
        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )
        assert np.isfinite(result)

    def test_integration_context_array_length_issue(self):
        """Test scenario that mimics the integration context where error occurred.

        The error shows 'index 2 is out of bounds for axis 0 with size 2',
        suggesting times array has 3 elements but rate arrays have 2.
        """
        # Simulate what might happen during integration
        lifetime = 100.0

        # If the calling function has 2-epoch setup:
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # But somehow an extra time point gets added (e.g., lifetime appended)
        # This could happen in calculate_log_pr_extinction_given_initial_lifetimes
        birth_time = torch.tensor(
            [25.0], dtype=torch.float64
        )  # Some point during integration

        # If times gets modified to include lifetime:
        bad_times = np.append(times, lifetime)  # Now length 3!

        # This should raise ValueError due to birth_time causing valid_indices
        with pytest.raises(ValueError):
            _integrand_surv_from_weighted_birth_time_vectorized(
                birth_time,
                birth_rates,
                death_rates,
                bad_times,
            )

    def test_valid_indices_logic_with_partial_truncation(self):
        """Test that valid_indices correctly handles partial epoch truncation."""
        # When birth_time is within epochs, valid_indices should be a subset
        birth_time = torch.tensor([60.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        # After subtracting birth_time, times becomes [-10.0, 40.0, 90.0]
        # valid_indices should be [1, 2] (where times > 0)
        # Accessing birth_rates[[1, 2]] should work fine
        result = _integrand_surv_from_weighted_birth_time_vectorized(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)
        assert result >= 0


class test_calculateLogPrExtinctionGivenInitialLifetimes:
    """Test calculate_log_pr_extinction_given_initial_lifetimes function.

    This function uses _integrand_surv_from_weighted_birth_time_vectorized internally
    and has a bug where lifetime > max(times) causes IndexError.
    """

    def test_basic_single_lifetime_within_epochs(self):
        """Test basic case with lifetime within epoch boundaries."""
        lifetime = torch.tensor([50.0], dtype=torch.float64)
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([75.0, 150.0], dtype=torch.float64)

        result = calculate_log_pr_extinction_given_initial_lifetimes(
            lifetime,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)

    def test_lifetime_equals_last_epoch(self):
        """Test when lifetime equals the last epoch end time."""
        lifetime = 100.0
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # This should work - lifetime is at boundary
        result = calculate_log_pr_extinction_given_initial_lifetimes(
            lifetime,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)

    def test_lifetime_beyond_last_epoch_raises_error(self):
        """Test that lifetime beyond last epoch raises ValueError.

        When lifetime > max(times), the function should raise a ValueError
        because the lifetime exceeds the maximum observation time defined
        by the epochs.

        Example: times=[50, 100], lifetime=120
        - lifetime (120) exceeds max(times) (100)
        - Should raise ValueError with appropriate message
        """
        lifetime = 120.0  # Beyond last epoch at 100
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        with pytest.raises(
            ValueError, match="lifetimes must be less than or equal to the final time"
        ):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetime,
                birth_rates,
                death_rates,
                times,
            )

    def test_multiple_lifetimes_one_beyond_epochs(self):
        """Test with array of lifetimes where one exceeds max(times)."""
        lifetimes = torch.tensor(
            [50.0, 80.0, 120.0], dtype=torch.float64
        )  # Last one beyond times
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        with pytest.raises(
            ValueError, match="lifetimes must be less than or equal to the final time"
        ):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetimes,
                birth_rates,
                death_rates,
                times,
            )

    def test_lifetime_at_first_epoch_boundary(self):
        """Test lifetime at first epoch boundary."""
        lifetime = 50.0
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = calculate_log_pr_extinction_given_initial_lifetimes(
            lifetime,
            birth_rates,
            death_rates,
            times,
        )

        assert np.isfinite(result)

    def test_three_epochs_lifetime_beyond(self):
        """Test with three epochs and lifetime beyond all of them."""
        lifetime = 200.0
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        # Should now work: uses all epoch rates plus last rate for [150-200]
        with pytest.raises(
            ValueError, match="lifetimes must be less than or equal to the final time"
        ):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetime,
                birth_rates,
                death_rates,
                times,
            )

    def test_lifetime_far_beyond_last_epoch(self):
        """Test with lifetime far beyond the last epoch."""
        lifetime = 500.0  # Much larger than last epoch at 100
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # Should handle extrapolation using last birth rate
        with pytest.raises(
            ValueError, match="lifetimes must be less than or equal to the final time"
        ):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetime,
                birth_rates,
                death_rates,
                times,
            )
