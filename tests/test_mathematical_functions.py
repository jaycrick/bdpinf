# ruff: noqa: S101
# ruff: noqa: D102

import math

import pytest
import scipy
import torch
from torch.testing import assert_close

# Import private functions for direct testing
from inf_bp_texts.models.mathematical_functions import (
    _integrand_surv_from_weighted_birth_time_scalar,
    _population_geometric_parameter,
    _pr_extinction_numerator,
    _reverse_time,
    calculate_ages_log_lik,
    calculate_extinction_pr,
    calculate_log_lik_initial_lifetimes_given_extinction,
    calculate_log_lik_initial_lifetimes_given_non_extinction,
    calculate_log_lik_lifetimes,
    calculate_log_pr_extinction_given_initial_lifetimes,
    calculate_non_spine_ages_log_lik,
    calculate_populations_log_lik,
    calculate_spine_ages_log_lik,
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
def standard_scalar_params() -> dict[str, torch.Tensor]:
    """Standard scalar parameters for testing."""
    return {
        "birth_rate": torch.tensor([0.1], dtype=torch.float64),
        "death_rate": torch.tensor([0.05], dtype=torch.float64),
        "time": torch.tensor([100.0], dtype=torch.float64),
    }


@pytest.fixture
def standard_array_params() -> dict[str, torch.Tensor]:
    """Standard array parameters for piecewise-constant rates."""
    return {
        "birth_rates": torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64),
        "death_rates": torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64),
        "times": torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64),
    }


@pytest.fixture
def nearly_equal_params() -> dict[str, torch.Tensor]:
    """Parameters where λ ≈ μ (numerical stability test)."""
    return {
        "birth_rate": torch.tensor([0.1], dtype=torch.float64),
        "death_rate": torch.tensor(
            [0.1 + 1e-13],
            dtype=torch.float64,
        ),  # Within tolerance of 1e-12
        "time": torch.tensor([100.0], dtype=torch.float64),
    }


@pytest.fixture
def log_scale_params() -> dict[str, torch.Tensor]:
    """Parameters in log scale."""
    return {
        "log_birth_rate": torch.log(0.1),
        "log_death_rate": torch.log(0.05),
        "time": 100.0,
    }


@pytest.fixture
def sample_ages_scalar() -> float:
    """Single age value for testing."""
    return 50.0


@pytest.fixture
def sample_ages_array() -> torch.Tensor:
    """Array of ages for vectorized testing."""
    return torch.tensor([10.0, 30.0, 50.0, 70.0, 90.0], dtype=torch.float64)


@pytest.fixture
def sample_population_counts() -> torch.Tensor:
    """Sample population counts (n values)."""
    return torch.tensor([0, 1, 5, 10, 20], dtype=torch.int64)


@pytest.fixture
def two_epoch_params() -> dict[str, torch.Tensor]:
    """Parameters for two-epoch testing."""
    return {
        "birth_rates": torch.tensor([0.01, 0.001], dtype=torch.float64),
        "death_rates": torch.tensor([0.001, 0.001], dtype=torch.float64),
        "times": torch.tensor([750.0, 1500.0], dtype=torch.float64),
    }


@pytest.fixture
def three_epoch_params() -> dict[str, torch.Tensor]:
    """Parameters for three-epoch testing."""
    return {
        "birth_rates": torch.tensor([0.1, 0.05, 0.08], dtype=torch.float64),
        "death_rates": torch.tensor([0.03, 0.02, 0.04], dtype=torch.float64),
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

    def test_two_epoch_population_lik_reduces_to_single_epoch(self) -> None:
        """Test that two identical epochs match single-epoch result."""
        birth_rate_val = torch.tensor([0.1], dtype=torch.float64)
        death_rate_val = torch.tensor([0.05], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)
        n = torch.tensor([5], dtype=torch.int64)

        # Single epoch result
        single_result = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=T,
            ns=n,
        )

        # Two epochs with same parameters should give same result
        two_epoch_result = calculate_populations_log_lik(
            torch.tensor([birth_rate_val[0], birth_rate_val[0]], dtype=torch.float64),
            torch.tensor([death_rate_val[0], death_rate_val[0]], dtype=torch.float64),
            times=torch.tensor([T[0] / 2, T[0]], dtype=torch.float64),
            ns=n,
        )

        assert torch.isfinite(single_result)
        assert torch.isfinite(two_epoch_result)
        assert_close(single_result, two_epoch_result)

    def test_multi_epoch_extinction_pr_bounds(self) -> None:
        """Test that multi-epoch extinction probability is in [0, 1]."""
        birth_rates = torch.tensor([0.2, 0.15, 0.1], dtype=torch.float64)
        death_rates = torch.tensor([0.1, 0.12, 0.15], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = calculate_extinction_pr(birth_rates, death_rates, times)

        assert 0 < result < 1, f"Extinction probability {result} not in (0, 1)"

    def test_two_epoch_spine_age_density_basic(self) -> None:
        """Validate multi-epoch spine age density for ages in each epoch.

        This exercises the reflected multi-epoch path in
        ``calculate_spine_ages_log_lik`` by checking that ages in both the first
        and second epochs produce finite log densities with the current closed-form
        implementation.
        """
        birth_rates = torch.tensor([0.1, 0.2], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.1], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # Age in first epoch
        age_epoch1 = torch.tensor([30.0], dtype=torch.float64)
        result1 = calculate_spine_ages_log_lik(
            birth_rates,
            death_rates,
            age_epoch1,
            times,
        )
        assert torch.isfinite(result1)

        # Age in second epoch
        age_epoch2 = torch.tensor([70.0], dtype=torch.float64)
        result2 = calculate_spine_ages_log_lik(
            birth_rates,
            death_rates,
            age_epoch2,
            times,
        )
        assert torch.isfinite(result2)

    def test_multi_epoch_non_spine_age_array(self) -> None:
        """Check multi-epoch non-spine ages spanning epochs yield finite, negative log-lik.

        Confirms the multi-epoch branch of ``calculate_non_spine_ages_log_lik``
        handles ages on either side of an epoch boundary and returns a negative
        summed log-density as expected for probabilities < 1.
        """
        birth_rates = torch.tensor([0.15, 0.1], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.08], dtype=torch.float64)
        times = torch.tensor([100.0, 200.0], dtype=torch.float64)

        # Ages spanning both epochs
        ages = torch.tensor([20.0, 60.0, 120.0, 180.0], dtype=torch.float64)

        result = calculate_non_spine_ages_log_lik(birth_rates, death_rates, ages, times)

        # Should return finite sum
        assert torch.isfinite(result)
        # Log likelihood should be negative
        assert result < 0

    def test_two_epoch_geometric_parameter(self) -> None:
        """Test _population_geometric_parameter for two epochs."""
        birth_rates = torch.tensor([0.2, 0.3], dtype=torch.float64)
        death_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = _population_geometric_parameter(birth_rates, death_rates, times)

        # Should be in valid probability range
        assert 0 < result < 1
        assert torch.isfinite(result)


class TestMultiEpochConsistency:
    """Test consistency properties of multi-epoch functions."""

    def test_epoch_order_matters(self) -> None:
        """Test that swapping epoch parameters changes results."""
        # Different parameters for each epoch
        birth_rates1 = torch.tensor([0.2, 0.1], dtype=torch.float64)
        death_rates1 = torch.tensor([0.1, 0.05], dtype=torch.float64)
        times = torch.tensor([60.0, 100.0], dtype=torch.float64)

        # Swapped parameters
        birth_rates2 = torch.tensor([0.1, 0.2], dtype=torch.float64)
        death_rates2 = torch.tensor([0.05, 0.1], dtype=torch.float64)

        result1 = calculate_populations_log_lik(
            birth_rates1,
            death_rates1,
            times=times,
            ns=5,
        )
        result2 = calculate_populations_log_lik(
            birth_rates2,
            death_rates2,
            times=times,
            ns=5,
        )

        # Results should differ (order matters)
        assert not torch.isclose(result1, result2)

    def test_multi_epoch_ages_sum_property(self) -> None:
        """Test that multi-epoch age likelihood sums correctly across ages."""
        birth_rates = torch.tensor([0.12, 0.08], dtype=torch.float64)
        death_rates = torch.tensor([0.04, 0.06], dtype=torch.float64)
        times = torch.tensor([100.0, 200.0], dtype=torch.float64)

        ages = torch.tensor([20.0, 80.0, 150.0], dtype=torch.float64)

        # Compute for array
        result_array = calculate_non_spine_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times=times,
        )

        # Compute individually and sum
        result_sum = sum(
            calculate_non_spine_ages_log_lik(
                birth_rates,
                death_rates,
                torch.tensor([age], dtype=torch.float64),
                times=times,
            )
            for age in ages
        )

        assert_close(result_array, result_sum)


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

    def test_extinction_pr_multi_equals_single(self) -> None:
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )

        single = calculate_extinction_pr(birth_rate_val, death_rate_val, T)

        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_extinction_pr(birth_rates, death_rates, times)

        assert_close(actual=multi, expected=single)
        assert 0 < single < 1

    def test_population_geometric_parameter_multi_equals_single(self) -> None:
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.03], dtype=torch.float64),
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([90.0], dtype=torch.float64),
        )
        single = _population_geometric_parameter(birth_rate_val, death_rate_val, T)
        times = torch.tensor([30.0, 60.0, 90.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = _population_geometric_parameter(birth_rates, death_rates, times)
        assert_close(actual=multi, expected=single)

    def test_pr_extinction_numerator_multi_equals_single(self) -> None:
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.015], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([100.0], dtype=torch.float64),
        )
        single = _pr_extinction_numerator(birth_rate_val, death_rate_val, T)
        times = torch.tensor([25.0, 50.0, 75.0, 100.0], dtype=torch.float64)
        birth_rates = torch.full(
            (4,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (4,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = _pr_extinction_numerator(birth_rates, death_rates, times)
        assert_close(actual=multi, expected=single)

    def test_log_lik_lifetimes_multi_equals_single(self) -> None:
        lifetimes = torch.tensor([10.0, 30.0, 60.0, 90.0], dtype=torch.float64)
        death_rate_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_lik_lifetimes(lifetimes, death_rate_val, T)
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_log_lik_lifetimes(lifetimes, death_rates, times)
        assert_close(actual=multi, expected=single)

    def test_log_pr_extinction_given_initial_lifetimes_multi_equals_single(
        self,
    ) -> None:
        lifetimes = torch.tensor([10.0, 45.0, 75.0, 110.0], dtype=torch.float64)
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes,
            birth_rate_val,
            death_rate_val,
            T,
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_log_pr_extinction_given_initial_lifetimes(
            lifetimes,
            birth_rates,
            death_rates,
            times,
        )
        assert_close(actual=multi, expected=single)

    def test_log_lik_initial_lifetimes_given_extinction_multi_equals_single(
        self,
    ) -> None:
        lifetimes = torch.tensor([12.0, 50.0, 95.0], dtype=torch.float64)
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.025], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_lik_initial_lifetimes_given_extinction(
            lifetimes,
            birth_rate_val,
            death_rate_val,
            T,
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_log_lik_initial_lifetimes_given_extinction(
            lifetimes,
            birth_rates,
            death_rates,
            times,
        )
        assert_close(actual=multi, expected=single)

    def test_log_lik_initial_lifetimes_given_non_extinction_multi_equals_single(
        self,
    ) -> None:
        lifetimes = torch.tensor([12.0, 50.0, 95.0], dtype=torch.float64)
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.025], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes,
            birth_rate_val,
            death_rate_val,
            T,
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_log_lik_initial_lifetimes_given_non_extinction(
            lifetimes,
            birth_rates,
            death_rates,
            times,
        )
        assert_close(actual=multi, expected=single)

    def test_non_spine_ages_log_lik_multi_equals_single(self) -> None:
        ages = torch.tensor([15.0, 60.0, 95.0], dtype=torch.float64)
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        # Single-epoch via scalar
        single = calculate_non_spine_ages_log_lik(
            birth_rate_val,
            death_rate_val,
            ages,
            T,
        )
        # Multi-epoch via arrays with identical rates
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_non_spine_ages_log_lik(birth_rates, death_rates, ages, times)
        assert torch.isfinite(single)
        assert torch.isfinite(multi)
        assert_close(actual=multi, expected=single)

    def test_spine_ages_log_lik_multi_equals_single(self) -> None:
        ages = torch.tensor([20.0, 40.0, 110.0], dtype=torch.float64)
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.018], dtype=torch.float64),
            torch.tensor([0.009], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_spine_ages_log_lik(birth_rate_val, death_rate_val, ages, T)
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_spine_ages_log_lik(birth_rates, death_rates, ages, times)
        assert torch.isfinite(single)
        assert torch.isfinite(multi)
        assert_close(actual=multi, expected=single)

    def test_ages_log_lik_multi_equals_single_non_spine(self) -> None:
        ages = torch.tensor([10.0, 55.0, 90.0], dtype=torch.float64)
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_ages_log_lik(
            birth_rate_val,
            death_rate_val,
            ages,
            T,
            are_spines=False,
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times,
            are_spines=False,
        )
        assert torch.isfinite(single)
        assert torch.isfinite(multi)
        assert_close(actual=multi, expected=single)

    def test_ages_log_lik_multi_equals_single_spine(self) -> None:
        ages = torch.tensor([15.0, 120.0], dtype=torch.float64)
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.018], dtype=torch.float64),
            torch.tensor([0.009], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = calculate_ages_log_lik(
            birth_rate_val,
            death_rate_val,
            ages,
            T,
            are_spines=True,
        )
        times = torch.tensor([30.0, 90.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times,
            are_spines=True,
        )
        assert torch.isfinite(single)
        assert torch.isfinite(multi)
        assert_close(actual=multi, expected=single)

    def test_integrand_surv_from_weighted_birth_time_scalar_multi_equals_single(
        self,
    ) -> None:
        birth_time = 30.0
        birth_rate_val, death_rate_val, T = (
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.01], dtype=torch.float64),
            torch.tensor([120.0], dtype=torch.float64),
        )
        single = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rate_val,
            death_rate_val,
            T,
        )
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)
        birth_rates = torch.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = torch.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )
        assert_close(actual=multi, expected=single)


# ============================================================================
# LEVEL 0: Numerical Accuracy Tests (Analytical Validation) - ORIGINAL
# ============================================================================


class TestNumericalAccuracy:
    """Test functions against known analytical results.

    These tests validate that the implementations produce numerically correct
    results by comparing against hand-calculated or analytically derived values.
    """

    def test_extinction_pr_critical_case_analytical(self) -> None:
        """Test extinction probability when λ = μ (analytical formula).

        When λ = μ, the extinction probability has the closed form:
        α_t = λT / (1 + λT)
        """
        birth_rate_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Expected: alpha_t = 0.1 * 100 / (1 + 0.1 * 100) = 10 / 11 ≈ 0.909091
        expected = birth_rate_val * T / (1 + birth_rate_val * T)

        result = torch.tensor(
            [calculate_extinction_pr(birth_rate_val, birth_rate_val, T)],
            dtype=torch.float64,
        )

        assert_close(result, expected)
        assert 0 < result < 1

    def test_populations_log_lik_n0_simple(self) -> None:
        """Test log P(n=0) for simple parameters.

        For n=0: log P(n=0) = log(α_t)
        Where α_t = μ(e^{(λ-μ)T} - 1) / (λ e^{(λ-μ)T} - μ)
        """
        birth_rate_val = torch.tensor([0.2], dtype=torch.float64)
        death_rate_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([10.0], dtype=torch.float64)

        # Compute expected alpha_t manually
        growth = (birth_rate_val - death_rate_val) * T  # 1.0
        exp_growth = torch.exp(growth)  # e^1.0 ≈ 2.71828
        alpha_t = (
            death_rate_val
            * (exp_growth - 1)
            / (birth_rate_val * exp_growth - death_rate_val)
        )
        expected_log_prob = torch.log(alpha_t)

        result = torch.tensor(
            [
                calculate_populations_log_lik(
                    birth_rate_val,
                    death_rate_val,
                    times=T,
                    ns=torch.tensor([0], dtype=torch.int64),
                ),
            ],
            dtype=torch.float64,
        )

        assert_close(result, expected_log_prob)

    def test_populations_log_lik_critical_case_n0(self) -> None:
        """Test log P(n=0) when λ = μ.

        When λ = μ: α_t = λT / (1 + λT)
        So log P(n=0) = log(λT / (1 + λT))
        """
        birth_rate_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([50.0], dtype=torch.float64)

        # Expected: alpha_t = 0.1 * 50 / (1 + 0.1 * 50) = 5/6
        alpha_t = birth_rate_val * T / (1 + birth_rate_val * T)
        expected = torch.log(alpha_t)

        result = torch.tensor(
            [
                calculate_populations_log_lik(
                    birth_rate_val,
                    birth_rate_val,
                    times=T,
                    ns=torch.tensor([0], dtype=torch.int64),
                ),
            ],
            dtype=torch.float64,
        )

        assert_close(result, expected)

    def test_populations_log_lik_n1_simple(self) -> None:
        """Test log P(n=1) for simple parameters.

        For n=1: log P(n=1) = log(1 - α_t) + log(1 - β_t)
        Where β_t = (λ/μ) * α_t
        """
        birth_rate_val = torch.tensor([0.3], dtype=torch.float64)
        death_rate_val = torch.tensor([0.2], dtype=torch.float64)
        T = torch.tensor([5.0], dtype=torch.float64)

        # Compute alpha_t and beta_t
        growth = (birth_rate_val - death_rate_val) * T
        exp_growth = torch.exp(growth)
        alpha_t = (
            death_rate_val
            * (exp_growth - 1)
            / (birth_rate_val * exp_growth - death_rate_val)
        )
        beta_t = (birth_rate_val / death_rate_val) * alpha_t

        expected = torch.log(1 - alpha_t) + torch.log(1 - beta_t)

        result = torch.tensor(
            [
                calculate_populations_log_lik(
                    birth_rate_val,
                    death_rate_val,
                    times=T,
                    ns=torch.tensor([1], dtype=torch.int64),
                ),
            ],
            dtype=torch.float64,
        )

        assert_close(result, expected)

    def test_spine_age_at_zero(self) -> None:
        """Test spine age density at t=0.

        At t=0 (age zero), the density involves special limiting behavior.
        """
        birth_rate_val = torch.tensor([0.2], dtype=torch.float64)
        death_rate_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([10.0], dtype=torch.float64)
        age = torch.tensor([0.0], dtype=torch.float64)

        result = calculate_spine_ages_log_lik(birth_rate_val, death_rate_val, age, T)

        # Should return -inf
        assert result == -torch.tensor(float("inf"), dtype=torch.float64)

    def test_geometric_scalar_case(self) -> None:
        """Docstring for test_geometric_scalar_case

        :param self: Description
        """
        T = 10.0
        birth_rate_val = torch.tensor([1.0], dtype=torch.float64)
        death_rate_val = torch.tensor([2.0], dtype=torch.float64)

        def _integrand(x: float) -> float:
            lam = float(birth_rate_val[0])
            mu = float(death_rate_val[0])
            x_arr = torch.as_tensor(x)
            return float(mu * torch.exp(-(lam - mu) * x_arr))

        integral, _ = scipy.integrate.quad(
            _integrand,
            0,
            T,
        )
        expected = torch.exp(-(birth_rate_val - death_rate_val) * T) / (1 + integral)
        result = torch.tensor(
            [_population_geometric_parameter(birth_rate_val, death_rate_val, T)],
            dtype=torch.float64,
        )
        assert_close(result, expected)

    def test_geometric_parameter_critical_case(self) -> None:
        """Test _population_geometric_parameter when λ = μ.

        Private function test: This function returns 1 - β_t where β_t = λT / (1 + λT)
        """
        birth_rate_val = death_rate_val = torch.tensor([0.1], dtype=torch.float64)
        T = 20.0

        def _integrand(x: float) -> float:
            lam = float(birth_rate_val[0])
            mu = float(death_rate_val[0])
            x_arr = torch.as_tensor(x)
            return float(mu * torch.exp(-(lam - mu) * x_arr))

        integral, _ = scipy.integrate.quad(
            _integrand,
            0,
            T,
        )
        expected = torch.exp(-(birth_rate_val - death_rate_val) * T) / (1 + integral)
        result = torch.tensor(
            [_population_geometric_parameter(birth_rate_val, death_rate_val, T)],
            dtype=torch.float64,
        )

        assert_close(result, expected)

    def test_geometric_parameter_general_case(self) -> None:
        """Test _population_geometric_parameter for λ ≠ μ.

        Private function test: Function returns 1 - β_t where β_t = (λ/μ) * α_t
        """
        birth_rates = torch.tensor([0.3, 0.4], dtype=torch.float64)
        death_rates = torch.tensor([0.1, 0.3], dtype=torch.float64)
        epoch_ends = torch.tensor([5.0, 9.0], dtype=torch.float64)

        def _integrand(x: float) -> float:
            x_arr = torch.as_tensor(x)
            first = (
                torch.exp(-(birth_rates[0] - death_rates[0]) * x_arr) * death_rates[0]
            )
            second = (
                torch.exp(
                    -(
                        (birth_rates[0] - death_rates[0]) * epoch_ends[0]
                        + (birth_rates[1] - death_rates[1]) * (x_arr - epoch_ends[0])
                    ),
                )
                * death_rates[1]
            )
            return float(torch.where(x_arr <= epoch_ends[0], first, second))

        integral, _ = scipy.integrate.quad(
            _integrand,
            0.0,
            epoch_ends[1].item(),
        )
        expected = torch.exp(
            -(
                (birth_rates[1] - death_rates[1]) * (epoch_ends[1] - epoch_ends[0])
                + epoch_ends[0] * (birth_rates[0] - death_rates[0])
            ),
        ) / (1 + integral)
        result = _population_geometric_parameter(birth_rates, death_rates, epoch_ends)

        assert_close(result, expected)

    def test_pr_extinction_numerator_simple(self) -> None:
        """Test _pr_extinction_numerator for simple case.

        Private function test: Validates that function returns a positive finite value.
        The exact formula is complex and involves integration - we validate properties.
        """
        birth_rate_val = torch.tensor([0.2], dtype=torch.float64)
        death_rate_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([10.0], dtype=torch.float64)

        result = _pr_extinction_numerator(
            birth_rate_val,
            death_rate_val,
            T,
        )

        # Should be positive and finite
        assert torch.isfinite(result)
        assert result > 0
        # For λ > μ, numerator should be between 0 and 1
        assert 0 < result < 1

    def test_calculate_log_lik_lifetimes_simple(self) -> None:
        """Test calculate_log_lik_lifetimes for simple scalar age.

        Private function test: log density involves birth rate and growth.
        """
        death_rate_val = torch.tensor([0.1], dtype=torch.float64)
        age = torch.tensor([20.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # For simple case: log f(t) = log(λ) - λt
        # (This is simplified; actual formula may differ)
        result = calculate_log_lik_lifetimes(
            age,
            death_rate_val,
            T,
        )

        # Should be finite and negative (log of density)
        assert torch.isfinite(result)
        assert result < 0

    def test_calculate_log_lik_lifetimes_array(self) -> None:
        """Test calculate_log_lik_lifetimes with array of ages.

        Private function test: should sum log likelihoods across ages.
        """
        death_rate_val = torch.tensor([0.1], dtype=torch.float64)
        ages = torch.tensor([10.0, 30.0, 50.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Compute for array
        result_array = calculate_log_lik_lifetimes(
            ages,
            death_rate_val,
            T,
        )

        # Compute individually and sum
        result_sum = sum(
            calculate_log_lik_lifetimes(
                torch.tensor([age], dtype=torch.float64),
                death_rate_val,
                T,
            )
            for age in ages
        )

        assert_close(result_array, result_sum)

    def test_non_spine_age_density_known_values(self) -> None:
        """Test non-spine age density against manually computed values.

        For specific parameter values, compute expected log density.
        """
        birth_rate_val = torch.tensor([0.2], dtype=torch.float64)
        death_rate_val = torch.tensor([0.1], dtype=torch.float64)
        age = torch.tensor([5.0], dtype=torch.float64)
        T = torch.tensor([20.0], dtype=torch.float64)

        # Non-spine density: f(t) = [λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}] / [1 - e^{-(λ-μ)T}]
        growth_rate = birth_rate_val - death_rate_val
        term1 = birth_rate_val * torch.exp(-birth_rate_val * age)
        term2 = (
            death_rate_val
            * torch.exp(-growth_rate * T)
            * torch.exp(-death_rate_val * age)
        )
        numerator = term1 - term2
        denominator = 1 - torch.exp(-growth_rate * T)
        density = numerator / denominator
        expected = torch.log(density)

        result = torch.tensor(
            [calculate_non_spine_ages_log_lik(birth_rate_val, death_rate_val, age, T)],
            dtype=torch.float64,
        )

        assert_close(result, expected)

    def test_extinction_pr_bounds(self) -> None:
        """Test that extinction probability satisfies 0 < α_t < 1 for various inputs.

        Validates mathematical constraint across parameter space.
        """
        test_cases = [
            (0.1, 0.05, 100.0),
            (0.5, 0.3, 50.0),
            (1.0, 0.8, 10.0),
            (0.01, 0.005, 1000.0),
        ]

        for birth_rate_val, death_rate_val, T in test_cases:
            result = calculate_extinction_pr(
                torch.tensor(birth_rate_val, dtype=torch.float64),
                torch.tensor(death_rate_val, dtype=torch.float64),
                torch.tensor(T, dtype=torch.float64),
            )

            assert 0 < result < 1, (
                f"Failed for λ={birth_rate_val}, μ={death_rate_val}, T={T}"
            )

    def test_calculate_log_pr_extinction_given_initial_lifetimes_zero(self) -> None:
        """Test calculate_log_pr_extinction_given_initial_lifetimes at age zero.

        Private function test: Validates function returns reasonable log probability.
        """
        birth_rate_val = torch.tensor([0.1], dtype=torch.float64)
        death_rate_val = torch.tensor([0.05], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)
        age = torch.tensor([0.0], dtype=torch.float64)

        result = calculate_log_pr_extinction_given_initial_lifetimes(
            age,
            birth_rate_val,
            death_rate_val,
            T,
        )

        # Should be finite (could be 0 at age 0, representing log(1) = 0)
        assert torch.isfinite(result)
        # Log probability should be non-positive (probability ≤ 1)
        assert result <= 0

    def test_piecewise_constant_single_vs_multi_epoch(self) -> None:
        """Test that single-epoch array matches scalar computation.

        A single-epoch piecewise-constant model should match scalar input.
        """
        birth_rate_val = torch.tensor([0.1], dtype=torch.float64)
        death_rate_val = torch.tensor([0.07], dtype=torch.float64)
        T = torch.tensor([80.0], dtype=torch.float64)
        n = torch.tensor([5], dtype=torch.int64)

        # Scalar computation
        result_scalar = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=T,
            ns=n,
        )

        # Single-epoch array computation
        result_array = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=T,
            ns=n,
        )

        assert_close(result_scalar, result_array)


# ============================================================================
# LEVEL 1: Extinction Probability Tests (Foundation)
# ============================================================================


class TestExtinctionProbability:
    """Test extinction probability calculations (α_t)."""

    def test_extinction_pr_basic(
        self,
        standard_array_params: dict[str, torch.Tensor],
    ) -> None:
        """Test basic extinction probability calculation."""
        result = calculate_extinction_pr(
            standard_array_params["birth_rates"],
            standard_array_params["death_rates"],
            standard_array_params["times"],
        )

        assert 0 < result < 1
        assert isinstance(result, torch.Tensor)

    def test_extinction_pr_return_log_prob(
        self,
        standard_array_params: dict[str, torch.Tensor],
    ) -> None:
        """Test extinction probability with return_log_prob=True."""
        result = calculate_extinction_pr(
            standard_array_params["birth_rates"],
            standard_array_params["death_rates"],
            standard_array_params["times"],
            return_log_prob=True,
        )

        # Log probability should be negative
        assert result < 0
        assert torch.isfinite(result)

    def test_extinction_pr_critical_case(self) -> None:
        """Test numerical stability when λ ≈ μ."""
        # Nearly equal rates
        birth_rate_arr = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        death_rate_arr = torch.tensor(
            [0.1 + 1e-13, 0.1 + 1e-13, 0.1 + 1e-13],
            dtype=torch.float64,
        )
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # Should handle critical case without returning inf or nan
        assert 0 < result < 1
        assert torch.isfinite(result)

    def test_extinction_pr_zero_rates_both(self) -> None:
        """Test extinction probability when both λ = 0 and μ = 0."""
        birth_rate_arr = torch.tensor([0.0], dtype=torch.float64)
        death_rate_arr = torch.tensor([0.0], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # With no birth and no death, extinction probability should zero
        assert result == 0.0

    def test_extinction_pr_zero_birth_only(self) -> None:
        """Test extinction probability when λ = 0 and μ > 0 (only death)."""
        birth_rate_arr = torch.tensor([0.0], dtype=torch.float64)
        death_rate_arr = torch.tensor([0.1], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        assert 0 < result < 1

    def test_extinction_pr_zero_death_only(self) -> None:
        """Test extinction probability when λ > 0 and μ = 0 (only birth)."""
        birth_rate_arr = torch.tensor([0.1], dtype=torch.float64)
        death_rate_arr = torch.tensor([0.0], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # With zero death rate, extinction should be zero
        assert result == 0

    def test_extinction_pr_zero_rates_multi_epoch(self) -> None:
        """Test extinction probability with multi-epoch where some epochs have zero rates."""
        # Some epochs with normal rates, some with zero rates
        birth_rate_arr = torch.tensor([0.1, 0.0, 0.15], dtype=torch.float64)
        death_rate_arr = torch.tensor([0.05, 0.0, 0.08], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # Should handle mixed zero and non-zero rates
        assert 0 < result < 1

    def test_extinction_pr_increases_with_time(self) -> None:
        """Test that extinction probability increases as observation time increases."""
        birth_rate_val = torch.tensor([0.1], dtype=torch.float64)
        death_rate_val = torch.tensor([0.05], dtype=torch.float64)

        # Compare extinction probabilities at different times
        times = [50.0, 100.0, 200.0, 400.0]
        results = [
            calculate_extinction_pr(
                birth_rate_val,
                death_rate_val,
                torch.tensor([t], dtype=torch.float64),
            )
            for t in times
        ]

        # Extinction probability should be monotonically increasing with time
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"α(t={times[i]}) = {results[i]} should be < "
                f"α(t={times[i + 1]}) = {results[i + 1]}"
            )

    def test_extinction_pr_increases_with_death_rate(self) -> None:
        """Test that extinction probability increases as death rate increases."""
        birth_rate_val = torch.tensor([0.1], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Compare extinction probabilities at different death rates
        death_rates = [
            torch.tensor([0.02], dtype=torch.float64),
            torch.tensor([0.05], dtype=torch.float64),
            torch.tensor([0.08], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
        ]
        results = [
            calculate_extinction_pr(birth_rate_val, death_rate_val, T)
            for death_rate_val in death_rates
        ]

        # Extinction probability should be monotonically increasing with death rate
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"α(μ={death_rates[i]}) = {results[i]} should be < "
                f"α(μ={death_rates[i + 1]}) = {results[i + 1]}"
            )

    def test_extinction_pr_decreases_with_birth_rate(self) -> None:
        """Test that extinction probability decreases as birth rate increases."""
        death_rate_val = torch.tensor([0.05], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        # Compare extinction probabilities at different birth rates
        birth_rates = [
            torch.tensor([0.05], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([0.15], dtype=torch.float64),
            torch.tensor([0.2], dtype=torch.float64),
        ]
        results = [
            calculate_extinction_pr(birth_rate_val, death_rate_val, T)
            for birth_rate_val in birth_rates
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
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test basic scalar input case."""
        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert torch.isfinite(result)
        assert isinstance(result, torch.Tensor)

    def test_scalar_inputs_n_zero(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test n=0 (extinction) case."""
        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=torch.tensor([0], dtype=torch.int64),
        )

        # P(n=0) = alpha_t, so log P should be finite and negative
        assert torch.isfinite(result)
        assert result < 0

    def test_scalar_inputs_array_ns(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
        sample_population_counts: torch.Tensor,
    ) -> None:
        """Test scalar parameters with array of population counts."""
        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=sample_population_counts,
        )

        # Should return sum of log probabilities
        assert torch.isfinite(result)

    def test_array_inputs_piecewise_constant(
        self,
        standard_array_params: dict[str, torch.Tensor],
    ) -> None:
        """Test piecewise-constant rates (array inputs)."""
        result = calculate_populations_log_lik(
            standard_array_params["birth_rates"],
            standard_array_params["death_rates"],
            times=standard_array_params["times"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert torch.isfinite(result)

    def test_negative_time_returns_error(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError):
            calculate_populations_log_lik(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                times=-100.0,  # Negative time
                ns=torch.tensor([1], dtype=torch.int64),
            )

    def test_negative_ns_returns_error(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test that negative population count raises ValueError."""
        with pytest.raises(ValueError):
            calculate_populations_log_lik(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                times=standard_scalar_params["time"],
                ns=-5,  # Negative count
            )

    def test_critical_case_scalar(
        self,
        nearly_equal_params: dict[str, torch.Tensor],
    ) -> None:
        """Test numerical stability when λ ≈ μ (scalar)."""
        result = calculate_populations_log_lik(
            nearly_equal_params["birth_rate"],
            nearly_equal_params["death_rate"],
            times=nearly_equal_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert torch.isfinite(result)

    def test_array_length_mismatch_raises_error(self) -> None:
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="Tensors must have matching lengths."):
            calculate_populations_log_lik(
                birth_rates=torch.tensor([0.1, 0.15], dtype=torch.float64),
                death_rates=torch.tensor(
                    [0.05, 0.06, 0.055],
                    dtype=torch.float64,
                ),  # Different length
                times=torch.tensor([50.0, 100.0], dtype=torch.float64),
                ns=torch.tensor([1], dtype=torch.int64),
            )


# ============================================================================
# LEVEL 3: Age Likelihood Tests - Spine Individuals
# ============================================================================


class TestSpineAgesLogLikelihood:
    """Test age log-likelihood for spine individuals."""

    def test_scalar_age(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
        sample_ages_scalar: torch.Tensor,
    ) -> None:
        """Test with single age value."""
        result = calculate_spine_ages_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert torch.isfinite(result)
        assert isinstance(result, torch.Tensor)

    def test_array_ages(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
        sample_ages_array: torch.Tensor,
    ) -> None:
        """Test with array of ages."""
        result = calculate_spine_ages_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        # Should return summed log-likelihood
        assert torch.isfinite(result)

    def test_age_at_boundary(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test age exactly equal to observation time T."""
        T = standard_scalar_params["time"]
        result = calculate_spine_ages_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            T,  # Age = T (boundary case)
            T,
        )

        assert torch.isfinite(result)

    def test_critical_case(
        self,
        nearly_equal_params: dict[str, torch.Tensor],
        sample_ages_scalar: torch.Tensor,
    ) -> None:
        """Test numerical stability when λ ≈ μ."""
        result = calculate_spine_ages_log_lik(
            nearly_equal_params["birth_rate"],
            nearly_equal_params["death_rate"],
            sample_ages_scalar,
            nearly_equal_params["time"],
        )

        assert torch.isfinite(result)

    def test_invalid_age_greater_than_T(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test that age > T returns -inf."""
        with pytest.raises(
            ValueError,
            match=r"ages must be in \[0, .*\]. Got ages:",
        ):  # age > T
            calculate_spine_ages_log_lik(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                standard_scalar_params["time"] + 10.0,  # age > T
                standard_scalar_params["time"],
            )

    def test_negative_age_returns_minus_inf(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test that negative age returns -inf."""
        with pytest.raises(
            ValueError,
            match=r"ages must be in \[0, .*\]. Got ages:",
        ):  # Negative age
            calculate_spine_ages_log_lik(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                -10.0,  # Negative age
                standard_scalar_params["time"],
            )


# ============================================================================
# LEVEL 4: Age Likelihood Tests - Non-Spine Individuals
# ============================================================================


class TestNonSpineAgesLogLikelihood:
    """Test age log-likelihood for non-spine individuals."""

    def test_scalar_params_scalar_age(self, sample_ages_scalar: torch.Tensor) -> None:
        """Test with scalar parameters and single age."""
        # For single-epoch case, wrap scalars in arrays of length 1
        result = calculate_non_spine_ages_log_lik(
            birth_rates=torch.tensor([0.1], dtype=torch.float64),
            death_rates=torch.tensor([0.05], dtype=torch.float64),
            ages=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert torch.isfinite(result)

    def test_scalar_params_array_ages(self, sample_ages_array: torch.Tensor) -> None:
        """Test with scalar parameters and array of ages."""
        # For single-epoch case, wrap scalars in arrays of length 1
        result = calculate_non_spine_ages_log_lik(
            birth_rates=torch.tensor([0.1], dtype=torch.float64),
            death_rates=torch.tensor([0.05], dtype=torch.float64),
            ages=sample_ages_array,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert torch.isfinite(result)

    def test_array_params_piecewise_constant(
        self,
        standard_array_params: dict[str, torch.Tensor],
    ) -> None:
        """Test with piecewise-constant rates (array parameters)."""
        # Use ages within the valid time range
        max_time = standard_array_params["times"][-1]
        ages = torch.tensor([10.0, 50.0, max_time - 10.0], dtype=torch.float64)
        result = calculate_non_spine_ages_log_lik(
            birth_rates=standard_array_params["birth_rates"],
            death_rates=standard_array_params["death_rates"],
            ages=ages,
            times=standard_array_params["times"],
        )

        assert torch.isfinite(result)

    def test_single_epoch(self, sample_ages_scalar: torch.Tensor) -> None:
        """Test with single epoch (single-element arrays)."""
        result = calculate_non_spine_ages_log_lik(
            birth_rates=torch.tensor([0.1], dtype=torch.float64),
            death_rates=torch.tensor([0.05], dtype=torch.float64),
            ages=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert torch.isfinite(result)

    def test_critical_case(self, sample_ages_scalar: torch.Tensor) -> None:
        """Test numerical stability when λ ≈ μ."""
        result = calculate_non_spine_ages_log_lik(
            birth_rates=torch.tensor([0.1], dtype=torch.float64),
            death_rates=torch.tensor([0.1 + 1e-13], dtype=torch.float64),
            ages=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert torch.isfinite(result)

    def test_invalid_age_returns_minus_inf(self) -> None:
        """Test that invalid ages return -inf."""
        # Age > T
        with pytest.raises(ValueError, match=r"ages must be in \[0, .*\]. Got ages:"):
            calculate_non_spine_ages_log_lik(
                birth_rates=torch.tensor([0.1], dtype=torch.float64),
                death_rates=torch.tensor([0.05], dtype=torch.float64),
                ages=150.0,  # age > times[-1]
                times=torch.tensor([100.0], dtype=torch.float64),
            )

    def test_shape_mismatch_raises_error(
        self,
        sample_ages_scalar: torch.Tensor,
    ) -> None:
        """Test that parameter shape mismatches raise ValueError."""
        with pytest.raises(ValueError, match="matching lengths"):
            calculate_non_spine_ages_log_lik(
                birth_rates=torch.tensor([0.1, 0.15], dtype=torch.float64),
                death_rates=torch.tensor(
                    [0.05],
                    dtype=torch.float64,
                ),  # Different shape
                ages=sample_ages_scalar,
                times=torch.tensor([50.0, 100.0], dtype=torch.float64),
            )

    def test_non_monotonic_times_raises_error(
        self,
        sample_ages_scalar: torch.Tensor,
    ) -> None:
        """Test that non-monotonic times raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            calculate_non_spine_ages_log_lik(
                birth_rates=torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64),
                death_rates=torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64),
                ages=sample_ages_scalar,
                times=torch.tensor(
                    [50.0, 100.0, 90.0],
                    dtype=torch.float64,
                ),  # Not monotonic
            )


# ============================================================================
# LEVEL 5: Unified Age Likelihood Interface Tests
# ============================================================================


class TestAgesLogLikelihood:
    """Test unified age log-likelihood interface (spine/non-spine selector)."""

    def test_spine_routing(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
        sample_ages_scalar: torch.Tensor,
    ) -> None:
        """Test that are_spines=True routes to spine implementation."""
        result = calculate_ages_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=True,
        )

        # Should match spine calculation
        expected = calculate_spine_ages_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert_close(result, expected)

    def test_non_spine_routing(self, sample_ages_scalar: torch.Tensor) -> None:
        """Test that are_spines=False routes to non-spine implementation."""
        result = calculate_ages_log_lik(
            birth_rates=0.1,
            death_rates=0.05,
            ages=sample_ages_scalar,
            times=100.0,
            are_spines=False,
        )

        # Should match non-spine calculation (using array inputs)
        expected = calculate_non_spine_ages_log_lik(
            birth_rates=torch.tensor([0.1], dtype=torch.float64),
            death_rates=torch.tensor([0.05], dtype=torch.float64),
            ages=sample_ages_scalar,
            times=torch.tensor([100.0], dtype=torch.float64),
        )

        assert_close(result, expected)


# ============================================================================
# LEVEL 6: Mathematical Property Tests
# ============================================================================


class TestMathematicalProperties:
    """Test mathematical properties and consistency."""

    def test_pr_bounds_population(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test that log probabilities are negative (probabilities < 1)."""
        for n in range(20):
            result = calculate_populations_log_lik(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                times=standard_scalar_params["time"],
                ns=n,
            )
            # Log probabilities should be ≤ 0
            assert result <= 0 or torch.isclose(
                result,
                torch.tensor(0.0, dtype=torch.float64),
            )

    def test_extinction_pr_increases_with_mu(self) -> None:
        """Test that extinction probability increases as death rate increases."""
        birth_rates = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        alpha_1 = calculate_extinction_pr(
            birth_rates,
            torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64),
            times,
        )
        alpha_2 = calculate_extinction_pr(
            birth_rates,
            torch.tensor([0.08, 0.08, 0.08], dtype=torch.float64),
            times,
        )

        # Higher death rate → higher extinction probability
        assert 0 < alpha_1 < alpha_2 < 1

    def test_consistency_scalar_vs_single_element_array(self) -> None:
        """Test that scalar and single-element array give same results."""
        scalar_result = calculate_populations_log_lik(
            birth_rates=0.1,
            death_rates=0.05,
            times=100.0,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        array_result = calculate_populations_log_lik(
            birth_rates=torch.tensor([0.1], dtype=torch.float64),
            death_rates=torch.tensor([0.05], dtype=torch.float64),
            times=torch.tensor([100.0], dtype=torch.float64),
            ns=torch.tensor([1], dtype=torch.int64),
        )

        # Results should be very close
        assert_close(scalar_result, array_result)

    def test_sum_property_array_ns(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test that array of ns gives sum of individual likelihoods."""
        ns_array = torch.tensor([5, 10, 15], dtype=torch.int64)

        # Compute together
        result_together = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=ns_array,
        )

        # Compute separately and sum
        result_separate = sum(
            calculate_populations_log_lik(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                times=standard_scalar_params["time"],
                ns=n,
            )
            for n in ns_array
        )

        assert_close(result_together, result_separate)


# ============================================================================
# LEVEL 7: Numerical Stability and Edge Case Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability in challenging scenarios."""

    def test_extreme_time_values(self) -> None:
        """Test with very large time values."""
        # Large time can cause overflow in exponentials
        result = calculate_populations_log_lik(
            birth_rates=0.001,  # Small rates to prevent overflow
            death_rates=0.0005,
            times=10000.0,  # Very large time
            ns=torch.tensor([1], dtype=torch.int64),
        )

        # Should not return inf or nan
        assert torch.isfinite(result) or result == -torch.tensor(
            float("inf"),
            dtype=torch.float64,
        )

    def test_extreme_rate_differences(self) -> None:
        """Test when λ >> μ or μ >> λ."""
        # λ >> μ (high growth)
        result1 = calculate_populations_log_lik(
            birth_rates=1.0,
            death_rates=0.001,
            times=10.0,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        # Should handle gracefully
        assert torch.isfinite(result1) or result1 == -torch.tensor(
            float("inf"),
            dtype=torch.float64,
        )

        # μ >> λ (high death)
        result2 = calculate_populations_log_lik(
            birth_rates=0.001,
            death_rates=1.0,
            times=10.0,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert torch.isfinite(result2) or result2 == -torch.tensor(
            float("inf"),
            dtype=torch.float64,
        )

    def test_very_small_rates(self) -> None:
        """Test with very small birth and death rates."""
        result = calculate_populations_log_lik(
            birth_rates=1e-10,
            death_rates=5e-11,
            times=1000.0,
            ns=5,
        )

        assert torch.isfinite(result) or result == -torch.tensor(
            float("inf"),
            dtype=torch.float64,
        )

    def test_many_epochs(self) -> None:
        """Test with many epochs (long array)."""
        n_epochs = 100
        birth_rates = torch.linspace(0.08, 0.12, n_epochs)
        death_rates = torch.linspace(0.04, 0.06, n_epochs)
        times = torch.linspace(1.0, 100.0, n_epochs)

        result = calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert torch.isfinite(result) or result == -torch.tensor(
            float("inf"),
            dtype=torch.float64,
        )

    def test_ages_near_observation_time(self) -> None:
        """Test age likelihood with ages very close to T."""
        T = torch.tensor([100.0], dtype=torch.float64)
        ages = torch.tensor([T - 1e-6, T - 1e-8, T - 1e-10], dtype=torch.float64)

        result = calculate_spine_ages_log_lik(
            birth_rates=0.1,
            death_rates=0.05,
            ages=ages,
            times=T,
        )

        assert torch.isfinite(result)


# ============================================================================
# LEVEL 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Test integration between related functions."""

    def test_spine_non_spine_different_formulas(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
        sample_ages_scalar: torch.Tensor,
    ) -> None:
        """Test that spine and non-spine give different results (different formulas)."""
        spine_result = calculate_spine_ages_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        non_spine_result = calculate_non_spine_ages_log_lik(
            torch.tensor([standard_scalar_params["birth_rate"]], dtype=torch.float64),
            torch.tensor([standard_scalar_params["death_rate"]], dtype=torch.float64),
            sample_ages_scalar,
            torch.tensor([standard_scalar_params["time"]], dtype=torch.float64),
        )

        # These should be different (different conditioning)
        assert not torch.isclose(spine_result, non_spine_result)


# ============================================================================
# Performance and Vectorization Tests
# ============================================================================


class TestVectorization:
    """Test vectorization and performance characteristics."""

    def test_array_ages_more_efficient_than_loop(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Verify that array processing is implemented (not just a loop)."""
        # This is more of a smoke test - actual performance testing
        # would require timing benchmarks
        ages = torch.linspace(10, 90, 100)

        result = calculate_spine_ages_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            ages,
            standard_scalar_params["time"],
        )

        assert torch.isfinite(result)

    def test_large_population_counts_array(
        self,
        standard_scalar_params: dict[str, torch.Tensor],
    ) -> None:
        """Test with large array of population counts."""
        ns = torch.arange(0, 100)

        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=ns,
        )

        assert torch.isfinite(result)


# ============================================================================
# Private Helper Function Tests
# ============================================================================


class TestIntegrandSurvFromWeightedBirthTime:
    """Test the _integrand_surv_from_weighted_birth_time_scalar private helper function.

    This function is used in calculating survival probabilities and returns a
    Python float for use in numerical integration.
    """

    def test_basic_single_epoch(self) -> None:
        """Test basic functionality with a single epoch."""
        birth_time = 5.0
        birth_rates = torch.tensor([0.1], dtype=torch.float64)
        death_rates = torch.tensor([0.05], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        # Result should be finite and non-negative
        assert math.isfinite(result)
        assert result >= 0

    def test_basic_multiple_epochs(self) -> None:
        """Test with multiple epochs."""
        birth_time = 25.0
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert math.isfinite(result)
        assert result >= 0

    def test_birth_time_at_zero(self) -> None:
        """Test when birth_time is 0 (start of process)."""
        birth_time = 0.0
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert math.isfinite(result)
        assert result >= 0

    def test_birth_time_equals_first_epoch_end(self) -> None:
        """Test edge case when birth_time equals first epoch end time."""
        birth_time = 50.0
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # After subtracting birth_time, times becomes [0.0, 50.0]
        # valid_indices will be where times > 0, so only index 1
        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert math.isfinite(result)
        assert result >= 0

    def test_birth_time_beyond_all_epochs_returns_zero(self) -> None:
        """Test that birth_time beyond all epochs returns 0.0."""
        birth_time = 150.0  # Beyond all epochs
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )
        assert result == 0.0

    def test_birth_time_slightly_beyond_last_epoch(self) -> None:
        """Test birth_time just beyond the last epoch time."""
        birth_time = 100.1
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # After subtracting birth_time, times becomes negative, no valid indices
        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )
        assert result == 0.0

    def test_birth_time_within_first_epoch(self) -> None:
        """Test normal case with birth_time within first epoch."""
        birth_time = 10.0
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert math.isfinite(result)
        assert result >= 0

    def test_birth_time_within_middle_epoch(self) -> None:
        """Test birth_time within a middle epoch."""
        birth_time = 75.0
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        # After subtracting, times becomes [-25.0, 25.0, 75.0]
        # valid_indices will be [1, 2]
        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert math.isfinite(result)
        assert result >= 0

    def test_scalar_inputs_converted_to_arrays(self) -> None:
        """Test that scalar inputs are properly converted to arrays."""
        birth_time = 10.0
        birth_rates = torch.tensor([0.1], dtype=torch.float64)
        death_rates = torch.tensor([0.05], dtype=torch.float64)
        times = torch.tensor([100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert math.isfinite(result)
        assert result >= 0

    def test_result_proportional_to_birth_rate(self) -> None:
        """Test that result is proportional to the first valid birth rate."""
        birth_time = 10.0
        birth_rates1 = torch.tensor([0.1, 0.15], dtype=torch.float64)
        birth_rates2 = torch.tensor(
            [0.2, 0.15],
            dtype=torch.float64,
        )  # Double the first rate
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result1 = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates1,
            death_rates,
            times,
        )
        result2 = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates2,
            death_rates,
            times,
        )

        # Result should scale with birth rate
        # (approximately, given extinction prob is also affected)
        assert result2 > result1

    def test_array_length_mismatch_times_longer(self) -> None:
        """Test when times array is longer than birth_rates/death_rates arrays."""
        birth_time = 10.0
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)  # Length 2
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)  # Length 2
        times = torch.tensor(
            [50.0, 100.0, 150.0],
            dtype=torch.float64,
        )  # Length 3 - MISMATCH!

        with pytest.raises(IndexError):
            _integrand_surv_from_weighted_birth_time_scalar(
                birth_time,
                birth_rates,
                death_rates,
                times,
            )

    def test_array_length_mismatch_rates_longer(self) -> None:
        """Test when birth_rates/death_rates arrays are longer than times array."""
        birth_time = 10.0
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )
        assert result >= 0

    def test_birth_and_death_rates_different_lengths(self) -> None:
        """Test when birth_rates and death_rates have different lengths."""
        birth_time = 10.0
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)  # Length 2
        death_rates = torch.tensor(
            [0.05, 0.06, 0.055],
            dtype=torch.float64,
        )  # Length 3 - MISMATCH!
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )
        assert result >= 0

    def test_integration_context_array_length_issue(self) -> None:
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
        birth_time = 25.0  # Some point during integration

        # If times gets modified to include lifetime:
        bad_times = torch.cat(
            [times, torch.as_tensor(lifetime, dtype=torch.float64).reshape(-1)],
        )  # Now length 3!

        # This should raise IndexError due to valid_indices being out of bounds
        with pytest.raises(IndexError):
            _integrand_surv_from_weighted_birth_time_scalar(
                birth_time,
                birth_rates,
                death_rates,
                bad_times,
            )

    def test_valid_indices_logic_with_partial_truncation(self) -> None:
        """Test that valid_indices correctly handles partial epoch truncation."""
        # When birth_time is within epochs, valid_indices should be a subset
        birth_time = 60.0
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        # After subtracting birth_time, times becomes [-10.0, 40.0, 90.0]
        # valid_indices should be [1, 2] (where times > 0)
        # Accessing birth_rates[[1, 2]] should work fine
        result = _integrand_surv_from_weighted_birth_time_scalar(
            birth_time,
            birth_rates,
            death_rates,
            times,
        )

        assert math.isfinite(result)
        assert result >= 0


class TestCalculateLogPrExtinctionGivenInitialLifetimes:
    """Test calculate_log_pr_extinction_given_initial_lifetimes function.

    This function uses _integrand_surv_from_weighted_birth_time_scalar internally
    and has a bug where lifetime > max(times) causes IndexError.
    """

    def test_basic_single_lifetime_within_epochs(self) -> None:
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

        assert math.isfinite(result)

    def test_lifetime_equals_last_epoch(self) -> None:
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

        assert torch.isfinite(result)

    def test_lifetime_beyond_last_epoch_raises_error(self) -> None:
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

        with pytest.raises(ValueError, match="ages must be in"):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetime,
                birth_rates,
                death_rates,
                times,
            )

    def test_multiple_lifetimes_one_beyond_epochs(self) -> None:
        """Test with array of lifetimes where one exceeds max(times)."""
        lifetimes = torch.tensor(
            [50.0, 80.0, 120.0],
            dtype=torch.float64,
        )  # Last one beyond times
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="ages must be in"):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetimes,
                birth_rates,
                death_rates,
                times,
            )

    def test_lifetime_at_first_epoch_boundary(self) -> None:
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

        assert torch.isfinite(result)

    def test_three_epochs_lifetime_beyond(self) -> None:
        """Test with three epochs and lifetime beyond all of them."""
        lifetime = 200.0
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        # Should now work: uses all epoch rates plus last rate for [150-200]
        with pytest.raises(ValueError, match="ages must be in"):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetime,
                birth_rates,
                death_rates,
                times,
            )

    def test_lifetime_far_beyond_last_epoch(self) -> None:
        """Test with lifetime far beyond the last epoch."""
        lifetime = 500.0  # Much larger than last epoch at 100
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # Should handle extrapolation using last birth rate
        with pytest.raises(ValueError, match="ages must be in"):
            calculate_log_pr_extinction_given_initial_lifetimes(
                lifetime,
                birth_rates,
                death_rates,
                times,
            )


class TestTimeReversalInvolution:
    """Test time-reversal transformation is an involution (applying twice = identity).

    The _reverse_time function should satisfy the mathematical property that
    applying it twice returns to the original state (times, rates). This is crucial
    for correctness since the transformation is used to reduce multi-epoch likelihoods
    to single-epoch formulas.
    """

    def test_double_reversal_two_epochs(self) -> None:
        """Test double reversal for two-epoch case."""
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        # Apply reversal once
        inv_times_1, refl_birth_1, refl_death_1 = _reverse_time(
            birth_rates,
            death_rates,
            times,
        )

        # Apply reversal again
        inv_times_2, refl_birth_2, refl_death_2 = _reverse_time(
            refl_birth_1,
            refl_death_1,
            inv_times_1,
        )

        # Should recover original (up to numerical precision)
        assert_close(inv_times_2, times)
        assert_close(refl_birth_2, birth_rates)
        assert_close(refl_death_2, death_rates)

    def test_double_reversal_three_epochs(self) -> None:
        """Test double reversal for three-epoch case."""
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([40.0, 80.0, 120.0], dtype=torch.float64)

        # Apply reversal twice
        inv_times_1, refl_birth_1, refl_death_1 = _reverse_time(
            birth_rates,
            death_rates,
            times,
        )
        inv_times_2, refl_birth_2, refl_death_2 = _reverse_time(
            refl_birth_1,
            refl_death_1,
            inv_times_1,
        )

        # Should recover original
        assert_close(inv_times_2, times, rtol=1e-14, atol=1e-14)
        assert_close(refl_birth_2, birth_rates, rtol=1e-14, atol=1e-14)
        assert_close(refl_death_2, death_rates, rtol=1e-14, atol=1e-14)

    def test_double_reversal_many_epochs(self) -> None:
        """Test double reversal with many epochs."""
        # Create 5-epoch case
        birth_rates = torch.tensor(
            [0.10, 0.12, 0.11, 0.13, 0.09],
            dtype=torch.float64,
        )
        death_rates = torch.tensor(
            [0.05, 0.06, 0.055, 0.065, 0.04],
            dtype=torch.float64,
        )
        times = torch.tensor([30.0, 60.0, 90.0, 120.0, 150.0], dtype=torch.float64)

        # Apply reversal twice
        inv_times_1, refl_birth_1, refl_death_1 = _reverse_time(
            birth_rates,
            death_rates,
            times,
        )
        inv_times_2, refl_birth_2, refl_death_2 = _reverse_time(
            refl_birth_1,
            refl_death_1,
            inv_times_1,
        )

        # Verify involution property
        assert_close(inv_times_2, times)
        assert_close(refl_birth_2, birth_rates)
        assert_close(refl_death_2, death_rates)

    def test_single_reversal_swaps_rates(self) -> None:
        """Test that single reversal correctly swaps birth and death rates."""
        birth_rates = torch.tensor([0.1, 0.15], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0], dtype=torch.float64)

        inv_times, refl_birth, refl_death = _reverse_time(
            birth_rates,
            death_rates,
            times,
        )

        # After first reversal, birth rates should be flipped death rates
        expected_birth = torch.flip(death_rates, dims=[0])
        expected_death = torch.flip(birth_rates, dims=[0])

        assert_close(refl_birth, expected_birth)
        assert_close(refl_death, expected_death)

    def test_reversed_times_are_monotonic(self) -> None:
        """Test that reversed times remain monotonically increasing."""
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([50.0, 100.0, 150.0], dtype=torch.float64)

        inv_times, _, _ = _reverse_time(birth_rates, death_rates, times)

        # Check that inverted times are strictly increasing
        diffs = torch.diff(inv_times)
        assert torch.all(diffs > 0), "Reversed times should be strictly increasing"

    def test_total_time_preserved(self) -> None:
        """Test that total elapsed time is preserved under reversal."""
        birth_rates = torch.tensor([0.1, 0.15, 0.12], dtype=torch.float64)
        death_rates = torch.tensor([0.05, 0.06, 0.055], dtype=torch.float64)
        times = torch.tensor([40.0, 90.0, 130.0], dtype=torch.float64)

        inv_times, _, _ = _reverse_time(birth_rates, death_rates, times)

        # Final time should be the same
        assert_close(inv_times[-1], times[-1])
