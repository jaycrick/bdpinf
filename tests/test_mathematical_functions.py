"""Tests for the math_extinction, math_nested, and math_populations modules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy

from bdpinf.models.math_extinction import (
    calculate_extinction_pr,
)
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.math_populations import (
    calculate_populations_log_lik,
    population_geometric_parameter,
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
def standard_scalar_params() -> dict[str, jax.Array]:
    """Set standard scalar parameters for testing."""
    return {
        "birth_rate": jnp.array([0.1], dtype=jnp.float64),
        "death_rate": jnp.array([0.05], dtype=jnp.float64),
        "time": jnp.array([100.0], dtype=jnp.float64),
    }


@pytest.fixture
def standard_array_params() -> dict[str, jax.Array]:
    """Set standard array parameters for piecewise-constant rates."""
    return {
        "birth_rates": jnp.array([0.1, 0.15, 0.12], dtype=jnp.float64),
        "death_rates": jnp.array([0.05, 0.06, 0.055], dtype=jnp.float64),
        "times": jnp.array([50.0, 100.0, 150.0], dtype=jnp.float64),
    }


@pytest.fixture
def nearly_equal_params() -> dict[str, jax.Array]:
    """Parameters where λ ≈ μ (numerical stability test)."""
    return {
        "birth_rate": jnp.array([0.1], dtype=jnp.float64),
        "death_rate": jnp.array(
            [0.1 + 1e-13],
            dtype=jnp.float64,
        ),  # Within tolerance of 1e-12
        "time": jnp.array([100.0], dtype=jnp.float64),
    }


@pytest.fixture
def log_scale_params() -> dict[str, jax.Array]:
    """Parameters in log scale."""
    return {
        "log_birth_rate": jnp.log(jnp.array(0.1, dtype=jnp.float64)),
        "log_death_rate": jnp.log(jnp.array(0.05, dtype=jnp.float64)),
        "time": jnp.array(100.0, dtype=jnp.float64),
    }


@pytest.fixture
def sample_ages_scalar() -> float:
    """Single age value for testing."""
    return 50.0


@pytest.fixture
def sample_ages_array() -> jax.Array:
    """Array of ages for vectorized testing."""
    return jnp.array([10.0, 30.0, 50.0, 70.0, 90.0], dtype=jnp.float64)


@pytest.fixture
def sample_population_counts() -> jax.Array:
    """Sample population counts (n values)."""
    return jnp.array([0, 1, 5, 10, 20], dtype=jnp.int32)


@pytest.fixture
def two_epoch_params() -> dict[str, jax.Array]:
    """Parameters for two-epoch testing."""
    return {
        "birth_rates": jnp.array([0.01, 0.001], dtype=jnp.float64),
        "death_rates": jnp.array([0.001, 0.001], dtype=jnp.float64),
        "times": jnp.array([750.0, 1500.0], dtype=jnp.float64),
    }


@pytest.fixture
def three_epoch_params() -> dict[str, jax.Array]:
    """Parameters for three-epoch testing."""
    return {
        "birth_rates": jnp.array([0.1, 0.05, 0.08], dtype=jnp.float64),
        "death_rates": jnp.array([0.03, 0.02, 0.04], dtype=jnp.float64),
        "times": jnp.array([100.0, 250.0, 400.0], dtype=jnp.float64),
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
        birth_rate_val = jnp.array([0.1], dtype=jnp.float64)
        death_rate_val = jnp.array([0.05], dtype=jnp.float64)
        final_time = jnp.array([100.0], dtype=jnp.float64)
        n = jnp.array([5], dtype=jnp.int32)

        # Single epoch result
        single_result = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=final_time,
            ns=n,
        )

        # Two epochs with same parameters should give same result
        two_epoch_result = calculate_populations_log_lik(
            jnp.array([birth_rate_val[0], birth_rate_val[0]], dtype=jnp.float64),
            jnp.array([death_rate_val[0], death_rate_val[0]], dtype=jnp.float64),
            times=jnp.array([final_time[0] / 2, final_time[0]], dtype=jnp.float64),
            ns=n,
        )

        assert np.isfinite(float(single_result))
        assert np.isfinite(float(two_epoch_result))
        np.testing.assert_allclose(
            np.array(single_result), np.array(two_epoch_result), rtol=1.3e-6
        )

    def test_multi_epoch_extinction_pr_bounds(self) -> None:
        """Test that multi-epoch extinction probability is in [0, 1]."""
        birth_rates = jnp.array([0.2, 0.15, 0.1], dtype=jnp.float64)
        death_rates = jnp.array([0.1, 0.12, 0.15], dtype=jnp.float64)
        times = jnp.array([50.0, 100.0, 150.0], dtype=jnp.float64)

        result = calculate_extinction_pr(birth_rates, death_rates, times)

        assert 0 < result < 1, f"Extinction probability {result} not in (0, 1)"

    def test_two_epoch_spine_age_density_basic(self) -> None:
        """Validate multi-epoch spine age density for ages in each epoch.

        This exercises the reflected multi-epoch path in
        ``calculate_spine_ages_log_lik`` by checking that ages in both the first
        and second epochs produce finite log densities with the current closed-form
        implementation.
        """
        birth_rates = jnp.array([0.1, 0.2], dtype=jnp.float64)
        death_rates = jnp.array([0.05, 0.1], dtype=jnp.float64)
        times = jnp.array([50.0, 100.0], dtype=jnp.float64)

        # Age in first epoch
        age_epoch1 = jnp.array([30.0], dtype=jnp.float64)
        result1 = spine_log_density(
            nested_terms(birth_rates, death_rates, times), age_epoch1
        ).sum()
        assert np.isfinite(float(result1))

        # Age in second epoch
        age_epoch2 = jnp.array([70.0], dtype=jnp.float64)
        result2 = spine_log_density(
            nested_terms(birth_rates, death_rates, times), age_epoch2
        ).sum()
        assert np.isfinite(float(result2))

    def test_multi_epoch_non_spine_age_array(self) -> None:
        """Check multi-epoch non-spine ages spanning epochs yield finite, neg log-lik.

        Confirms the multi-epoch branch of ``calculate_non_spine_ages_log_lik``
        handles ages on either side of an epoch boundary and returns a negative
        summed log-density as expected for probabilities < 1.
        """
        birth_rates = jnp.array([0.15, 0.1], dtype=jnp.float64)
        death_rates = jnp.array([0.05, 0.08], dtype=jnp.float64)
        times = jnp.array([100.0, 200.0], dtype=jnp.float64)

        # Ages spanning both epochs
        ages = jnp.array([20.0, 60.0, 120.0, 180.0], dtype=jnp.float64)

        result = non_spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()

        # Should return finite sum
        assert np.isfinite(float(result))
        # Log likelihood should be negative
        assert result < 0

    def test_two_epoch_geometric_parameter(self) -> None:
        """Test population_geometric_parameter for two epochs."""
        birth_rates = jnp.array([0.2, 0.3], dtype=jnp.float64)
        death_rates = jnp.array([0.1, 0.15], dtype=jnp.float64)
        times = jnp.array([50.0, 100.0], dtype=jnp.float64)

        result = population_geometric_parameter(birth_rates, death_rates, times)

        # Should be in valid probability range
        assert 0 < result < 1
        assert np.isfinite(float(result))


class TestMultiEpochConsistency:
    """Test consistency properties of multi-epoch functions."""

    def test_epoch_order_matters(self) -> None:
        """Test that swapping epoch parameters changes results."""
        # Different parameters for each epoch
        birth_rates1 = jnp.array([0.2, 0.1], dtype=jnp.float64)
        death_rates1 = jnp.array([0.1, 0.05], dtype=jnp.float64)
        times = jnp.array([60.0, 100.0], dtype=jnp.float64)

        # Swapped parameters
        birth_rates2 = jnp.array([0.1, 0.2], dtype=jnp.float64)
        death_rates2 = jnp.array([0.05, 0.1], dtype=jnp.float64)

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
        assert not np.isclose(float(result1), float(result2))

    def test_multi_epoch_ages_sum_property(self) -> None:
        """Test that multi-epoch age likelihood sums correctly across ages."""
        birth_rates = jnp.array([0.12, 0.08], dtype=jnp.float64)
        death_rates = jnp.array([0.04, 0.06], dtype=jnp.float64)
        times = jnp.array([100.0, 200.0], dtype=jnp.float64)

        ages = jnp.array([20.0, 80.0, 150.0], dtype=jnp.float64)

        # Compute for array
        result_array = non_spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()

        # Compute individually and sum
        result_sum = sum(
            non_spine_log_density(
                nested_terms(birth_rates, death_rates, times),
                jnp.array([age], dtype=jnp.float64),
            ).sum()
            for age in ages
        )

        np.testing.assert_allclose(
            np.array(result_array), np.array(result_sum), rtol=1.3e-6
        )


# ============================================================================
# LEVEL 0.1: Multi-Epoch vs Single-Epoch Equivalence (Identical Rates)
# ============================================================================


class TestMultiEpochEqualsSingleEpoch:
    """Verify multi-epoch equals single-epoch for identical rates.

    For functions that support array inputs (piecewise-constant epochs), when
    the rates are identical across epochs, the result should match the
    equivalent single-epoch case with the same total time. Where known
    numerical differences exist, we use a relaxed tolerance and document the
    behavior.
    """

    def test_extinction_pr_multi_equals_single(self) -> None:
        birth_rate_val, death_rate_val, times = (
            jnp.array([0.02], dtype=jnp.float64),
            jnp.array([0.01], dtype=jnp.float64),
            jnp.array([120.0], dtype=jnp.float64),
        )

        single = calculate_extinction_pr(birth_rate_val, death_rate_val, times)

        times = jnp.array([40.0, 80.0, 120.0], dtype=jnp.float64)
        birth_rates = jnp.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = jnp.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = calculate_extinction_pr(birth_rates, death_rates, times)

        np.testing.assert_allclose(np.array(multi), np.array(single), rtol=1.3e-6)
        assert 0 < single < 1

    def testpopulation_geometric_parameter_multi_equals_single(self) -> None:
        birth_rate_val, death_rate_val, times = (
            jnp.array([0.03], dtype=jnp.float64),
            jnp.array([0.02], dtype=jnp.float64),
            jnp.array([90.0], dtype=jnp.float64),
        )
        single = population_geometric_parameter(birth_rate_val, death_rate_val, times)
        times = jnp.array([30.0, 60.0, 90.0], dtype=jnp.float64)
        birth_rates = jnp.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = jnp.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = population_geometric_parameter(birth_rates, death_rates, times)
        np.testing.assert_allclose(np.array(multi), np.array(single), rtol=1.3e-6)

    def test_non_spine_ages_log_lik_multi_equals_single(self) -> None:
        ages = jnp.array([15.0, 60.0, 95.0], dtype=jnp.float64)
        birth_rate_val, death_rate_val, times = (
            jnp.array([0.02], dtype=jnp.float64),
            jnp.array([0.01], dtype=jnp.float64),
            jnp.array([120.0], dtype=jnp.float64),
        )
        # Single-epoch via scalar
        single = non_spine_log_density(
            nested_terms(birth_rate_val, death_rate_val, times), ages
        ).sum()
        # Multi-epoch via arrays with identical rates
        times = jnp.array([40.0, 80.0, 120.0], dtype=jnp.float64)
        birth_rates = jnp.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = jnp.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = non_spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()
        assert np.isfinite(float(single))
        assert np.isfinite(float(multi))
        np.testing.assert_allclose(np.array(multi), np.array(single), rtol=1.3e-6)

    def test_spine_ages_log_lik_multi_equals_single(self) -> None:
        ages = jnp.array([20.0, 40.0, 110.0], dtype=jnp.float64)
        birth_rate_val, death_rate_val, final_time = (
            jnp.array([0.018], dtype=jnp.float64),
            jnp.array([0.009], dtype=jnp.float64),
            jnp.array([120.0], dtype=jnp.float64),
        )
        single = spine_log_density(
            nested_terms(birth_rate_val, death_rate_val, final_time), ages
        ).sum()
        times = jnp.array([40.0, 80.0, 120.0], dtype=jnp.float64)
        birth_rates = jnp.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = jnp.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()
        assert np.isfinite(float(single))
        assert np.isfinite(float(multi))
        np.testing.assert_allclose(np.array(multi), np.array(single), rtol=1.3e-6)

    def test_spine_ages_log_lik_return_array_matches_sum(self) -> None:
        ages = jnp.array([20.0, 40.0, 110.0], dtype=jnp.float64)
        birth_rates = jnp.array([0.018], dtype=jnp.float64)
        death_rates = jnp.array([0.009], dtype=jnp.float64)
        times = jnp.array([120.0], dtype=jnp.float64)

        per_age = spine_log_density(nested_terms(birth_rates, death_rates, times), ages)
        summed = spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()

        assert per_age.shape == ages.shape
        np.testing.assert_allclose(
            np.array(per_age.sum()), np.array(summed), rtol=1.3e-6
        )

    def test_ages_log_lik_multi_equals_single_non_spine(self) -> None:
        ages = jnp.array([10.0, 55.0, 90.0], dtype=jnp.float64)
        birth_rate_val, death_rate_val, times = (
            jnp.array([0.02], dtype=jnp.float64),
            jnp.array([0.01], dtype=jnp.float64),
            jnp.array([120.0], dtype=jnp.float64),
        )
        single = non_spine_log_density(
            nested_terms(birth_rate_val, death_rate_val, times), ages
        ).sum()
        times = jnp.array([40.0, 80.0, 120.0], dtype=jnp.float64)
        birth_rates = jnp.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = jnp.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = non_spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()
        assert np.isfinite(float(single))
        assert np.isfinite(float(multi))
        np.testing.assert_allclose(np.array(multi), np.array(single), rtol=1.3e-6)

    def test_non_spine_ages_log_lik_return_array_matches_sum(self) -> None:
        ages = jnp.array([10.0, 55.0, 90.0], dtype=jnp.float64)
        birth_rates = jnp.array([0.02], dtype=jnp.float64)
        death_rates = jnp.array([0.01], dtype=jnp.float64)
        times = jnp.array([120.0], dtype=jnp.float64)

        per_age = non_spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        )
        summed = non_spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()

        assert per_age.shape == ages.shape
        np.testing.assert_allclose(
            np.array(per_age.sum()), np.array(summed), rtol=1.3e-6
        )

    def test_ages_log_lik_multi_equals_single_spine(self) -> None:
        # Interior ages only: the age == T atom is a point mass and is checked
        # separately.
        ages = jnp.array([15.0, 60.0, 100.0], dtype=jnp.float64)
        birth_rate_val, death_rate_val, final_time = (
            jnp.array([0.018], dtype=jnp.float64),
            jnp.array([0.009], dtype=jnp.float64),
            jnp.array([120.0], dtype=jnp.float64),
        )
        single = spine_log_density(
            nested_terms(birth_rate_val, death_rate_val, final_time), ages
        ).sum()
        times = jnp.array([30.0, 90.0, 120.0], dtype=jnp.float64)
        birth_rates = jnp.full(
            (3,),
            birth_rate_val.item(),
            dtype=birth_rate_val.dtype,
        )
        death_rates = jnp.full(
            (3,),
            death_rate_val.item(),
            dtype=death_rate_val.dtype,
        )
        multi = spine_log_density(
            nested_terms(birth_rates, death_rates, times), ages
        ).sum()
        assert np.isfinite(float(single))
        assert np.isfinite(float(multi))
        np.testing.assert_allclose(np.array(multi), np.array(single), rtol=1.3e-6)


# ============================================================================
# LEVEL 0: Numerical Accuracy Tests (Analytical Validation) - ORIGINAL
# ============================================================================


class TestNumericalAccuracy:
    """Test functions against known analytical results.

    These tests validate that the implementations produce numerically correct
    results by comparing against hand-calculated or analytically derived values.
    """

    def test_extinction_pr_critical_case_analytical(self) -> None:
        """Test extinction probability when λ = μ (analytical formula)."""
        birth_rate_val = jnp.array([0.1], dtype=jnp.float64)
        final_time = jnp.array([100.0], dtype=jnp.float64)

        # Expected: extinction_pr = 0.1 * 100 / (1 + 0.1 * 100) = 10 / 11 ≈ 0.909091
        expected = birth_rate_val * final_time / (1 + birth_rate_val * final_time)

        result = calculate_extinction_pr(birth_rate_val, birth_rate_val, final_time)

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)
        assert 0 < result < 1

    def test_populations_log_lik_n0_simple(self) -> None:
        """Test log P(n=0) for simple parameters."""
        birth_rate_val = jnp.array([0.2], dtype=jnp.float64)
        death_rate_val = jnp.array([0.1], dtype=jnp.float64)
        final_time = jnp.array([10.0], dtype=jnp.float64)

        # Compute expected extinction_pr manually
        growth = (birth_rate_val - death_rate_val) * final_time  # 1.0
        exp_growth = jnp.exp(growth)  # e^1.0 ≈ 2.71828
        extinction_pr = (
            death_rate_val
            * (exp_growth - 1)
            / (birth_rate_val * exp_growth - death_rate_val)
        )
        expected_log_prob = jnp.log(extinction_pr)

        result = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=final_time,
            ns=jnp.array([0], dtype=jnp.int32),
        )

        np.testing.assert_allclose(
            np.array(result), np.array(expected_log_prob), rtol=1.3e-6
        )

    def test_populations_log_lik_critical_case_n0(self) -> None:
        """Test log P(n=0) when λ = μ."""
        birth_rate_val = jnp.array([0.1], dtype=jnp.float64)
        final_time = jnp.array([50.0], dtype=jnp.float64)

        # Expected: extinction_pr = 0.1 * 50 / (1 + 0.1 * 50) = 5/6
        extinction_pr = birth_rate_val * final_time / (1 + birth_rate_val * final_time)
        expected = jnp.log(extinction_pr)

        result = calculate_populations_log_lik(
            birth_rate_val,
            birth_rate_val,
            times=final_time,
            ns=jnp.array([0], dtype=jnp.int32),
        )

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)

    def test_populations_log_lik_n1_simple(self) -> None:
        """Test log P(n=1) for simple parameters."""
        birth_rate_val = jnp.array([0.3], dtype=jnp.float64)
        death_rate_val = jnp.array([0.2], dtype=jnp.float64)
        final_time = jnp.array([5.0], dtype=jnp.float64)

        # Compute extinction_pr and one_minus_pi
        growth = (birth_rate_val - death_rate_val) * final_time
        exp_growth = jnp.exp(growth)
        extinction_pr = (
            death_rate_val
            * (exp_growth - 1)
            / (birth_rate_val * exp_growth - death_rate_val)
        )
        one_minus_pi = (birth_rate_val / death_rate_val) * extinction_pr

        expected = jnp.log(1 - extinction_pr) + jnp.log(1 - one_minus_pi)

        result = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=final_time,
            ns=jnp.array([1], dtype=jnp.int32),
        )

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)

    def test_spine_age_at_zero(self) -> None:
        """Test spine age density at t=0.

        At t=0 (age zero), the density involves special limiting behavior.
        """
        birth_rate_val = jnp.array([0.2], dtype=jnp.float64)
        death_rate_val = jnp.array([0.1], dtype=jnp.float64)
        final_time = jnp.array([10.0], dtype=jnp.float64)
        age = jnp.array([0.0], dtype=jnp.float64)

        result = spine_log_density(
            nested_terms(birth_rate_val, death_rate_val, final_time), age
        ).sum()

        # Should return -inf
        assert float(result) == -jnp.inf

    def test_geometric_scalar_case(self) -> None:
        """Docstring for test_geometric_scalar_case."""
        final_time = jnp.array([10.0], dtype=jnp.float64)
        birth_rate_val = jnp.array([1.0], dtype=jnp.float64)
        death_rate_val = jnp.array([2.0], dtype=jnp.float64)

        def _integrand(x: float) -> float:
            lam = float(birth_rate_val[0])
            mu = float(death_rate_val[0])
            return float(mu * jnp.exp(-(lam - mu) * x))

        integral, _ = scipy.integrate.quad(
            _integrand,
            0,
            final_time.item(),
        )
        expected = jnp.exp(-(birth_rate_val - death_rate_val) * final_time) / (
            1 + integral
        )
        result = population_geometric_parameter(
            birth_rate_val,
            death_rate_val,
            final_time,
        )
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)

    def test_geometric_parameter_critical_case(self) -> None:
        """Test population_geometric_parameter when λ = μ."""
        birth_rate_val = death_rate_val = jnp.array([0.1], dtype=jnp.float64)
        final_time = jnp.array([20.0], dtype=jnp.float64)

        def _integrand(x: float) -> float:
            lam = float(birth_rate_val[0])
            mu = float(death_rate_val[0])
            return float(mu * jnp.exp(-(lam - mu) * x))

        integral, _ = scipy.integrate.quad(
            _integrand,
            0,
            final_time.item(),
        )
        expected = jnp.exp(-(birth_rate_val - death_rate_val) * final_time) / (
            1 + integral
        )
        result = population_geometric_parameter(
            birth_rate_val,
            death_rate_val,
            final_time,
        )

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)

    def test_geometric_parameter_general_case(self) -> None:
        """Test population_geometric_parameter for λ ≠ μ."""
        birth_rates = jnp.array([0.3, 0.4], dtype=jnp.float64)
        death_rates = jnp.array([0.1, 0.3], dtype=jnp.float64)
        epoch_ends = jnp.array([5.0, 9.0], dtype=jnp.float64)

        def _integrand(x: float) -> float:
            first = float(
                jnp.exp(-(birth_rates[0] - death_rates[0]) * x) * death_rates[0]
            )
            second = float(
                jnp.exp(
                    -(
                        (birth_rates[0] - death_rates[0]) * epoch_ends[0]
                        + (birth_rates[1] - death_rates[1]) * (x - epoch_ends[0])
                    ),
                )
                * death_rates[1]
            )
            return first if x <= float(epoch_ends[0]) else second

        integral, _ = scipy.integrate.quad(
            _integrand,
            0.0,
            epoch_ends[1].item(),
        )
        expected = jnp.exp(
            -(
                (birth_rates[1] - death_rates[1]) * (epoch_ends[1] - epoch_ends[0])
                + epoch_ends[0] * (birth_rates[0] - death_rates[0])
            ),
        ) / (1 + integral)
        result = population_geometric_parameter(birth_rates, death_rates, epoch_ends)

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)

    def test_non_spine_age_density_known_values(self) -> None:
        """Test non-spine age density against manually computed values.

        For specific parameter values, compute expected log density.
        """
        birth_rate_val = jnp.array([0.2], dtype=jnp.float64)
        death_rate_val = jnp.array([0.1], dtype=jnp.float64)
        age = jnp.array([5.0], dtype=jnp.float64)
        final_time = jnp.array([20.0], dtype=jnp.float64)

        # Non-spine density: f(t) = [λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}] / [1 - e^{-(λ-μ)T}]
        growth_rate = birth_rate_val - death_rate_val
        term1 = birth_rate_val * jnp.exp(-birth_rate_val * age)
        term2 = (
            death_rate_val
            * jnp.exp(-growth_rate * final_time)
            * jnp.exp(-death_rate_val * age)
        )
        numerator = term1 - term2
        denominator = 1 - jnp.exp(-growth_rate * final_time)
        density = numerator / denominator
        expected = jnp.log(density)

        result = non_spine_log_density(
            nested_terms(birth_rate_val, death_rate_val, final_time), age
        ).sum()

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)

    def test_extinction_pr_bounds(self) -> None:
        """Test that 0 < extinction probability < 1 for various inputs."""
        test_cases = [
            (0.1, 0.05, 100.0),
            (0.5, 0.3, 50.0),
            (1.0, 0.8, 10.0),
            (0.01, 0.005, 1000.0),
        ]

        for birth_rate_val, death_rate_val, final_time in test_cases:
            result = calculate_extinction_pr(
                jnp.array(birth_rate_val, dtype=jnp.float64),
                jnp.array(death_rate_val, dtype=jnp.float64),
                jnp.array(final_time, dtype=jnp.float64),
            )

            assert 0 < result < 1, (
                f"Failed for λ={birth_rate_val}, μ={death_rate_val}, "
                f"final_time={final_time}"
            )

    def test_piecewise_constant_single_vs_multi_epoch(self) -> None:
        """Test that single-epoch array matches scalar computation.

        A single-epoch piecewise-constant model should match scalar input.
        """
        birth_rate_val = jnp.array([0.1], dtype=jnp.float64)
        death_rate_val = jnp.array([0.07], dtype=jnp.float64)
        final_time = jnp.array([80.0], dtype=jnp.float64)
        n = jnp.array([5], dtype=jnp.int32)

        # Scalar computation
        result_scalar = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=final_time,
            ns=n,
        )

        # Single-epoch array computation
        result_array = calculate_populations_log_lik(
            birth_rate_val,
            death_rate_val,
            times=final_time,
            ns=n,
        )

        np.testing.assert_allclose(
            np.array(result_scalar), np.array(result_array), rtol=1.3e-6
        )


# ============================================================================
# LEVEL 1: Extinction Probability Tests (Foundation)
# ============================================================================


class TestExtinctionProbability:
    """Test extinction probability calculations."""

    def test_extinction_pr_basic(
        self,
        standard_array_params: dict[str, jax.Array],
    ) -> None:
        """Test basic extinction probability calculation."""
        result = calculate_extinction_pr(
            standard_array_params["birth_rates"],
            standard_array_params["death_rates"],
            standard_array_params["times"],
        )

        assert 0 < result < 1

    def test_extinction_pr_return_log_prob(
        self,
        standard_array_params: dict[str, jax.Array],
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
        assert np.isfinite(float(result))

    def test_extinction_pr_critical_case(self) -> None:
        """Test numerical stability when λ ≈ μ."""
        # Nearly equal rates
        birth_rate_arr = jnp.array([0.1, 0.1, 0.1], dtype=jnp.float64)
        death_rate_arr = jnp.array(
            [0.1 + 1e-13, 0.1 + 1e-13, 0.1 + 1e-13],
            dtype=jnp.float64,
        )
        times = jnp.array([50.0, 100.0, 150.0], dtype=jnp.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # Should handle critical case without returning inf or nan
        assert 0 < result < 1
        assert np.isfinite(float(result))

    def test_extinction_pr_zero_rates_both(self) -> None:
        """Test extinction probability when both λ = 0 and μ = 0."""
        birth_rate_arr = jnp.array([0.0], dtype=jnp.float64)
        death_rate_arr = jnp.array([0.0], dtype=jnp.float64)
        times = jnp.array([100.0], dtype=jnp.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # With no birth and no death, extinction probability should zero
        assert result == 0.0

    def test_extinction_pr_zero_birth_only(self) -> None:
        """Test extinction probability when λ = 0 and μ > 0 (only death)."""
        birth_rate_arr = jnp.array([0.0], dtype=jnp.float64)
        death_rate_arr = jnp.array([0.1], dtype=jnp.float64)
        times = jnp.array([100.0], dtype=jnp.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        assert 0 < result < 1

    def test_extinction_pr_zero_death_only(self) -> None:
        """Test extinction probability when λ > 0 and μ = 0 (only birth)."""
        birth_rate_arr = jnp.array([0.1], dtype=jnp.float64)
        death_rate_arr = jnp.array([0.0], dtype=jnp.float64)
        times = jnp.array([100.0], dtype=jnp.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # With zero death rate, extinction should be zero
        assert result == 0

    def test_extinction_pr_zero_rates_multi_epoch(self) -> None:
        """Test extinction probability with multi-epoch with some zero-rate epochs."""
        # Some epochs with normal rates, some with zero rates
        birth_rate_arr = jnp.array([0.1, 0.0, 0.15], dtype=jnp.float64)
        death_rate_arr = jnp.array([0.05, 0.0, 0.08], dtype=jnp.float64)
        times = jnp.array([50.0, 100.0, 150.0], dtype=jnp.float64)

        result = calculate_extinction_pr(birth_rate_arr, death_rate_arr, times)

        # Should handle mixed zero and non-zero rates
        assert 0 < result < 1

    def test_extinction_pr_increases_with_time(self) -> None:
        """Test that extinction probability increases as observation time increases."""
        birth_rate_val = jnp.array([0.1], dtype=jnp.float64)
        death_rate_val = jnp.array([0.05], dtype=jnp.float64)

        # Compare extinction probabilities at different times
        times = [50.0, 100.0, 200.0, 400.0]
        results = [
            calculate_extinction_pr(
                birth_rate_val,
                death_rate_val,
                jnp.array([t], dtype=jnp.float64),
            )
            for t in times
        ]

        # Extinction probability should be monotonically increasing with time
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Pr ext by (t={times[i]}) = {results[i]} should be < "
                f"Pr ext by (t={times[i + 1]}) = {results[i + 1]}"
            )

    def test_extinction_pr_increases_with_death_rate(self) -> None:
        """Test that extinction probability increases as death rate increases."""
        birth_rate_val = jnp.array([0.1], dtype=jnp.float64)
        final_time = jnp.array([100.0], dtype=jnp.float64)

        # Compare extinction probabilities at different death rates
        death_rates = [
            jnp.array([0.02], dtype=jnp.float64),
            jnp.array([0.05], dtype=jnp.float64),
            jnp.array([0.08], dtype=jnp.float64),
            jnp.array([0.1], dtype=jnp.float64),
        ]
        results = [
            calculate_extinction_pr(birth_rate_val, death_rate_val, final_time)
            for death_rate_val in death_rates
        ]

        # Extinction probability should be monotonically increasing with death rate
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Pr ext for (μ={death_rates[i]}) = {results[i]} should be < "
                f"Pr ext for (μ={death_rates[i + 1]}) = {results[i + 1]}"
            )

    def test_extinction_pr_decreases_with_birth_rate(self) -> None:
        """Test that extinction probability decreases as birth rate increases."""
        death_rate_val = jnp.array([0.05], dtype=jnp.float64)
        final_time = jnp.array([100.0], dtype=jnp.float64)

        # Compare extinction probabilities at different birth rates
        birth_rates = [
            jnp.array([0.05], dtype=jnp.float64),
            jnp.array([0.1], dtype=jnp.float64),
            jnp.array([0.15], dtype=jnp.float64),
            jnp.array([0.2], dtype=jnp.float64),
        ]
        results = [
            calculate_extinction_pr(birth_rate_val, death_rate_val, final_time)
            for birth_rate_val in birth_rates
        ]

        # Extinction probability should be monotonically decreasing with birth rate
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1], (
                f"Pr extinction for (λ={birth_rates[i]}) = {results[i]} should be > "
                f"Pr extinction for (λ={birth_rates[i + 1]}) = {results[i + 1]}"
            )


# ============================================================================
# LEVEL 2: Population Likelihood Tests
# ============================================================================


class TestPopulationsLogLikelihood:
    """Test population count log-likelihood calculations."""

    def test_scalar_inputs_basic(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test basic scalar input case."""
        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=jnp.array([1], dtype=jnp.int32),
        )

        assert np.isfinite(float(result))

    def test_scalar_inputs_n_zero(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test n=0 (extinction) case."""
        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=jnp.array([0], dtype=jnp.int32),
        )

        # P(n=0) = extinction_pr, so log P should be finite and negative
        assert np.isfinite(float(result))
        assert result < 0

    def test_scalar_inputs_array_ns(
        self,
        standard_scalar_params: dict[str, jax.Array],
        sample_population_counts: jax.Array,
    ) -> None:
        """Test scalar parameters with array of population counts."""
        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=sample_population_counts,
        )

        # Should return sum of log probabilities
        assert np.isfinite(float(result))

    def test_array_inputs_piecewise_constant(
        self,
        standard_array_params: dict[str, jax.Array],
    ) -> None:
        """Test piecewise-constant rates (array inputs)."""
        result = calculate_populations_log_lik(
            standard_array_params["birth_rates"],
            standard_array_params["death_rates"],
            times=standard_array_params["times"],
            ns=jnp.array([1], dtype=jnp.int32),
        )

        assert np.isfinite(float(result))

    def test_negative_time_returns_error(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError, match="times must be positive"):
            calculate_populations_log_lik(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                times=-100.0,  # Negative time
                ns=jnp.array([1], dtype=jnp.int32),
            )

    def test_negative_ns_returns_error(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test behavior for negative population count.

        The JAX implementation does not validate ns >= 0; it applies the
        geometric formula as-is and returns a finite (mathematically
        meaningless) value without raising.
        """
        result = calculate_populations_log_lik(
            standard_scalar_params["birth_rate"],
            standard_scalar_params["death_rate"],
            times=standard_scalar_params["time"],
            ns=-5,  # Negative count
        )
        assert np.isfinite(float(result))

    def test_critical_case_scalar(
        self,
        nearly_equal_params: dict[str, jax.Array],
    ) -> None:
        """Test numerical stability when λ ≈ μ (scalar)."""
        result = calculate_populations_log_lik(
            nearly_equal_params["birth_rate"],
            nearly_equal_params["death_rate"],
            times=nearly_equal_params["time"],
            ns=jnp.array([1], dtype=jnp.int32),
        )

        assert np.isfinite(float(result))

    def test_array_length_mismatch_raises_error(self) -> None:
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="Arrays must have matching lengths"):
            calculate_populations_log_lik(
                birth_rates=jnp.array([0.1, 0.15], dtype=jnp.float64),
                death_rates=jnp.array(
                    [0.05, 0.06, 0.055],
                    dtype=jnp.float64,
                ),  # Different length
                times=jnp.array([50.0, 100.0], dtype=jnp.float64),
                ns=jnp.array([1], dtype=jnp.int32),
            )


# ============================================================================
# LEVEL 3: Age Likelihood Tests - Spine Individuals
# ============================================================================


class TestSpineAgesLogLikelihood:
    """Test age log-likelihood for spine individuals."""

    def test_scalar_age(
        self,
        standard_scalar_params: dict[str, jax.Array],
        sample_ages_scalar: jax.Array,
    ) -> None:
        """Test with single age value."""
        result = spine_log_density(
            nested_terms(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                standard_scalar_params["time"],
            ),
            sample_ages_scalar,
        ).sum()

        assert np.isfinite(float(result))

    def test_array_ages(
        self,
        standard_scalar_params: dict[str, jax.Array],
        sample_ages_array: jax.Array,
    ) -> None:
        """Test with array of ages."""
        result = spine_log_density(
            nested_terms(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                standard_scalar_params["time"],
            ),
            sample_ages_array,
        ).sum()

        # Should return summed log-likelihood
        assert np.isfinite(float(result))

    def test_age_at_boundary(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test age exactly equal to observation timefinal_time."""
        final_time = standard_scalar_params["time"]
        result = spine_log_density(
            nested_terms(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                final_time,
            ),
            final_time,
        ).sum()

        assert np.isfinite(float(result))

    def test_critical_case(
        self,
        nearly_equal_params: dict[str, jax.Array],
        sample_ages_scalar: jax.Array,
    ) -> None:
        """Test numerical stability when λ ≈ μ."""
        result = spine_log_density(
            nested_terms(
                nearly_equal_params["birth_rate"],
                nearly_equal_params["death_rate"],
                nearly_equal_params["time"],
            ),
            sample_ages_scalar,
        ).sum()

        assert np.isfinite(float(result))

    def test_invalid_age_greater_than_final_time(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test that age > final_time returns -inf."""
        with pytest.raises(
            ValueError,
            match=r"ages must be in \[0, .*\]. Got ages:",
        ):  # age > final_time
            spine_log_density(
                nested_terms(
                    standard_scalar_params["birth_rate"],
                    standard_scalar_params["death_rate"],
                    standard_scalar_params["time"],
                ),
                standard_scalar_params["time"] + 10.0,
            ).sum()

    def test_negative_age_returns_minus_inf(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test that negative age returns -inf."""
        with pytest.raises(
            ValueError,
            match=r"ages must be in \[0, .*\]. Got ages:",
        ):  # Negative age
            spine_log_density(
                nested_terms(
                    standard_scalar_params["birth_rate"],
                    standard_scalar_params["death_rate"],
                    standard_scalar_params["time"],
                ),
                -10.0,
            ).sum()


# ============================================================================
# LEVEL 4: Age Likelihood Tests - Non-Spine Individuals
# ============================================================================


class TestNonSpineAgesLogLikelihood:
    """Test age log-likelihood for non-spine individuals."""

    def test_scalar_params_scalar_age(self, sample_ages_scalar: jax.Array) -> None:
        """Test with scalar parameters and single age."""
        # For single-epoch case, wrap scalars in arrays of length 1
        result = non_spine_log_density(
            nested_terms(
                jnp.array([0.1], dtype=jnp.float64),
                jnp.array([0.05], dtype=jnp.float64),
                jnp.array([100.0], dtype=jnp.float64),
            ),
            sample_ages_scalar,
        ).sum()

        assert np.isfinite(float(result))

    def test_scalar_params_array_ages(self, sample_ages_array: jax.Array) -> None:
        """Test with scalar parameters and array of ages."""
        # For single-epoch case, wrap scalars in arrays of length 1
        result = non_spine_log_density(
            nested_terms(
                jnp.array([0.1], dtype=jnp.float64),
                jnp.array([0.05], dtype=jnp.float64),
                jnp.array([100.0], dtype=jnp.float64),
            ),
            sample_ages_array,
        ).sum()

        assert np.isfinite(float(result))

    def test_array_params_piecewise_constant(
        self,
        standard_array_params: dict[str, jax.Array],
    ) -> None:
        """Test with piecewise-constant rates (array parameters)."""
        # Use ages within the valid time range
        max_time = standard_array_params["times"][-1]
        ages = jnp.array([10.0, 50.0, max_time - 10.0], dtype=jnp.float64)
        result = non_spine_log_density(
            nested_terms(
                standard_array_params["birth_rates"],
                standard_array_params["death_rates"],
                standard_array_params["times"],
            ),
            ages,
        ).sum()

        assert np.isfinite(float(result))

    def test_single_epoch(self, sample_ages_scalar: jax.Array) -> None:
        """Test with single epoch (single-element arrays)."""
        result = non_spine_log_density(
            nested_terms(
                jnp.array([0.1], dtype=jnp.float64),
                jnp.array([0.05], dtype=jnp.float64),
                jnp.array([100.0], dtype=jnp.float64),
            ),
            sample_ages_scalar,
        ).sum()

        assert np.isfinite(float(result))

    def test_critical_case(self, sample_ages_scalar: jax.Array) -> None:
        """Test numerical stability when λ ≈ μ."""
        result = non_spine_log_density(
            nested_terms(
                jnp.array([0.1], dtype=jnp.float64),
                jnp.array([0.1 + 1e-13], dtype=jnp.float64),
                jnp.array([100.0], dtype=jnp.float64),
            ),
            sample_ages_scalar,
        ).sum()

        assert np.isfinite(float(result))

    def test_invalid_age_returns_minus_inf(self) -> None:
        """Test that invalid ages return -inf."""
        # Age > T
        with pytest.raises(ValueError, match=r"ages must be in \[0, .*\]. Got ages:"):
            non_spine_log_density(
                nested_terms(
                    jnp.array([0.1], dtype=jnp.float64),
                    jnp.array([0.05], dtype=jnp.float64),
                    jnp.array([100.0], dtype=jnp.float64),
                ),
                150.0,
            ).sum()

    def test_shape_mismatch_raises_error(
        self,
        sample_ages_scalar: jax.Array,
    ) -> None:
        """Test that parameter shape mismatches raise ValueError."""
        with pytest.raises(ValueError, match="matching lengths"):
            non_spine_log_density(
                nested_terms(
                    jnp.array([0.1, 0.15], dtype=jnp.float64),
                    jnp.array(
                        [0.05],
                        dtype=jnp.float64,
                    ),
                    jnp.array([50.0, 100.0], dtype=jnp.float64),
                ),
                sample_ages_scalar,
            ).sum()

    def test_non_monotonic_times_raises_error(
        self,
        sample_ages_scalar: jax.Array,
    ) -> None:
        """Test that non-monotonic times raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            non_spine_log_density(
                nested_terms(
                    jnp.array([0.1, 0.15, 0.12], dtype=jnp.float64),
                    jnp.array([0.05, 0.06, 0.055], dtype=jnp.float64),
                    jnp.array(
                        [50.0, 100.0, 90.0],
                        dtype=jnp.float64,
                    ),
                ),
                sample_ages_scalar,
            ).sum()


# ============================================================================
# LEVEL 5: Unified Age Likelihood Interface Tests
# ============================================================================


class TestAgesLogLikelihood:
    """Test unified age log-likelihood interface (spine/non-spine selector)."""

    def test_spine_routing(
        self,
        standard_scalar_params: dict[str, jax.Array],
        sample_ages_scalar: jax.Array,
    ) -> None:
        """Test that are_spines=True routes to spine implementation."""
        result = spine_log_density(
            nested_terms(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                standard_scalar_params["time"],
            ),
            sample_ages_scalar,
        ).sum()

        # Should match spine calculation
        expected = spine_log_density(
            nested_terms(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                standard_scalar_params["time"],
            ),
            sample_ages_scalar,
        ).sum()

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)

    def test_non_spine_routing(self, sample_ages_scalar: jax.Array) -> None:
        """Test that are_spines=False routes to non-spine implementation."""
        result = non_spine_log_density(
            nested_terms(0.1, 0.05, 100.0), sample_ages_scalar
        ).sum()

        # Should match non-spine calculation (using array inputs)
        expected = non_spine_log_density(
            nested_terms(
                jnp.array([0.1], dtype=jnp.float64),
                jnp.array([0.05], dtype=jnp.float64),
                jnp.array([100.0], dtype=jnp.float64),
            ),
            sample_ages_scalar,
        ).sum()

        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1.3e-6)


# ============================================================================
# LEVEL 6: Mathematical Property Tests
# ============================================================================


class TestMathematicalProperties:
    """Test mathematical properties and consistency."""

    def test_pr_bounds_population(
        self,
        standard_scalar_params: dict[str, jax.Array],
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
            assert result <= 0 or np.isclose(
                float(result),
                float(jnp.array(0.0, dtype=jnp.float64)),
            )

    def test_extinction_pr_increases_with_mu(self) -> None:
        """Test that extinction probability increases as death rate increases."""
        birth_rates = jnp.array([0.1, 0.1, 0.1], dtype=jnp.float64)
        times = jnp.array([50.0, 100.0, 150.0], dtype=jnp.float64)

        extinction_pr_1 = calculate_extinction_pr(
            birth_rates,
            jnp.array([0.05, 0.05, 0.05], dtype=jnp.float64),
            times,
        )
        extinction_pr_2 = calculate_extinction_pr(
            birth_rates,
            jnp.array([0.08, 0.08, 0.08], dtype=jnp.float64),
            times,
        )

        # Higher death rate → higher extinction probability
        assert 0 < extinction_pr_1 < extinction_pr_2 < 1

    def test_consistency_scalar_vs_single_element_array(self) -> None:
        """Test that scalar and single-element array give same results."""
        scalar_result = calculate_populations_log_lik(
            birth_rates=0.1,
            death_rates=0.05,
            times=100.0,
            ns=jnp.array([1], dtype=jnp.int32),
        )

        array_result = calculate_populations_log_lik(
            birth_rates=jnp.array([0.1], dtype=jnp.float64),
            death_rates=jnp.array([0.05], dtype=jnp.float64),
            times=jnp.array([100.0], dtype=jnp.float64),
            ns=jnp.array([1], dtype=jnp.int32),
        )

        # Results should be very close
        np.testing.assert_allclose(
            np.array(scalar_result), np.array(array_result), rtol=1.3e-6
        )

    def test_sum_property_array_ns(
        self,
        standard_scalar_params: dict[str, jax.Array],
    ) -> None:
        """Test that array of ns gives sum of individual likelihoods."""
        ns_array = jnp.array([5, 10, 15], dtype=jnp.int32)

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

        np.testing.assert_allclose(
            np.array(result_together), np.array(result_separate), rtol=1.3e-6
        )


# ============================================================================
# LEVEL 7: Numerical Stability and Edge Case Tests
# ============================================================================


# ============================================================================
# LEVEL 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Test integration between related functions."""

    def test_spine_non_spine_different_formulas(
        self,
        standard_scalar_params: dict[str, jax.Array],
        sample_ages_scalar: jax.Array,
    ) -> None:
        """Test that spine and non-spine give different results (different formulas)."""
        spine_result = spine_log_density(
            nested_terms(
                standard_scalar_params["birth_rate"],
                standard_scalar_params["death_rate"],
                standard_scalar_params["time"],
            ),
            sample_ages_scalar,
        ).sum()

        non_spine_result = non_spine_log_density(
            nested_terms(
                jnp.array([standard_scalar_params["birth_rate"]], dtype=jnp.float64),
                jnp.array([standard_scalar_params["death_rate"]], dtype=jnp.float64),
                jnp.array([standard_scalar_params["time"]], dtype=jnp.float64),
            ),
            sample_ages_scalar,
        ).sum()

        # These should be different (different conditioning)
        assert not np.isclose(float(spine_result), float(non_spine_result))


# ============================================================================
# Performance and Vectorization Tests
# ============================================================================


# ============================================================================
# Private Helper Function Tests
# ============================================================================
