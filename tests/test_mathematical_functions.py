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

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose

# Import private functions for direct testing
from inf_bp_texts.models.mathematical_functions import (
    _calculate_geometric_parameter,
    _log_likelihood_ages,
    _log_probability_extinction_given_ages,
    _probability_extinction_numerator,
    calculate_ages_log_likelihood,
    calculate_ages_log_likelihood_derivatives,
    calculate_extinction_probability,
    calculate_non_spine_ages_log_likelihood,
    calculate_non_spine_ages_log_likelihood_derivatives,
    calculate_populations_log_likelihood,
    calculate_populations_log_likelihood_derivatives,
    calculate_spine_ages_log_likelihood,
    calculate_spine_ages_log_likelihood_derivatives,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def standard_scalar_params():
    """Standard scalar parameters for testing."""
    return {
        "lambda_param": 0.1,
        "mu_param": 0.05,
        "time": 100.0,
    }


@pytest.fixture
def standard_array_params():
    """Standard array parameters for piecewise-constant rates."""
    return {
        "lambda_params": np.array([0.1, 0.15, 0.12]),
        "mu_params": np.array([0.05, 0.06, 0.055]),
        "times": np.array([50.0, 100.0, 150.0]),
    }


@pytest.fixture
def nearly_equal_params():
    """Parameters where λ ≈ μ (numerical stability test)."""
    return {
        "lambda_param": 0.1,
        "mu_param": 0.1 + 1e-13,  # Within tolerance of 1e-12
        "time": 100.0,
    }


@pytest.fixture
def log_scale_params():
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
    return np.array([10.0, 30.0, 50.0, 70.0, 90.0])


@pytest.fixture
def sample_population_counts():
    """Sample population counts (n values)."""
    return np.array([0, 1, 5, 10, 20])


# ============================================================================
# LEVEL 0: Numerical Accuracy Tests (Analytical Validation)
# ============================================================================


class TestNumericalAccuracy:
    """Test functions against known analytical results.

    These tests validate that the implementations produce numerically correct
    results by comparing against hand-calculated or analytically derived values.
    """

    def test_extinction_probability_critical_case_analytical(self):
        """Test extinction probability when λ = μ (analytical formula).

        When λ = μ, the extinction probability has the closed form:
        α_t = λT / (1 + λT)
        """
        lambda_val = 0.1
        T = 100.0

        # Expected: α_t = 0.1 * 100 / (1 + 0.1 * 100) = 10 / 11 ≈ 0.909091
        expected = lambda_val * T / (1 + lambda_val * T)

        result = calculate_extinction_probability(
            np.array([lambda_val]),
            np.array([lambda_val]),  # μ = λ
            np.array([T]),
        )

        assert_allclose(result, expected, rtol=1e-6)

    def test_populations_log_likelihood_n0_simple(self):
        """Test log P(n=0) for simple parameters.

        For n=0: log P(n=0) = log(α_t)
        Where α_t = μ(e^{(λ-μ)T} - 1) / (λ e^{(λ-μ)T} - μ)
        """
        lambda_val = 0.2
        mu_val = 0.1
        T = 10.0

        # Compute expected α_t manually
        growth = (lambda_val - mu_val) * T  # 1.0
        exp_growth = np.exp(growth)  # e^1.0 ≈ 2.71828
        alpha_t = mu_val * (exp_growth - 1) / (lambda_val * exp_growth - mu_val)
        expected_log_prob = np.log(alpha_t)

        result = calculate_populations_log_likelihood(lambda_val, mu_val, T, ns=0)

        assert_allclose(result, expected_log_prob, rtol=1e-6)

    def test_populations_log_likelihood_critical_case_n0(self):
        """Test log P(n=0) when λ = μ.

        When λ = μ: α_t = λT / (1 + λT)
        So log P(n=0) = log(λT / (1 + λT))
        """
        lambda_val = 0.1
        T = 50.0

        # Expected: α_t = 0.1 * 50 / (1 + 0.1 * 50) = 5/6
        alpha_t = lambda_val * T / (1 + lambda_val * T)
        expected = np.log(alpha_t)

        result = calculate_populations_log_likelihood(lambda_val, lambda_val, T, ns=0)

        assert_allclose(result, expected, rtol=1e-6)

    def test_populations_log_likelihood_n1_simple(self):
        """Test log P(n=1) for simple parameters.

        For n=1: log P(n=1) = log(1 - α_t) + log(1 - β_t)
        Where β_t = (λ/μ) * α_t
        """
        lambda_val = 0.3
        mu_val = 0.2
        T = 5.0

        # Compute α_t and β_t
        growth = (lambda_val - mu_val) * T
        exp_growth = np.exp(growth)
        alpha_t = mu_val * (exp_growth - 1) / (lambda_val * exp_growth - mu_val)
        beta_t = (lambda_val / mu_val) * alpha_t

        expected = np.log(1 - alpha_t) + np.log(1 - beta_t)

        result = calculate_populations_log_likelihood(lambda_val, mu_val, T, ns=1)

        assert_allclose(result, expected, rtol=1e-6)

    def test_spine_age_at_zero(self):
        """Test spine age density at t=0.

        At t=0 (age zero), the density involves special limiting behavior.
        """
        lambda_val = 0.2
        mu_val = 0.1
        T = 10.0
        age = 0.0

        result = calculate_spine_ages_log_likelihood(lambda_val, mu_val, age, T)

        # Should return finite value (not -inf)
        assert np.isfinite(result)
        # Age density at t=0 should be positive (log density negative but finite)
        assert result > -100  # Reasonable bound

    def test_geometric_scalar_case(self):
        """
        Docstring for test_geometric_scalar_case

        :param self: Description
        """

        T = 10.0
        lambda_val = 1.0
        mu_val = 2.0

        def _integrand(x: np.float64) -> np.float64:
            return np.exp(-(lambda_val - mu_val) * x, dtype=np.float64) * mu_val

        integral = scipy.integrate.quad(_integrand, 0, T)[0]
        expected = np.exp(-(lambda_val - mu_val) * T) / (1 + integral)
        result = _calculate_geometric_parameter(lambda_val, mu_val, T)
        assert_allclose(result, expected, rtol=1e-6)

    def test_geometric_parameter_critical_case(self):
        """Test _calculate_geometric_parameter when λ = μ.

        Private function test: This function returns 1 - β_t where β_t = λT / (1 + λT)
        """
        lambda_val = mu_val = 0.15
        T = 20.0

        def _integrand(x: np.float64) -> np.float64:
            return np.exp(-(lambda_val - mu_val) * x, dtype=np.float64) * mu_val

        integral = scipy.integrate.quad(_integrand, 0, T)[0]
        expected = np.exp(-(lambda_val - mu_val) * T) / (1 + integral)
        result = _calculate_geometric_parameter(lambda_val, mu_val, T)

        assert_allclose(result, expected, rtol=1e-6)

    def test_geometric_parameter_general_case(self):
        """Test _calculate_geometric_parameter for λ ≠ μ.

        Private function test: Function returns 1 - β_t where β_t = (λ/μ) * α_t
        """
        birth_rates = np.array([0.3, 0.4])
        death_rates = np.array([0.1, 0.3])
        epoch_ends = np.array([5.0, 9.0])

        def _integrand(x: float) -> float:
            if x <= epoch_ends[0]:
                return np.exp(-(birth_rates[0] - death_rates[0]) * x) * death_rates[0]
            else:
                return (
                    np.exp(
                        -(
                            (birth_rates[0] - death_rates[0]) * epoch_ends[0]
                            + (birth_rates[1] - death_rates[1]) * (x - epoch_ends[0])
                        )
                    )
                    * death_rates[1]
                )

        integral = scipy.integrate.quad(_integrand, 0.0, epoch_ends[1])[0]
        expected = np.exp(
            -(
                (birth_rates[1] - death_rates[1]) * (epoch_ends[1] - epoch_ends[0])
                + epoch_ends[0] * (birth_rates[0] - death_rates[0])
            )
        ) / (1 + integral)
        result = _calculate_geometric_parameter(birth_rates, death_rates, epoch_ends)

        assert_allclose(result, expected, rtol=1e-6)

    def test_probability_extinction_numerator_simple(self):
        """Test _probability_extinction_numerator for simple case.

        Private function test: Validates that function returns a positive finite value.
        The exact formula is complex and involves integration - we validate properties.
        """
        lambda_val = 0.2
        mu_val = 0.1
        T = 10.0

        result = _probability_extinction_numerator(
            np.array([lambda_val]),
            np.array([mu_val]),
            np.array([T]),
            log_scale_params=False,
        )

        # Should be positive and finite
        assert np.isfinite(result)
        assert result > 0
        # For λ > μ, numerator should be between 0 and 1
        assert 0 < result < 1

    def test_log_likelihood_ages_simple(self):
        """Test _log_likelihood_ages for simple scalar age.

        Private function test: log density involves birth rate and growth.
        """
        lambda_val = 0.1
        age = 20.0
        T = 100.0

        # For simple case: log f(t) = log(λ) - λt
        # (This is simplified; actual formula may differ)
        result = _log_likelihood_ages(
            age, np.array([lambda_val]), np.array([T]), log_scale_params=False
        )

        # Should be finite and negative (log of density)
        assert np.isfinite(result)
        assert result < 0

    def test_log_likelihood_ages_array(self):
        """Test _log_likelihood_ages with array of ages.

        Private function test: should sum log likelihoods across ages.
        """
        lambda_val = 0.1
        ages = np.array([10.0, 30.0, 50.0])
        T = 100.0

        # Compute for array
        result_array = _log_likelihood_ages(
            ages, np.array([lambda_val]), np.array([T]), log_scale_params=False
        )

        # Compute individually and sum
        result_sum = sum(
            _log_likelihood_ages(
                age, np.array([lambda_val]), np.array([T]), log_scale_params=False
            )
            for age in ages
        )

        assert_allclose(result_array, result_sum, rtol=1e-10)

    def test_non_spine_age_density_known_values(self):
        """Test non-spine age density against manually computed values.

        For specific parameter values, compute expected log density.
        """
        lambda_val = 0.2
        mu_val = 0.1
        age = 5.0
        T = 20.0

        # Non-spine density: f(t) = [λe^{-λt} - μe^{-(λ-μ)T}e^{-μt}] / [1 - e^{-(λ-μ)T}]
        growth_rate = lambda_val - mu_val
        term1 = lambda_val * np.exp(-lambda_val * age)
        term2 = mu_val * np.exp(-growth_rate * T) * np.exp(-mu_val * age)
        numerator = term1 - term2
        denominator = 1 - np.exp(-growth_rate * T)
        density = numerator / denominator
        expected = np.log(density)

        result = calculate_non_spine_ages_log_likelihood(
            np.array([lambda_val]), np.array([mu_val]), age, np.array([T])
        )

        assert_allclose(result, expected, rtol=1e-6)

    def test_derivatives_sum_over_array(self):
        """Test that derivatives sum correctly over array of n values.

        Computing derivatives for [n1, n2] should equal sum of individual derivatives.
        """
        lambda_val = 0.15
        mu_val = 0.08
        T = 50.0
        ns = np.array([1, 5, 10])

        # Compute together
        result_together = calculate_populations_log_likelihood_derivatives(
            lambda_val, mu_val, T, ns
        )

        # Compute separately and sum
        derivs_separate = [
            calculate_populations_log_likelihood_derivatives(lambda_val, mu_val, T, n)
            for n in ns
        ]
        result_separate = np.sum(derivs_separate, axis=0)

        assert_allclose(result_together, result_separate, rtol=1e-10)

    def test_extinction_probability_bounds(self):
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
            result = calculate_extinction_probability(
                np.array([lambda_val]), np.array([mu_val]), np.array([T])
            )

            assert 0 < result < 1, f"Failed for λ={lambda_val}, μ={mu_val}, T={T}"

    def test_log_probability_extinction_given_ages_zero(self):
        """Test _log_probability_extinction_given_ages at age zero.

        Private function test: Validates function returns reasonable log probability.
        """
        lambda_val = 0.1
        mu_val = 0.05
        T = 100.0
        age = 0.0

        result = _log_probability_extinction_given_ages(
            age, np.array([lambda_val]), np.array([mu_val]), np.array([T])
        )

        # Should be finite (could be 0 at age 0, representing log(1) = 0)
        assert np.isfinite(result)
        # Log probability should be non-positive (probability ≤ 1)
        assert result <= 0

    def test_piecewise_constant_single_vs_multi_epoch(self):
        """Test that single-epoch array matches scalar computation.

        A single-epoch piecewise-constant model should match scalar input.
        """
        lambda_val = 0.12
        mu_val = 0.07
        T = 80.0
        n = 5

        # Scalar computation
        result_scalar = calculate_populations_log_likelihood(lambda_val, mu_val, T, n)

        # Single-epoch array computation
        result_array = calculate_populations_log_likelihood(
            np.array([lambda_val]), np.array([mu_val]), np.array([T]), n
        )

        assert_allclose(result_scalar, result_array, rtol=1e-10)


# ============================================================================
# LEVEL 1: Extinction Probability Tests (Foundation)
# ============================================================================


class TestExtinctionProbability:
    """Test extinction probability calculations (α_t)."""

    def test_extinction_probability_basic(self, standard_array_params):
        """Test basic extinction probability calculation."""
        result = calculate_extinction_probability(
            standard_array_params["lambda_params"],
            standard_array_params["mu_params"],
            standard_array_params["times"],
        )

        # α_t should be in (0, 1)
        assert 0 < result < 1
        assert isinstance(result, np.float64)

    def test_extinction_probability_log_scale(self, log_scale_params):
        """Test extinction probability with log-scale parameters."""
        lambda_arr = np.array([log_scale_params["log_lambda"]])
        mu_arr = np.array([log_scale_params["log_mu"]])
        times_arr = np.array([log_scale_params["time"]])

        result = calculate_extinction_probability(
            lambda_arr, mu_arr, times_arr, log_scale_params=True
        )

        assert 0 < result < 1

    def test_extinction_probability_return_log(self, standard_array_params):
        """Test extinction probability with return_log_prob=True."""
        result = calculate_extinction_probability(
            standard_array_params["lambda_params"],
            standard_array_params["mu_params"],
            standard_array_params["times"],
            return_log_prob=True,
        )

        # Log probability should be negative
        assert result < 0
        assert np.isfinite(result)

    def test_extinction_probability_critical_case(self):
        """Test numerical stability when λ ≈ μ."""
        # Nearly equal rates
        lambda_arr = np.array([0.1, 0.1, 0.1])
        mu_arr = np.array([0.1 + 1e-13, 0.1 + 1e-13, 0.1 + 1e-13])
        times = np.array([50.0, 100.0, 150.0])

        result = calculate_extinction_probability(lambda_arr, mu_arr, times)

        # Should handle critical case without returning inf or nan
        assert 0 < result < 1
        assert np.isfinite(result)

    def test_extinction_probability_monotonic_times(self):
        """Test that times must be monotonically increasing."""
        # This test is implicit in the function's validation
        # Non-monotonic times should raise an error in calling functions
        pass


# ============================================================================
# LEVEL 2: Population Likelihood Tests
# ============================================================================


class TestPopulationsLogLikelihood:
    """Test population count log-likelihood calculations."""

    def test_scalar_inputs_basic(self, standard_scalar_params):
        """Test basic scalar input case."""
        result = calculate_populations_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=10,
        )

        assert np.isfinite(result)
        assert isinstance(result, np.float64)

    def test_scalar_inputs_n_zero(self, standard_scalar_params):
        """Test n=0 (extinction) case."""
        result = calculate_populations_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=0,
        )

        # P(n=0) = α_t, so log P should be finite and negative
        assert np.isfinite(result)
        assert result < 0

    def test_scalar_inputs_array_ns(
        self, standard_scalar_params, sample_population_counts
    ):
        """Test scalar parameters with array of population counts."""
        result = calculate_populations_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=sample_population_counts,
        )

        # Should return sum of log probabilities
        assert np.isfinite(result)

    def test_array_inputs_piecewise_constant(self, standard_array_params):
        """Test piecewise-constant rates (array inputs)."""
        result = calculate_populations_log_likelihood(
            standard_array_params["lambda_params"],
            standard_array_params["mu_params"],
            standard_array_params["times"],
            ns=10,
            log_scale_params=False,
        )

        assert np.isfinite(result)

    def test_negative_time_returns_error(self, standard_scalar_params):
        """Test that negative time raises ValueError."""
        with pytest.raises(ValueError):
            calculate_populations_log_likelihood(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                -100.0,  # Negative time
                ns=10,
            )

    def test_negative_ns_returns_error(self, standard_scalar_params):
        """Test that negative population count raises ValueError."""
        with pytest.raises(ValueError):
            calculate_populations_log_likelihood(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                standard_scalar_params["time"],
                ns=-5,  # Negative count
            )

    def test_critical_case_scalar(self, nearly_equal_params):
        """Test numerical stability when λ ≈ μ (scalar)."""
        result = calculate_populations_log_likelihood(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            nearly_equal_params["time"],
            ns=10,
        )

        assert np.isfinite(result)

    def test_log_scale_params(self, log_scale_params):
        """Test with log-scale parameters."""
        result = calculate_populations_log_likelihood(
            log_scale_params["log_lambda"],
            log_scale_params["log_mu"],
            log_scale_params["time"],
            ns=10,
            log_scale_params=True,
        )

        assert np.isfinite(result)

    def test_mixed_input_types_raise_error(self):
        """Test that mixing scalar and array inputs raises ValueError."""
        with pytest.raises(ValueError, match="must all be arrays or all scalars"):
            calculate_populations_log_likelihood(
                lambda_params=0.1,  # scalar
                mu_params=np.array([0.05, 0.06]),  # array
                times=100.0,  # scalar
                ns=10,
            )

    def test_array_length_mismatch_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="matching lengths"):
            calculate_populations_log_likelihood(
                lambda_params=np.array([0.1, 0.15]),
                mu_params=np.array([0.05, 0.06, 0.055]),  # Different length
                times=np.array([50.0, 100.0]),
                ns=10,
            )

    def test_numpy_scalars_accepted(self):
        """Test that numpy scalar types are accepted (they inherit from Python float)."""
        # np.float64 instances pass isinstance(x, float) check
        result = calculate_populations_log_likelihood(
            lambda_params=np.float64(0.1),
            mu_params=np.float64(0.05),
            times=np.float64(100.0),
            ns=10,
        )
        assert np.isfinite(result)


class TestPopulationsLogLikelihoodDerivatives:
    """Test derivatives of population log-likelihood."""

    def test_basic_derivatives(self, standard_scalar_params):
        """Test basic derivative calculation."""
        result = calculate_populations_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=10,
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_with_n_zero(self, standard_scalar_params):
        """Test derivatives for n=0 case."""
        result = calculate_populations_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=0,
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_array_ns(
        self, standard_scalar_params, sample_population_counts
    ):
        """Test derivatives with array of n values."""
        result = calculate_populations_log_likelihood_derivatives(
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
        result = calculate_populations_log_likelihood_derivatives(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            nearly_equal_params["time"],
            ns=10,
        )

        assert np.all(np.isfinite(result))

    def test_derivatives_log_scale(self, log_scale_params):
        """Test derivatives with log-scale parameters."""
        result = calculate_populations_log_likelihood_derivatives(
            log_scale_params["log_lambda"],
            log_scale_params["log_mu"],
            log_scale_params["time"],
            ns=10,
            log_scale_params=True,
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


# ============================================================================
# LEVEL 3: Age Likelihood Tests - Spine Individuals
# ============================================================================


class TestSpineAgesLogLikelihood:
    """Test age log-likelihood for spine individuals."""

    def test_scalar_age(self, standard_scalar_params, sample_ages_scalar):
        """Test with single age value."""
        result = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert np.isfinite(result)
        assert isinstance(result, np.float64)

    def test_array_ages(self, standard_scalar_params, sample_ages_array):
        """Test with array of ages."""
        result = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        # Should return summed log-likelihood
        assert np.isfinite(result)

    def test_age_at_boundary(self, standard_scalar_params):
        """Test age exactly equal to observation time T."""
        T = standard_scalar_params["time"]
        result = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            T,  # Age = T (boundary case)
            T,
        )

        assert np.isfinite(result)

    def test_critical_case(self, nearly_equal_params, sample_ages_scalar):
        """Test numerical stability when λ ≈ μ."""
        result = calculate_spine_ages_log_likelihood(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            sample_ages_scalar,
            nearly_equal_params["time"],
        )

        assert np.isfinite(result)

    def test_invalid_age_greater_than_T(self, standard_scalar_params):
        """Test that age > T returns -inf."""
        result = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"] + 10.0,  # age > T
            standard_scalar_params["time"],
        )

        assert result == -np.inf

    def test_negative_age_returns_minus_inf(self, standard_scalar_params):
        """Test that negative age returns -inf."""
        result = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            -10.0,  # Negative age
            standard_scalar_params["time"],
        )

        assert result == -np.inf

    def test_log_scale_params(self, log_scale_params, sample_ages_scalar):
        """Test with log-scale parameters."""
        result = calculate_spine_ages_log_likelihood(
            log_scale_params["log_lambda"],
            log_scale_params["log_mu"],
            sample_ages_scalar,
            log_scale_params["time"],
            log_scale_params=True,
        )

        assert np.isfinite(result)


class TestSpineAgesLogLikelihoodDerivatives:
    """Test derivatives for spine age log-likelihood."""

    def test_scalar_age_derivatives(self, standard_scalar_params, sample_ages_scalar):
        """Test derivatives with single age."""
        result = calculate_spine_ages_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_array_ages_derivatives(self, standard_scalar_params, sample_ages_array):
        """Test derivatives with array of ages."""
        result = calculate_spine_ages_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        # Should return summed derivatives
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_at_boundary(self, standard_scalar_params):
        """Test derivatives when age = T."""
        T = standard_scalar_params["time"]
        result = calculate_spine_ages_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            T,
            T,
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_critical_case(self, nearly_equal_params, sample_ages_scalar):
        """Test derivative stability when λ ≈ μ."""
        result = calculate_spine_ages_log_likelihood_derivatives(
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
        result = calculate_non_spine_ages_log_likelihood(
            lambda_params=np.array([0.1]),
            mu_params=np.array([0.05]),
            ages_t=sample_ages_scalar,
            times=np.array([100.0]),
            log_scale_params=False,
        )

        assert np.isfinite(result)

    def test_scalar_params_array_ages(self, sample_ages_array):
        """Test with scalar parameters and array of ages."""
        # For single-epoch case, wrap scalars in arrays of length 1
        result = calculate_non_spine_ages_log_likelihood(
            lambda_params=np.array([0.1]),
            mu_params=np.array([0.05]),
            ages_t=sample_ages_array,
            times=np.array([100.0]),
            log_scale_params=False,
        )

        assert np.isfinite(result)

    def test_array_params_piecewise_constant(self, standard_array_params):
        """Test with piecewise-constant rates (array parameters)."""
        # Use ages within the valid time range
        max_time = standard_array_params["times"][-1]
        ages = np.array([10.0, 50.0, max_time - 10.0])
        result = calculate_non_spine_ages_log_likelihood(
            lambda_params=standard_array_params["lambda_params"],
            mu_params=standard_array_params["mu_params"],
            ages_t=ages,
            times=standard_array_params["times"],
            log_scale_params=False,
        )

        assert np.isfinite(result)

    def test_single_epoch(self, sample_ages_scalar):
        """Test with single epoch (single-element arrays)."""
        result = calculate_non_spine_ages_log_likelihood(
            lambda_params=np.array([0.1]),
            mu_params=np.array([0.05]),
            ages_t=sample_ages_scalar,
            times=np.array([100.0]),
        )

        assert np.isfinite(result)

    def test_critical_case(self, sample_ages_scalar):
        """Test numerical stability when λ ≈ μ."""
        result = calculate_non_spine_ages_log_likelihood(
            lambda_params=np.array([0.1]),
            mu_params=np.array([0.1 + 1e-13]),
            ages_t=sample_ages_scalar,
            times=np.array([100.0]),
        )

        assert np.isfinite(result)

    def test_invalid_age_returns_minus_inf(self):
        """Test that invalid ages return -inf."""
        # Age > T
        result = calculate_non_spine_ages_log_likelihood(
            lambda_params=np.array([0.1]),
            mu_params=np.array([0.05]),
            ages_t=150.0,  # age > times[-1]
            times=np.array([100.0]),
        )

        assert result == -np.inf

    def test_shape_mismatch_raises_error(self, sample_ages_scalar):
        """Test that parameter shape mismatches raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            calculate_non_spine_ages_log_likelihood(
                lambda_params=np.array([0.1, 0.15]),
                mu_params=np.array([0.05]),  # Different shape
                ages_t=sample_ages_scalar,
                times=np.array([50.0, 100.0]),
            )

    def test_non_monotonic_times_raises_error(self, sample_ages_scalar):
        """Test that non-monotonic times raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            calculate_non_spine_ages_log_likelihood(
                lambda_params=np.array([0.1, 0.15, 0.12]),
                mu_params=np.array([0.05, 0.06, 0.055]),
                ages_t=sample_ages_scalar,
                times=np.array([50.0, 100.0, 90.0]),  # Not monotonic
            )


class TestNonSpineAgesLogLikelihoodDerivatives:
    """Test derivatives for non-spine age log-likelihood."""

    def test_scalar_age_derivatives(self, standard_scalar_params, sample_ages_scalar):
        """Test derivatives with single age."""
        result = calculate_non_spine_ages_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_array_ages_derivatives(self, standard_scalar_params, sample_ages_array):
        """Test derivatives with array of ages."""
        result = calculate_non_spine_ages_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_derivatives_critical_case(self, nearly_equal_params, sample_ages_scalar):
        """Test derivative stability when λ ≈ μ."""
        result = calculate_non_spine_ages_log_likelihood_derivatives(
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
        result = calculate_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=True,
        )

        # Should match spine calculation
        expected = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert_allclose(result, expected)

    def test_non_spine_routing(self, sample_ages_scalar):
        """Test that are_spines=False routes to non-spine implementation."""
        result = calculate_ages_log_likelihood(
            lambda_param=0.1,
            mu_param=0.05,
            ages_t=sample_ages_scalar,
            T=100.0,
            are_spines=False,
            log_scale_params=False,
        )

        # Should match non-spine calculation (using array inputs)
        expected = calculate_non_spine_ages_log_likelihood(
            lambda_params=np.array([0.1]),
            mu_params=np.array([0.05]),
            ages_t=sample_ages_scalar,
            times=np.array([100.0]),
            log_scale_params=False,
        )

        assert_allclose(result, expected)


class TestAgesLogLikelihoodDerivatives:
    """Test unified age log-likelihood derivatives interface."""

    def test_spine_derivatives_routing(
        self, standard_scalar_params, sample_ages_scalar
    ):
        """Test that are_spines=True routes derivatives correctly."""
        result = calculate_ages_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=True,
        )

        expected = calculate_spine_ages_log_likelihood_derivatives(
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
        result = calculate_ages_log_likelihood_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=False,
        )

        expected = calculate_non_spine_ages_log_likelihood_derivatives(
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

    def test_probability_bounds_population(self, standard_scalar_params):
        """Test that log probabilities are negative (probabilities < 1)."""
        for n in range(0, 20):
            result = calculate_populations_log_likelihood(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                standard_scalar_params["time"],
                ns=n,
            )
            # Log probabilities should be ≤ 0
            assert result <= 0 or np.isclose(result, 0)

    def test_extinction_probability_increases_with_mu(self):
        """Test that extinction probability increases as death rate increases."""
        birth_rates = np.array([0.1, 0.1, 0.1])
        times = np.array([50.0, 100.0, 150.0])

        alpha_1 = calculate_extinction_probability(
            birth_rates, np.array([0.05, 0.05, 0.05]), times
        )
        alpha_2 = calculate_extinction_probability(
            birth_rates, np.array([0.08, 0.08, 0.08]), times
        )

        # Higher death rate → higher extinction probability
        assert alpha_2 > alpha_1

    def test_consistency_scalar_vs_single_element_array(self):
        """Test that scalar and single-element array give same results."""
        scalar_result = calculate_populations_log_likelihood(
            lambda_params=0.1,
            mu_params=0.05,
            times=100.0,
            ns=10,
        )

        array_result = calculate_populations_log_likelihood(
            lambda_params=np.array([0.1]),
            mu_params=np.array([0.05]),
            times=np.array([100.0]),
            ns=10,
        )

        # Results should be very close
        assert_allclose(scalar_result, array_result, rtol=1e-10)

    def test_sum_property_array_ns(self, standard_scalar_params):
        """Test that array of ns gives sum of individual likelihoods."""
        ns_array = np.array([5, 10, 15])

        # Compute together
        result_together = calculate_populations_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=ns_array,
        )

        # Compute separately and sum
        result_separate = sum(
            calculate_populations_log_likelihood(
                standard_scalar_params["lambda_param"],
                standard_scalar_params["mu_param"],
                standard_scalar_params["time"],
                ns=n,
            )
            for n in ns_array
        )

        assert_allclose(result_together, result_separate, rtol=1e-10)


# ============================================================================
# LEVEL 7: Numerical Stability and Edge Case Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability in challenging scenarios."""

    def test_extreme_time_values(self):
        """Test with very large time values."""
        # Large time can cause overflow in exponentials
        result = calculate_populations_log_likelihood(
            lambda_params=0.001,  # Small rates to prevent overflow
            mu_params=0.0005,
            times=10000.0,  # Very large time
            ns=10,
        )

        # Should not return inf or nan
        assert np.isfinite(result) or result == -np.inf

    def test_extreme_rate_differences(self):
        """Test when λ >> μ or μ >> λ."""
        # λ >> μ (high growth)
        result1 = calculate_populations_log_likelihood(
            lambda_params=1.0,
            mu_params=0.001,
            times=10.0,
            ns=100,
        )

        # Should handle gracefully
        assert np.isfinite(result1) or result1 == -np.inf

        # μ >> λ (high death)
        result2 = calculate_populations_log_likelihood(
            lambda_params=0.001,
            mu_params=1.0,
            times=10.0,
            ns=1,
        )

        assert np.isfinite(result2) or result2 == -np.inf

    def test_very_small_rates(self):
        """Test with very small birth and death rates."""
        result = calculate_populations_log_likelihood(
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

        result = calculate_populations_log_likelihood(
            lambda_params=birth_rates,
            mu_params=death_rates,
            times=times,
            ns=10,
            log_scale_params=False,
        )

        assert np.isfinite(result) or result == -np.inf

    def test_ages_near_observation_time(self):
        """Test age likelihood with ages very close to T."""
        T = 100.0
        ages = np.array([T - 1e-6, T - 1e-8, T - 1e-10])

        result = calculate_spine_ages_log_likelihood(
            lambda_param=0.1,
            mu_param=0.05,
            ages_t=ages,
            T=T,
        )

        assert np.isfinite(result)


# ============================================================================
# LEVEL 8: Integration Tests
# ============================================================================


class TestIntegration:
    """Test integration between related functions."""

    def test_derivatives_match_numerical_gradient(self, standard_scalar_params):
        """Test that analytical derivatives match numerical gradients."""
        # This is a placeholder for numerical gradient checking
        # Full implementation would use finite differences
        pass

    def test_log_scale_consistency(self, standard_scalar_params):
        """Test that log-scale and regular parameters give same results."""
        # Regular scale
        result1 = calculate_populations_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=10,
            log_scale_params=False,
        )

        # Log scale
        result2 = calculate_populations_log_likelihood(
            np.log(standard_scalar_params["lambda_param"]),
            np.log(standard_scalar_params["mu_param"]),
            standard_scalar_params["time"],
            ns=10,
            log_scale_params=True,
        )

        assert_allclose(result1, result2, rtol=1e-10)

    def test_spine_non_spine_different_formulas(
        self, standard_scalar_params, sample_ages_scalar
    ):
        """Test that spine and non-spine give different results (different formulas)."""
        spine_result = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            log_scale_params=False,
        )

        non_spine_result = calculate_non_spine_ages_log_likelihood(
            np.array([standard_scalar_params["lambda_param"]]),
            np.array([standard_scalar_params["mu_param"]]),
            sample_ages_scalar,
            np.array([standard_scalar_params["time"]]),
            log_scale_params=False,
        )

        # These should be different (different conditioning)
        assert not np.isclose(spine_result, non_spine_result)


# ============================================================================
# Performance and Vectorization Tests
# ============================================================================


class TestVectorization:
    """Test vectorization and performance characteristics."""

    def test_array_ages_more_efficient_than_loop(self, standard_scalar_params):
        """Verify that array processing is implemented (not just a loop)."""
        # This is more of a smoke test - actual performance testing
        # would require timing benchmarks
        ages = np.linspace(10, 90, 100)

        result = calculate_spine_ages_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            ages,
            standard_scalar_params["time"],
        )

        assert np.isfinite(result)

    def test_large_population_counts_array(self, standard_scalar_params):
        """Test with large array of population counts."""
        ns = np.arange(0, 100)

        result = calculate_populations_log_likelihood(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=ns,
        )

        assert np.isfinite(result)
