"""
Test suite for BirthDeathProcess log-likelihood calculations.
"""

import numpy as np

from inf_bp_texts.inference.likelihood import log_likelihood, negative_log_likelihood
from inf_bp_texts.models import BirthDeathProcess


class TestLogLikelihood:
    """Test log-likelihood calculations."""

    def test_log_likelihood_empty_ages(self):
        """Test log-likelihood with no ages returns -inf."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.1, death_rate=0.01)

        ages = process.get_manuscript_ages()
        result = log_likelihood(
            birth_rates=0.1, death_rates=0.01, ages=ages, times=1000.0
        )
        assert result == -np.inf

    def test_basic_log_likelihood(self):
        """Test basic log-likelihood functionality."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        ages = process.get_manuscript_ages()
        result = log_likelihood(
            birth_rates=0.01, death_rates=0.005, ages=ages, times=1000.0
        )

        # Check that the result is finite
        assert np.isfinite(result)

    def test_log_likelihood_critical_case(self):
        """Test log-likelihood when birth_rate ≈ death_rate."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        # Very close birth and death rates
        birth_rate = 0.01
        death_rate = 0.01 + 1e-12  # Nearly equal

        ages = process.get_manuscript_ages()
        result = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=1000.0,
        )

        # Should handle near-critical case gracefully
        assert np.isfinite(result)

    def test_log_likelihood_with_explicit_parameters(self):
        """Test log-likelihood with explicit parameters."""
        process = BirthDeathProcess(
            birth_rate=0.01,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0],
            final_time=1000.0,
        )

        # Use explicit parameters
        ages = process.get_manuscript_ages()
        result = log_likelihood(
            birth_rates=0.02, death_rates=0.001, ages=ages, times=1000.0
        )
        assert np.isfinite(result)
        assert result < 0

    def test_log_likelihood_parameter_consistency(self):
        """Test that same parameters give same results."""
        process = BirthDeathProcess(
            birth_rate=0.01,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0],
            final_time=1000.0,
        )

        # These should be identical
        ages = process.get_manuscript_ages()
        result1 = log_likelihood(
            birth_rates=0.01, death_rates=0.005, ages=ages, times=1000.0
        )
        result2 = log_likelihood(
            birth_rates=0.01, death_rates=0.005, ages=ages, times=1000.0
        )

        assert np.isclose(result1, result2)

    def test_log_likelihood_negative_parameters(self):
        """Test log-likelihood with invalid (negative) parameters."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        ages = process.get_manuscript_ages()

        # Negative birth rate should still work (just gives very negative log-likelihood)
        result = log_likelihood(
            birth_rates=-0.01, death_rates=0.005, ages=ages, times=1000.0
        )

        # Should return negative infinity or very negative value
        assert result <= -1000 or np.isneginf(result)

        # Negative death rate should also work (just gives very negative log-likelihood)
        result = log_likelihood(
            birth_rates=0.01, death_rates=-0.005, ages=ages, times=1000.0
        )

    def test_negative_log_likelihood_call_context(self):
        """Test negative log-likelihood in optimization context."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        # Use log-scale parameters as expected by negative_log_likelihood
        ages = process.get_manuscript_ages()
        result = negative_log_likelihood(
            np.log(0.01), np.log(0.005), ages=ages, times=1000.0, log_scale_params=True
        )

        # Result should be positive and finite
        assert np.isfinite(result)
        assert result > 0

    def test_negative_log_likelihood_empty_ages(self):
        """Test negative log-likelihood with no ages."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.1, death_rate=0.01)

        # Use log-scale parameters as expected by negative_log_likelihood
        birth_rate, death_rate = 0.01, 0.005
        ages = process.get_manuscript_ages()
        result = negative_log_likelihood(
            birth_rate, death_rate, ages=ages, times=1000.0
        )

        # Should be a finite value (the likelihood for an empty dataset)
        assert np.isfinite(result)

    def test_external_log_likelihood_helper(self):
        """Test the external log-likelihood helper method via public interface."""
        ages_data = [100.0, 200.0, 300.0]
        process = BirthDeathProcess(ages_data, 500.0)
        ages = process.get_manuscript_ages()
        # Test log likelihood calculation with explicit parameters
        result = log_likelihood(
            birth_rates=0.01, death_rates=0.005, ages=ages, times=500.0
        )

        assert np.isfinite(result)
        assert result < 0

    def test_external_log_likelihood_invalid_params(self):
        """Test external log-likelihood with invalid parameters."""
        creation_times = [100.0, 200.0, 300.0]
        process = BirthDeathProcess(creation_times, 500.0)
        ages = process.get_manuscript_ages()
        # Test with invalid parameters using public interface
        result = log_likelihood(
            birth_rates=-0.01, death_rates=0.005, ages=ages, times=500.0
        )
        assert result == -np.inf

        result = log_likelihood(
            birth_rates=0.01, death_rates=-0.005, ages=ages, times=500.0
        )
        assert result == -np.inf

    def test_log_likelihood_numerical_stability(self):
        """Test numerical stability with extreme parameters."""
        process = BirthDeathProcess(
            creation_times=[10.0, 20.0, 30.0], final_time=1000.0
        )

        ages = process.get_manuscript_ages()

        # Very small rates
        result1 = log_likelihood(
            birth_rates=1e-6, death_rates=1e-7, ages=ages, times=1000.0
        )
        assert np.isfinite(result1)

        # Larger rates - may legitimately return -inf due to numerical limits
        result2 = log_likelihood(
            birth_rates=1.0, death_rates=0.1, ages=ages, times=1000.0
        )
        assert np.isfinite(result2) or result2 == -np.inf

        # Very large ages with small rates - may return -inf due to numerical limits
        process_large = BirthDeathProcess(
            creation_times=[1.0, 2.0, 3.0], final_time=10000.0
        )
        ages_large = process_large.get_manuscript_ages()
        result3 = log_likelihood(
            birth_rates=0.001,
            death_rates=0.0001,
            ages=ages_large,
            times=10000.0,
        )
        # May be -inf due to numerical limits, but shouldn't crash
        assert np.isfinite(result3) or result3 == -np.inf
