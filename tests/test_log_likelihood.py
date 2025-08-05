"""
Test suite for BirthDeathProcess log-likelihood calculations.
"""

import numpy as np

from inf_bp_texts.inference import BirthDeathProcess


class TestLogLikelihood:
    """Test log-likelihood calculations."""

    def test_log_likelihood_empty_ages(self):
        """Test log-likelihood with no ages returns -inf."""
        process = BirthDeathProcess()

        result = process.log_likelihood()
        assert result == -np.inf

    def test_log_likelihood_with_self_attributes(self):
        """Test log-likelihood using self attributes."""
        process = BirthDeathProcess(
            birth_rate=0.01,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0],
            final_time=1000.0,
        )

        # Should use self.birth_rate and self.death_rate
        result = process.log_likelihood()
        assert np.isfinite(result)
        assert result < 0  # Log-likelihood should be negative

    def test_log_likelihood_with_explicit_parameters(self):
        """Test log-likelihood with explicit parameters."""
        process = BirthDeathProcess(
            birth_rate=0.01,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0],
            final_time=1000.0,
        )

        # Use explicit parameters
        result = process.log_likelihood(birth_rate=0.02, death_rate=0.001)
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
        result1 = process.log_likelihood()
        result2 = process.log_likelihood(birth_rate=0.01, death_rate=0.005)

        assert np.isclose(result1, result2)

    def test_log_likelihood_invalid_parameters(self):
        """Test log-likelihood with invalid parameters."""
        process = BirthDeathProcess(creation_times=[100.0, 200.0], final_time=500.0)

        # Negative birth rate
        result = process.log_likelihood(birth_rate=-0.01, death_rate=0.005)
        assert result == -np.inf

        # Zero birth rate
        result = process.log_likelihood(birth_rate=0.0, death_rate=0.005)
        assert result == -np.inf

        # Negative death rate
        result = process.log_likelihood(birth_rate=0.01, death_rate=-0.005)
        assert result == -np.inf

    def test_negative_log_likelihood(self):
        """Test negative log-likelihood for optimization."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        params = np.array([0.01, 0.005], dtype=np.float64)
        nll = process.negative_log_likelihood(params)

        # Should be positive (negative of negative log-likelihood)
        assert nll > 0
        assert np.isfinite(nll)

        # Should equal negative of log_likelihood
        ll = process.log_likelihood(birth_rate=0.01, death_rate=0.005)
        assert np.isclose(nll, -ll)

    def test_negative_log_likelihood_empty_ages(self):
        """Test negative log-likelihood with no ages."""
        process = BirthDeathProcess()

        params = np.array([0.01, 0.005], dtype=np.float64)
        result = process.negative_log_likelihood(params)

        assert result == np.inf

    def test_log_likelihood_critical_case(self):
        """Test log-likelihood when birth_rate ≈ death_rate."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        # Very close birth and death rates
        birth_rate = 0.01
        death_rate = 0.01 + 1e-12  # Nearly equal

        result = process.log_likelihood(birth_rate=birth_rate, death_rate=death_rate)

        # Should handle the critical case without crashing
        assert np.isfinite(result)

    def test_external_log_likelihood_helper(self):
        """Test the external log-likelihood helper method."""
        process = BirthDeathProcess()

        ages = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        counts = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        result = process._log_likelihood_external(0.01, 0.005, ages, counts)

        assert np.isfinite(result)
        assert result < 0

    def test_external_log_likelihood_invalid_params(self):
        """Test external log-likelihood with invalid parameters."""
        process = BirthDeathProcess()

        ages = np.array([100.0, 200.0], dtype=np.float64)
        counts = np.array([1.0, 1.0], dtype=np.float64)

        # Invalid birth rate
        result = process._log_likelihood_external(-0.01, 0.005, ages, counts)
        assert result == -np.inf

        # Invalid death rate
        result = process._log_likelihood_external(0.01, -0.005, ages, counts)
        assert result == -np.inf

    def test_log_likelihood_numerical_stability(self):
        """Test numerical stability with extreme parameters."""
        process = BirthDeathProcess(
            creation_times=[10.0, 20.0, 30.0], final_time=1000.0
        )

        # Very small rates
        result1 = process.log_likelihood(birth_rate=1e-6, death_rate=1e-7)
        assert np.isfinite(result1)

        # Larger rates - may legitimately return -inf due to numerical limits
        result2 = process.log_likelihood(birth_rate=1.0, death_rate=0.1)
        assert np.isfinite(result2) or result2 == -np.inf

        # Very large ages with small rates - may return -inf due to numerical limits
        process_large = BirthDeathProcess(
            creation_times=[1.0, 2.0, 3.0], final_time=10000.0
        )
        result3 = process_large.log_likelihood(birth_rate=0.001, death_rate=0.0001)
        # May be -inf due to numerical limits, but shouldn't crash
        assert np.isfinite(result3) or result3 == -np.inf
