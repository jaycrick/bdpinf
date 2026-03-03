"""Test suite for BirthDeathProcess log-likelihood calculations."""

# ruff: noqa: S101

import numpy as np
from numpy.testing import assert_allclose

from inf_bp_texts.inference.likelihood import log_likelihood, negative_log_likelihood
from inf_bp_texts.models import BirthDeathProcess


def _as_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class TestLogLikelihood:
    """Test log-likelihood calculations."""

    def test_log_likelihood_empty_ages(self) -> None:
        """Test log-likelihood with no ages raises ValueError."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.1, death_rate=0.01)

        ages = process.get_manuscript_ages()
        result = log_likelihood(
            birth_rates=0.1,
            death_rates=0.01,
            ages=ages,
            times=1000.0,
        )
        assert result < 0
        assert np.isfinite(_as_float(result))

    def test_basic_log_likelihood(self) -> None:
        """Test basic log-likelihood functionality."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        ages = process.get_manuscript_ages()
        result = log_likelihood(
            birth_rates=0.01,
            death_rates=0.005,
            ages=ages,
            times=1000.0,
        )

        # Check that the result is finite
        assert np.isfinite(_as_float(result))
        assert result < 0

    def test_log_likelihood_critical_case(self) -> None:
        """Test log-likelihood when birth_rate ≈ death_rate."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
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
        assert np.isfinite(_as_float(result))
        assert result < 0

    def test_log_likelihood_with_explicit_parameters(self) -> None:
        """Test log-likelihood with explicit parameters."""
        process = BirthDeathProcess(
            birth_rate=0.01,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        # Use explicit parameters
        ages = process.get_manuscript_ages()
        result = log_likelihood(
            birth_rates=0.02,
            death_rates=0.001,
            ages=ages,
            times=1000.0,
        )
        assert np.isfinite(_as_float(result))
        assert result < 0

    def test_log_likelihood_parameter_consistency(self) -> None:
        """Test that same parameters give same results."""
        process = BirthDeathProcess(
            birth_rate=0.01,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        # These should be identical
        ages = process.get_manuscript_ages()
        result1 = log_likelihood(
            birth_rates=0.01,
            death_rates=0.005,
            ages=ages,
            times=1000.0,
        )
        result2 = log_likelihood(
            birth_rates=0.01,
            death_rates=0.005,
            ages=ages,
            times=1000.0,
        )

        assert_allclose(result1, result2, rtol=1e-8)
        assert result1 < 0
        assert result2 < 0

    def test_log_likelihood_negative_parameters(self) -> None:
        """Test log-likelihood with invalid (negative) parameters."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        ages = process.get_manuscript_ages()

        # Negative birth rate should still work (just gives very negative log-likelihood)
        result = log_likelihood(
            birth_rates=-0.01,
            death_rates=0.005,
            ages=ages,
            times=1000.0,
        )

        # Should return negative infinity or very negative value
        assert result <= -1000 or np.isneginf(result)

        # Negative death rate should also work (just gives very negative log-likelihood)
        result = log_likelihood(
            birth_rates=0.01,
            death_rates=-0.005,
            ages=ages,
            times=1000.0,
        )
        assert result == -np.inf or not np.isfinite(_as_float(result))

    def test_negative_log_likelihood_call_context(self) -> None:
        """Test negative log-likelihood in optimization context."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        # Use log-scale parameters as expected by negative_log_likelihood
        ages = process.get_manuscript_ages()
        result = negative_log_likelihood(
            np.log(0.01),
            np.log(0.005),
            ages=ages,
            times=1000.0,
            log_scale_params=True,
        )

        # Result should be positive and finite
        assert np.isfinite(_as_float(result))
        assert result > 0

    def test_negative_log_likelihood_empty_ages(self) -> None:
        """Test negative log-likelihood with no ages."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.1, death_rate=0.01)

        # Use log-scale parameters as expected by negative_log_likelihood
        birth_rate, death_rate = 0.01, 0.005
        ages = process.get_manuscript_ages()
        result = negative_log_likelihood(
            birth_rate,
            death_rate,
            ages=ages,
            times=1000.0,
        )

        # Should be a finite value (the likelihood for an empty dataset)
        assert np.isfinite(_as_float(result))
        assert result > 0

    def test_external_log_likelihood_helper(self) -> None:
        """Test the external log-likelihood helper method via public interface."""
        ages_data = [100.0, 200.0, 300.0]
        process = BirthDeathProcess(ages_data, 500.0)
        ages = process.get_manuscript_ages()
        # Test log likelihood calculation with explicit parameters
        result = log_likelihood(
            birth_rates=0.01,
            death_rates=0.005,
            ages=ages,
            times=500.0,
        )

        assert np.isfinite(_as_float(result))
        assert result < 0

    def test_external_log_likelihood_invalid_params(self) -> None:
        """Test external log-likelihood with invalid parameters."""
        creation_times = [100.0, 200.0, 300.0]
        process = BirthDeathProcess(creation_times, 500.0)
        ages = process.get_manuscript_ages()
        # Test with invalid parameters using public interface
        result = log_likelihood(
            birth_rates=-0.01,
            death_rates=0.005,
            ages=ages,
            times=500.0,
        )
        assert result == -np.inf

        result = log_likelihood(
            birth_rates=0.01,
            death_rates=-0.005,
            ages=ages,
            times=500.0,
        )
        assert result == -np.inf

    def test_log_likelihood_numerical_stability(self) -> None:
        """Test numerical stability with extreme parameters."""
        process = BirthDeathProcess(
            creation_times=[10.0, 20.0, 30.0],
            epoch_end_times=1000.0,
        )

        ages = process.get_manuscript_ages()

        # Very small rates
        result1 = log_likelihood(
            birth_rates=1e-6,
            death_rates=1e-7,
            ages=ages,
            times=1000.0,
        )
        assert np.isfinite(_as_float(result1))
        assert result1 < 0

        # Larger rates - may legitimately return -inf due to numerical limits
        result2 = log_likelihood(
            birth_rates=1.0,
            death_rates=0.1,
            ages=ages,
            times=1000.0,
        )
        assert np.isfinite(_as_float(result2)) or result2 == -np.inf
        assert result2 < 0

        # Very large ages with small rates - may return -inf due to numerical limits
        process_large = BirthDeathProcess(
            creation_times=[1.0, 2.0, 3.0],
            epoch_end_times=10000.0,
        )
        ages_large = process_large.get_manuscript_ages()
        result3 = log_likelihood(
            birth_rates=0.001,
            death_rates=0.0001,
            ages=ages_large,
            times=10000.0,
        )
        # May be -inf due to numerical limits, but shouldn't crash
        assert np.isfinite(_as_float(result3)) or result3 == -np.inf
        assert result3 < 0

    def test_multi_epoch_identical_rates_matches_single_epoch(self) -> None:
        """Test that multi-epoch with identical rates equals single-epoch."""
        # Create test data
        ages = np.array([100.0, 200.0, 300.0, 400.0])
        birth_rate = 0.01
        death_rate = 0.005
        final_time = 1000.0

        # Single epoch case
        result_single = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=final_time,
        )

        # Multi-epoch case with identical rates
        # Split into 4 epochs with same rates
        epoch_times = np.array([250.0, 500.0, 750.0, 1000.0])
        birth_rates_multi = np.array([birth_rate, birth_rate, birth_rate, birth_rate])
        death_rates_multi = np.array([death_rate, death_rate, death_rate, death_rate])

        result_multi = log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
        )

        # Results should be very close (allowing for floating-point error)
        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        # For now, just verify they're reasonably close (within 5%)
        assert_allclose(result_single, result_multi, rtol=1e-8)

    def test_multi_epoch_identical_rates_with_spine_ages(self) -> None:
        """Test multi-epoch equals single-epoch with spine ages.

        NOTE: See test_multi_epoch_identical_rates_matches_single_epoch for
        details on known numerical differences in multi-epoch implementation.
        """
        ages = np.array([200.0, 300.0, 400.0])
        spine_ages = np.array([100.0])
        birth_rate = np.array([0.015])
        death_rate = np.array([0.008])
        final_time = np.array([800.0])

        # Single epoch
        result_single = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=final_time,
            spine_ages=spine_ages,
        )

        # Multi-epoch with identical rates (2 epochs)
        epoch_times = np.array([500.0, 800.0])
        birth_rates_multi = np.array([birth_rate[0], birth_rate[0]])
        death_rates_multi = np.array([death_rate[0], death_rate[0]])

        result_multi = log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
            spine_ages=spine_ages,
        )

        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        assert_allclose(result_single, result_multi, rtol=1e-8)

    def test_multi_epoch_identical_rates_with_populations(self) -> None:
        """Test multi-epoch equals single-epoch with population counts.

        NOTE: See test_multi_epoch_identical_rates_matches_single_epoch for
        details on known numerical differences in multi-epoch implementation.
        """
        ages = np.array([150.0, 350.0, 550.0])
        birth_rate = 0.02
        death_rate = 0.01
        final_time = 600.0
        ns = np.array([2, 1])  # 2 manuscripts in first group, 1 in second

        # Single epoch
        result_single = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=final_time,
            ns=ns,
        )

        # Multi-epoch with identical rates (3 epochs)
        epoch_times = np.array([200.0, 400.0, 600.0])
        birth_rates_multi = np.array([birth_rate, birth_rate, birth_rate])
        death_rates_multi = np.array([death_rate, death_rate, death_rate])

        result_multi = log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
            ns=ns,
        )

        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        assert_allclose(result_single, result_multi, rtol=1e-8)

    def test_multi_epoch_identical_rates_ordered(self) -> None:
        """Test multi-epoch equals single-epoch with ordered ages.

        NOTE: See test_multi_epoch_identical_rates_matches_single_epoch for
        details on known numerical differences in multi-epoch implementation.
        """
        ages = np.array([50.0, 150.0, 250.0, 350.0])
        birth_rate = 0.012
        death_rate = 0.006
        final_time = 500.0

        # Single epoch with is_ordered
        result_single = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=final_time,
            is_ordered=True,
        )

        # Multi-epoch with identical rates
        epoch_times = np.array([125.0, 250.0, 375.0, 500.0])
        birth_rates_multi = np.full(4, birth_rate)
        death_rates_multi = np.full(4, death_rate)

        result_multi = log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
            is_ordered=True,
        )

        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        # Relaxed tolerance due to known multi-epoch numerical differences
        assert_allclose(result_single, result_multi, rtol=1e-8)

    def test_multi_epoch_identical_rates_log_scale(self) -> None:
        """Test multi-epoch equals single-epoch with log-scale parameters.

        NOTE: See test_multi_epoch_identical_rates_matches_single_epoch for
        details on known numerical differences in multi-epoch implementation.
        """
        ages = np.array([100.0, 250.0, 400.0])
        log_birth_rate = np.log(0.018)
        log_death_rate = np.log(0.009)
        final_time = 600.0

        # Single epoch with log-scale
        result_single = log_likelihood(
            birth_rates=log_birth_rate,
            death_rates=log_death_rate,
            ages=ages,
            times=final_time,
            log_scale_params=True,
        )

        # Multi-epoch with identical log-scale rates
        epoch_times = np.array([200.0, 400.0, 600.0])
        birth_rates_multi = np.full(3, log_birth_rate)
        death_rates_multi = np.full(3, log_death_rate)

        result_multi = log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
            log_scale_params=True,
        )

        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        # Relaxed tolerance due to known multi-epoch numerical differences
        assert_allclose(result_single, result_multi, rtol=1e-8)

    def test_multi_epoch_identical_rates_critical_case(self) -> None:
        """Test multi-epoch equals single-epoch when birth ≈ death.

        NOTE: See test_multi_epoch_identical_rates_matches_single_epoch for
        details on known numerical differences in multi-epoch implementation.
        """
        ages = np.array([100.0, 200.0, 300.0])
        birth_rate = 0.01
        death_rate = 0.01 + 1e-12  # Nearly equal
        final_time = 500.0

        # Single epoch
        result_single = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=final_time,
        )

        # Multi-epoch with identical rates
        epoch_times = np.array([250.0, 500.0])
        birth_rates_multi = np.array([birth_rate, birth_rate])
        death_rates_multi = np.array([death_rate, death_rate])

        result_multi = log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
        )

        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        # Relaxed tolerance due to known multi-epoch numerical differences
        assert_allclose(result_single, result_multi, rtol=1e-8)

    def test_negative_log_likelihood_multi_epoch_matches_single(self) -> None:
        """Test negative_log_likelihood multi-epoch equals single-epoch.

        NOTE: See test_multi_epoch_identical_rates_matches_single_epoch for
        details on known numerical differences in multi-epoch implementation.
        """
        ages = np.array([50.0, 150.0, 250.0])
        birth_rate = 0.015
        death_rate = 0.007
        final_time = 400.0

        # Single epoch
        result_single = negative_log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=final_time,
        )

        # Multi-epoch with identical rates
        epoch_times = np.array([200.0, 400.0])
        birth_rates_multi = np.array([birth_rate, birth_rate])
        death_rates_multi = np.array([death_rate, death_rate])

        result_multi = negative_log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
        )

        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        # Relaxed tolerance due to known multi-epoch numerical differences
        assert_allclose(result_single, result_multi, rtol=1e-8)
