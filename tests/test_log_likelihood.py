"""Test suite for BirthDeathProcess log-likelihood calculations."""

import re
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bdpinf.inference.likelihood import log_likelihood
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
)
from bdpinf.models.math_populations import calculate_populations_log_lik
from bdpinf.process import BirthDeathProcess


def _as_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


_VALID_CALLS: dict[str, tuple[dict[str, Any], float]] = {
    "explicit_spine": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [10.0],
        },
        -20.3580373516016,
    ),
    "explicit_spine_interior": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [9.0],
        },
        -21.358003642599236,
    ),
    "ordered": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "is_ordered": True,
        },
        -19.272185697492247,
    ),
    "unknown_spine": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
        },
        -16.920659590678973,
    ),
    "age_at_T_auto_spine": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 10.0],
            "times": 10.0,
        },
        -11.022881124012265,
    ),
    "single_age": (
        {"birth_rates": 0.5, "death_rates": 0.3, "ages": [3.0], "times": 10.0},
        -4.815155091167325,
    ),
    "single_age_scalar": (
        {"birth_rates": 0.5, "death_rates": 0.3, "ages": 3.0, "times": 10.0},
        -4.815155091167325,
    ),
    "ns_equal": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "ns": [5.0],
        },
        -16.920659590678973,
    ),
    "ns_greater": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "ns": [8.0],
        },
        -17.102835719770756,
    ),
    "ns_two_groups_greater": (
        {
            "birth_rates": [0.5, 0.4],
            "death_rates": [0.3, 0.3],
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": [5.0, 10.0],
            "ns": [4.0, 6.0],
        },
        -18.58361466533383,
    ),
    "log_scale": (
        {
            "birth_rates": -0.7,
            "death_rates": -1.2,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "log_scale_params": True,
        },
        -16.874234351397412,
    ),
    "initial_population_3": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "initial_population": 3,
        },
        -16.920659590678973,
    ),
    "no_conditioning": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "condition_on_survival": False,
        },
        -17.752262241048197,
    ),
    "multi_epoch": (
        {
            "birth_rates": [0.5, 0.4],
            "death_rates": [0.3, 0.35],
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": [4.0, 10.0],
        },
        -15.633415541864018,
    ),
    "multi_epoch_ordered": (
        {
            "birth_rates": [0.5, 0.4],
            "death_rates": [0.3, 0.35],
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": [4.0, 10.0],
            "is_ordered": True,
        },
        -17.601525475028314,
    ),
    "multi_spine_with_ns": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [7.0, 10.0],
            "ns": [7.0],
        },
        -23.089599754340234,
    ),
    "empty_spine_treated_as_none": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [],
        },
        -16.920659590678973,
    ),
    "empty_ages": (
        {"birth_rates": 0.5, "death_rates": 0.3, "ages": [], "times": 10.0},
        -0.5715510001299191,
    ),
    "nonpositive_rate_inf": (
        {
            "birth_rates": 0.0,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
        },
        float("-inf"),
    ),
    "negative_rate_inf": (
        {
            "birth_rates": 0.5,
            "death_rates": -0.1,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
        },
        float("-inf"),
    ),
    "nonfinite_rate_inf": (
        {
            "birth_rates": float("inf"),
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
        },
        float("-inf"),
    ),
    "log_scale_overflow_inf": (
        {
            "birth_rates": 1000.0,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "log_scale_params": True,
        },
        float("-inf"),
    ),
    "ordered_ns_greater": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "is_ordered": True,
            "ns": [9.0],
        },
        -19.51508720294796,
    ),
    "spine_equal_T_with_ns_none": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [],
            "times": 10.0,
            "spine_ages": [10.0],
        },
        -5.0,
    ),
}

_INVALID_CALLS: dict[str, tuple[dict[str, Any], str | None]] = {
    # value is the exact ValueError message, or None where the call returns -inf
    "shape_mismatch": (
        {
            "birth_rates": [0.5, 0.4],
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
        },
        "birth_rates, death_rates, and times must have the same shape, got (2,), (1,), (1,).",
    ),
    "ordered_and_spine": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [10.0],
            "is_ordered": True,
        },
        "Cannot have both is_ordered=True and spine_ages provided.",
    ),
    "multi_spine_no_ns": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [7.0, 10.0],
        },
        "When multiple spine ages are provided, population sizes (ns) must also be provided.",
    ),
    "spine_negative": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [-1.0],
        },
        "Spine ages must be in the range [0, 10.0].",
    ),
    "spine_above_T": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [11.0],
        },
        "Spine ages must be in the range [0, 10.0].",
    ),
    "times_nonpositive": (
        {"birth_rates": 0.5, "death_rates": 0.3, "ages": [], "times": 0.0},
        "Observation times must be positive.",
    ),
    "times_decreasing": (
        {
            "birth_rates": [0.5, 0.4],
            "death_rates": [0.3, 0.3],
            "ages": [1.0, 2.0],
            "times": [10.0, 5.0],
        },
        "Observation times must be non-decreasing.",
    ),
    "age_negative": (
        {"birth_rates": 0.5, "death_rates": 0.3, "ages": [-1.0, 2.0], "times": 10.0},
        "Ages must be in the range [0, 10.0].",
    ),
    "age_above_T": (
        {"birth_rates": 0.5, "death_rates": 0.3, "ages": [1.0, 12.0], "times": 10.0},
        "Ages must be in the range [0, 10.0].",
    ),
    "ns_too_small": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "ns": [3.0],
        },
        "Sum of the populations must equal the number of ages provided.",
    ),
    "two_ages_at_T": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 10.0, 10.0],
            "times": 10.0,
        },
        "More than one age equals the final observation time. Exactly one spine is assumed; provide explicit spine_ages.",
    ),
    "zero_rate_log_scale": (
        {
            "birth_rates": -1000.0,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "log_scale_params": True,
        },
        None,
    ),
    "dbl_shape_over_rate": (
        {
            "birth_rates": [-0.5, 0.4],
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
        },
        "birth_rates, death_rates, and times must have the same shape, got (2,), (1,), (1,).",
    ),
    "dbl_rate_over_ordered_spine": (
        {
            "birth_rates": -0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [10.0],
            "is_ordered": True,
        },
        None,
    ),
    "dbl_ordered_spine_over_times": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [],
            "times": 0.0,
            "spine_ages": [0.0],
            "is_ordered": True,
        },
        "Cannot have both is_ordered=True and spine_ages provided.",
    ),
    "dbl_multi_spine_ns_over_range": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "spine_ages": [7.0, 11.0],
        },
        "When multiple spine ages are provided, population sizes (ns) must also be provided.",
    ),
    "dbl_spine_range_over_times": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [],
            "times": -1.0,
            "spine_ages": [0.0],
        },
        "Spine ages must be in the range [0, -1.0].",
    ),
    "dbl_times_pos_over_decreasing": (
        {
            "birth_rates": [0.5, 0.4],
            "death_rates": [0.3, 0.3],
            "ages": [],
            "times": [-1.0, -2.0],
        },
        "Observation times must be positive.",
    ),
    "dbl_decreasing_over_age_range": (
        {
            "birth_rates": [0.5, 0.4],
            "death_rates": [0.3, 0.3],
            "ages": [4.0],
            "times": [5.0, 3.0],
        },
        "Observation times must be non-decreasing.",
    ),
    "dbl_age_range_over_ns": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 12.0],
            "times": 10.0,
            "ns": [1.0],
        },
        "Ages must be in the range [0, 10.0].",
    ),
    "dbl_ns_over_nonfinite_rate": (
        {
            "birth_rates": float("inf"),
            "death_rates": 0.3,
            "ages": [1.0, 2.5, 4.0, 6.5, 8.0],
            "times": 10.0,
            "ns": [3.0],
        },
        "Sum of the populations must equal the number of ages provided.",
    ),
    "dbl_nonfinite_over_two_terminal": (
        {
            "birth_rates": float("inf"),
            "death_rates": 0.3,
            "ages": [1.0, 10.0, 10.0],
            "times": 10.0,
        },
        None,
    ),
    "dbl_rate_over_age_range": (
        {"birth_rates": 0.0, "death_rates": 0.3, "ages": [1.0, 12.0], "times": 10.0},
        None,
    ),
    "dbl_spine_range_over_ages_range": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [12.0],
            "times": 10.0,
            "spine_ages": [11.0],
        },
        "Spine ages must be in the range [0, 10.0].",
    ),
    "dbl_warn_then_two_terminal": (
        {
            "birth_rates": 0.5,
            "death_rates": 0.3,
            "ages": [1.0, 10.0, 10.0],
            "times": 10.0,
            "ns": [6.0],
        },
        "More than one age equals the final observation time. Exactly one spine is assumed; provide explicit spine_ages.",
    ),
}


class TestLogLikelihood:
    """Test log-likelihood calculations."""

    def test_log_likelihood_empty_ages(self) -> None:
        """Test log-likelihood with no ages raises ValueError."""
        process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=1000.0,
            birth_rates=0.1,
            death_rates=0.01,
        )

        ages = process.ages
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

        ages = process.ages
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

        ages = process.ages
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
            birth_rates=0.01,
            death_rates=0.005,
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        # Use explicit parameters
        ages = process.ages
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
            birth_rates=0.01,
            death_rates=0.005,
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        # These should be identical
        ages = process.ages
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

        ages = process.ages

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
        ages = process.ages
        result = -log_likelihood(
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
        process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=1000.0,
            birth_rates=0.1,
            death_rates=0.01,
        )

        # Use log-scale parameters as expected by negative_log_likelihood
        birth_rate, death_rate = 0.01, 0.005
        ages = process.ages
        result = -log_likelihood(
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
        process = BirthDeathProcess(creation_times=ages_data, epoch_end_times=500.0)
        ages = process.ages
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
        process = BirthDeathProcess(
            creation_times=creation_times,
            epoch_end_times=500.0,
        )
        ages = process.ages
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

        ages = process.ages

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
        ages_large = process_large.ages
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
        ns = np.array([2, 1])  # 2 records in first group, 1 in second

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
        # Relaxed tolerance: critical (λ≈μ) path has larger rounding in JAX multi-epoch
        assert_allclose(result_single, result_multi, rtol=1e-7)

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
        result_single = -log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=final_time,
        )

        # Multi-epoch with identical rates
        epoch_times = np.array([200.0, 400.0])
        birth_rates_multi = np.array([birth_rate, birth_rate])
        death_rates_multi = np.array([death_rate, death_rate])

        result_multi = -log_likelihood(
            birth_rates=birth_rates_multi,
            death_rates=death_rates_multi,
            ages=ages,
            times=epoch_times,
        )

        assert np.isfinite(_as_float(result_single))
        assert np.isfinite(_as_float(result_multi))
        # Relaxed tolerance due to known multi-epoch numerical differences
        assert_allclose(result_single, result_multi, rtol=1e-8)

    def test_unknown_spine_default_near_critical_stable(self) -> None:
        """Unknown-spine default branch remains stable for birth_rate ≈ death_rate."""
        ages = np.array([120.0, 240.0, 360.0])
        birth_rate = 0.01
        death_rate = 0.01 + 1e-12

        result = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages,
            times=500.0,
        )

        assert np.isfinite(_as_float(result))
        assert result < 0

    def test_unknown_spine_default_differs_from_all_non_spine(self) -> None:
        """Default unknown-spine branch should differ from old all-non-spine age sum."""
        birth_rates = np.array([0.02])
        death_rates = np.array([0.007])
        ages = np.array([100.0, 180.0, 320.0])
        times = np.array([500.0])

        default_result = log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        all_non_spine_result = (
            calculate_populations_log_lik(
                birth_rates=birth_rates,
                death_rates=death_rates,
                times=times,
                ns=np.array([len(ages)]),
                initial_population=1,
            )
            + non_spine_log_density(
                nested_terms(birth_rates, death_rates, times), ages
            ).sum()
        )

        assert not np.isclose(
            _as_float(default_result),
            _as_float(all_non_spine_result),
        )


class TestRecordedCalls:
    """Pin ``log_likelihood`` on valid and invalid calls recorded before refactoring."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("name", list(_VALID_CALLS))
    def test_valid_calls_match_recording(self, name: str) -> None:
        kwargs, expected = _VALID_CALLS[name]
        result = _as_float(log_likelihood(**kwargs))
        if np.isinf(expected):
            assert result == expected
        else:
            assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("name", list(_INVALID_CALLS))
    def test_invalid_calls_match_recording(self, name: str) -> None:
        kwargs, message = _INVALID_CALLS[name]
        if message is None:
            assert _as_float(log_likelihood(**kwargs)) == -np.inf
        else:
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                log_likelihood(**kwargs)

    def test_partial_age_warning_text_category_and_location(self) -> None:
        kwargs, _ = _VALID_CALLS["ns_greater"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            log_likelihood(**kwargs)
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        assert str(caught[0].message) == (
            "Sum of the populations exceeds the number of ages provided; "
            "not all ages have been provided (age information is partial)."
        )
        assert caught[0].filename == __file__

    def test_partial_age_warning_precedes_later_error(self) -> None:
        kwargs, message = _INVALID_CALLS["dbl_warn_then_two_terminal"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match=f"^{re.escape(message or '')}$"):
                log_likelihood(**kwargs)
        assert len(caught) == 1
