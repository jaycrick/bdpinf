"""
Test suite for BirthDeathProcess utility functions and helper methods.
"""

import numpy as np
import pytest

from inf_bp_texts.inference.likelihood import log_likelihood
from inf_bp_texts.models import BirthDeathProcess


class TestUtilityFunctions:
    """Test utility functions and helper methods."""

    def test_predict_method(self):
        """Test prediction method."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.1, death_rate=0.01)

        ages = np.array([10.0, 50.0, 100.0, 200.0])
        predictions = process.predict(ages)

        # Should return array of same length
        assert len(predictions) == len(ages)

        # All predictions should be positive
        assert all(pred > 0 for pred in predictions)

        # Predictions should decrease with age (exponential decay)
        assert predictions[0] > predictions[1] > predictions[2] > predictions[3]

    def test_predict_unfitted_model_warning(self):
        """Test that prediction on unfitted model raises an error."""
        process = BirthDeathProcess([], 1000.0)  # No parameters provided, so unfitted
        ages = np.array([100.0, 200.0])

        # Should raise ValueError for unfitted model
        with pytest.raises(
            ValueError, match="Cannot predict without fitted birth_rate and death_rate"
        ):
            process.predict(ages)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_creation_times(self):
        """Test behavior with empty creation times."""
        process = BirthDeathProcess([], 1000.0)

        assert len(process.creation_times) == 0
        assert len(process.get_manuscript_ages()) == 0

        # Should handle empty state gracefully
        summary = process.get_process_summary()
        assert summary["n_manuscripts"] == 0

    def test_zero_rates(self):
        """Test behavior with zero rates."""
        # Zero birth rate should be invalid for log-likelihood
        process = BirthDeathProcess([], 1000.0, birth_rate=0.0, death_rate=0.01)
        process.creation_times = [100.0]
        process.set_final_time(200.0)

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()
        ll = log_likelihood(0.0, 0.01, ages, process.final_time)
        assert ll == -np.inf

    def test_negative_rates(self):
        """Test behavior with negative rates."""
        process = BirthDeathProcess([], 1000.0)
        process.creation_times = [100.0]
        process.set_final_time(200.0)

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()

        # Negative rates should give -inf log-likelihood
        ll = log_likelihood(-0.01, 0.01, ages, process.final_time)
        assert ll == -np.inf

        ll = log_likelihood(0.01, -0.01, ages, process.final_time)
        assert ll == -np.inf

    def test_very_large_ages(self):
        """Test behavior with very large ages."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.001, death_rate=0.0001)
        process.creation_times = [1.0, 2.0, 3.0]
        process.set_final_time(100000.0)  # Very large final time

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()

        # Should handle large ages without crashing
        ll = log_likelihood(0.001, 0.0001, ages, process.final_time)
        # Result might be -inf due to numerical limits, but shouldn't crash
        assert np.isfinite(ll) or ll == -np.inf

    def test_single_manuscript(self):
        """Test behavior with single manuscript."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.01, death_rate=0.005)
        process.add_manuscript(100.0)
        process.set_final_time(500.0)

        assert len(process.get_manuscript_ages()) == 1
        assert process.get_manuscript_ages()[0] == 400.0

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()

        # Should be able to calculate log-likelihood
        ll = log_likelihood(0.01, 0.005, ages, process.final_time)
        assert np.isfinite(ll)

    def test_final_time_before_creation_time(self):
        """Test behavior when final_time is before creation_time."""
        process = BirthDeathProcess([], 1000.0)
        process.add_manuscript(1000.0)
        process.set_final_time(500.0)  # Before creation time

        # The manuscript should be filtered out (since creation time > final time)
        # So there should be no manuscripts left
        ages = process.get_manuscript_ages()
        assert len(ages) == 0
