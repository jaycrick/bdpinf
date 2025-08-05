"""
Test suite for BirthDeathProcess utility functions and helper methods.
"""

import numpy as np
import pandas as pd
import pytest

from inf_bp_texts.inference import (
    BirthDeathProcess,
    compare_models,
    prepare_manuscript_data,
)


class TestUtilityFunctions:
    """Test utility functions and helper methods."""

    def test_prepare_manuscript_data_basic(self):
        """Test basic manuscript data preparation."""
        # Create test data
        df = pd.DataFrame(
            {
                "midYear": [1200, 1300, 1400, 1500, 1600],
                "manuscript_id": [1, 2, 3, 4, 5],
            }
        )

        current_year = 2000
        result = prepare_manuscript_data(df, current_year=current_year)

        # Should have age column
        assert "age" in result.columns

        # Ages should be calculated correctly
        expected_ages = [800, 700, 600, 500, 400]
        np.testing.assert_array_equal(result["age"].values, expected_ages)

        # Should have century and period columns
        assert "century" in result.columns
        assert "period" in result.columns

    def test_prepare_manuscript_data_custom_columns(self):
        """Test manuscript data preparation with custom column names."""
        df = pd.DataFrame({"date": [1100, 1200, 1300], "id": [1, 2, 3]})

        result = prepare_manuscript_data(df, current_year=2025, date_column="date")

        assert "age" in result.columns
        expected_ages = [925, 825, 725]
        np.testing.assert_array_equal(result["age"].values, expected_ages)

    def test_prepare_manuscript_data_removes_future_dates(self):
        """Test that future dates are removed."""
        df = pd.DataFrame(
            {
                "midYear": [1500, 2030, 1800, 2100],  # 2030 and 2100 are future
                "id": [1, 2, 3, 4],
            }
        )

        result = prepare_manuscript_data(df, current_year=2025)

        # Should only have 2 manuscripts (1500 and 1800)
        assert len(result) == 2
        assert all(result["age"] >= 0)

    def test_prepare_manuscript_data_periods(self):
        """Test historical period assignment."""
        df = pd.DataFrame({"midYear": [300, 750, 1200, 1600], "id": [1, 2, 3, 4]})

        result = prepare_manuscript_data(df, current_year=2000)

        expected_periods = [
            "Early Christian",
            "Early Medieval",
            "Late Medieval",
            "Post-Medieval",
        ]
        assert result["period"].tolist() == expected_periods

    def test_prepare_manuscript_data_centuries(self):
        """Test century calculation."""
        df = pd.DataFrame(
            {"midYear": [150, 250, 950, 1050, 1450, 1550], "id": [1, 2, 3, 4, 5, 6]}
        )

        result = prepare_manuscript_data(df, current_year=2000)

        expected_centuries = [
            2,
            3,
            10,
            11,
            15,
            16,
        ]  # Century calculation: floor(year/100) + 1
        np.testing.assert_array_equal(result["century"].values, expected_centuries)

    def test_compare_models_success(self):
        """Test successful model comparison."""
        # Create test data
        df = pd.DataFrame(
            {
                "age": [100.0, 200.0, 300.0, 400.0, 500.0],
                "count": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

        # Create different models
        model1 = BirthDeathProcess(birth_rate=0.01, death_rate=0.005)
        model2 = BirthDeathProcess(birth_rate=0.02, death_rate=0.001)

        models = [model1, model2]

        result = compare_models(models, df, count_column="count")

        # Should return DataFrame with model comparison
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Check required columns
        expected_cols = [
            "model_id",
            "birth_rate",
            "death_rate",
            "log_likelihood",
            "aic",
            "bic",
            "n_params",
            "success",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_compare_models_with_failures(self):
        """Test model comparison when some models fail."""
        # Create problematic data that might cause fitting to fail
        df = pd.DataFrame({"age": [1.0, 2.0], "count": [1.0, 1.0]})  # Very limited data

        # Create models that might fail
        model1 = BirthDeathProcess(birth_rate=0.001, death_rate=0.001)
        model2 = BirthDeathProcess(
            birth_rate=100.0, death_rate=50.0
        )  # Extreme parameters

        models = [model1, model2]

        result = compare_models(models, df, count_column="count")

        # Should still return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

        # Should have success column indicating failures
        assert "success" in result.columns

    def test_predict_method(self):
        """Test prediction method."""
        process = BirthDeathProcess(birth_rate=0.1, death_rate=0.01)

        ages = np.array([10.0, 50.0, 100.0, 200.0])
        predictions = process.predict(ages)

        # Should return array of same length
        assert len(predictions) == len(ages)

        # All predictions should be positive
        assert all(pred > 0 for pred in predictions)

        # Predictions should decrease with age (exponential decay)
        assert predictions[0] > predictions[1] > predictions[2] > predictions[3]

    def test_predict_unfitted_model_warning(self):
        """Test that prediction on unfitted model produces warning."""
        process = BirthDeathProcess()
        ages = np.array([100.0, 200.0])

        # Should work but may produce warning
        with pytest.warns(UserWarning, match="Model has not been fitted"):
            predictions = process.predict(ages)
            assert len(predictions) == len(ages)

    def test_predict_consistency(self):
        """Test prediction consistency with expected_manuscripts."""
        process = BirthDeathProcess(birth_rate=0.05, death_rate=0.02)

        age = 150.0
        time_window = 1.0

        # predict() uses birth_rate * exp(-death_rate * age)
        pred = process.predict(np.array([age]))[0]

        # expected_manuscripts() uses birth_rate * time_window * exp(-death_rate * age)
        expected = process.expected_manuscripts(age, time_window)

        # Should be identical when time_window = 1
        assert np.isclose(pred, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_creation_times(self):
        """Test behavior with empty creation times."""
        process = BirthDeathProcess()

        assert len(process.creation_times) == 0
        assert len(process.ages) == 0

        # Should handle empty state gracefully
        summary = process.get_process_summary()
        assert summary["n_manuscripts"] == 0

    def test_zero_rates(self):
        """Test behavior with zero rates."""
        # Zero birth rate should be invalid for log-likelihood
        process = BirthDeathProcess(birth_rate=0.0, death_rate=0.01)
        process.creation_times = [100.0]
        process.set_final_time(200.0)

        ll = process.log_likelihood()
        assert ll == -np.inf

    def test_negative_rates(self):
        """Test behavior with negative rates."""
        process = BirthDeathProcess()
        process.creation_times = [100.0]
        process.set_final_time(200.0)

        # Negative rates should give -inf log-likelihood
        ll = process.log_likelihood(birth_rate=-0.01, death_rate=0.01)
        assert ll == -np.inf

        ll = process.log_likelihood(birth_rate=0.01, death_rate=-0.01)
        assert ll == -np.inf

    def test_very_large_ages(self):
        """Test behavior with very large ages."""
        process = BirthDeathProcess(birth_rate=0.001, death_rate=0.0001)
        process.creation_times = [1.0, 2.0, 3.0]
        process.set_final_time(100000.0)  # Very large final time

        # Should handle large ages without crashing
        ll = process.log_likelihood()
        # Result might be -inf due to numerical limits, but shouldn't crash
        assert np.isfinite(ll) or ll == -np.inf

    def test_single_manuscript(self):
        """Test behavior with single manuscript."""
        process = BirthDeathProcess(birth_rate=0.01, death_rate=0.005)
        process.add_manuscript(100.0)
        process.set_final_time(500.0)

        assert len(process.ages) == 1
        assert process.ages[0] == 400.0

        # Should be able to calculate log-likelihood
        ll = process.log_likelihood()
        assert np.isfinite(ll)

    def test_final_time_before_creation_time(self):
        """Test behavior when final_time is before creation_time."""
        process = BirthDeathProcess()
        process.add_manuscript(1000.0)
        process.set_final_time(500.0)  # Before creation time

        # Age would be negative
        assert process.ages[0] == -500.0

        # This is an invalid scenario, but the code should handle it
        # Log-likelihood calculation might fail or return -inf
        ll = process.log_likelihood()
        # Just check that it doesn't crash
        assert ll == -np.inf or np.isfinite(ll)
