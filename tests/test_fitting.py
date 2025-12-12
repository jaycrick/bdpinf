"""
Test suite for modular fitting functionality.
"""

import numpy as np
import pandas as pd
import pytest

from inf_bp_texts.inference.fitting import fit_birth_death_processes, fit_from_dataframe
from inf_bp_texts.models import BirthDeathProcess


class TestFitting:
    """Test parameter fitting functionality using modular approach."""

    def test_fit_birth_death_process_with_times_arrays(self):
        """Test fitting with times as array."""
        ages = np.array([100.0, 200.0, 300.0, 500.0, 700.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes(
            [ages],
            times,
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_fit_birth_death_process_success(self):
        """Test successful fitting with ages array."""
        # Create some test ages
        ages = np.array([100.0, 200.0, 300.0, 500.0, 700.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes([ages], times)

        # Should return valid result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"]

        if result["success"]:
            assert "birth_rate" in result
            assert "death_rate" in result
            assert "log_likelihood" in result
            assert result["birth_rate"] > 0
            assert result["death_rate"] > 0

    def test_fit_birth_death_process_no_ages(self):
        """Test fitting fails with no ages."""
        ages = np.array([], dtype=np.float64)
        times = 1000.0

        with pytest.raises(ValueError, match="No valid data points for fitting"):
            fit_birth_death_processes([ages], times)

    def test_fit_birth_death_process_custom_initial_params(self):
        """Test fitting with custom initial parameters."""
        ages = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        times = 1000.0
        initial_birth_rates = np.array([0.01])
        initial_death_rates = np.array([0.002])

        result = fit_birth_death_processes(
            [ages],
            times,
            initial_birth_rates=initial_birth_rates,
            initial_death_rates=initial_death_rates,
        )

        # Should succeed regardless of initial parameters
        assert isinstance(result, dict)
        assert result["success"]
        assert "success" in result

    def test_fit_from_dataframe_success(self):
        """Test successful fitting with external DataFrame."""
        # Create test data
        np.random.seed(42)
        ages = np.random.exponential(scale=200, size=50)
        df = pd.DataFrame({"age": ages, "count": np.ones(len(ages))})

        result = fit_from_dataframe(df, age_column="age", times=1000.0)

        # Should return valid result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"]

        if result["success"]:
            assert "birth_rate" in result
            assert "death_rate" in result
            assert result["birth_rate"] > 0
            assert result["death_rate"] > 0

    def test_fit_from_dataframe_individual_records(self):
        """Test fitting with individual manuscript records."""
        # Create individual manuscript records instead of aggregated counts
        ages = []
        for age, count in [(100.0, 5), (200.0, 3), (300.0, 2), (400.0, 1)]:
            ages.extend([age] * count)

        df = pd.DataFrame({"age": ages})

        result = fit_from_dataframe(df, age_column="age", times=1000.0)

        # Should return valid result structure
        assert result["success"]
        assert isinstance(result, dict)
        assert "success" in result

    def test_fit_from_dataframe_no_valid_data_points(self):
        """Test fitting fails with no valid data points."""
        # DataFrame with no data
        df = pd.DataFrame({"age": []})

        with pytest.raises(ValueError, match="No valid data points for fitting"):
            fit_from_dataframe(df, age_column="age", times=1000.0)

    def test_fit_from_dataframe_missing_age_column(self):
        """Test fitting fails when age column is missing."""
        df = pd.DataFrame({"wrong_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'age' not found"):
            fit_from_dataframe(df, age_column="age", times=1000.0)

    def test_fit_birth_death_process_different_methods(self):
        """Test fitting with the default optimization strategy."""
        # Test that the function works with the default optimization strategy
        ages = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes([ages], times)
        # Should return a result
        assert isinstance(result, dict)
        assert "success" in result
        if result["success"]:
            assert result["birth_rate"] > 0
            assert result["death_rate"] > 0

    def test_fit_results_structure(self):
        """Test that fit results have the expected structure."""
        ages = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        times = 1000.0
        result = fit_birth_death_processes([ages], times)

        assert result["success"]

        expected_keys = [
            "success",
            "birth_rate",
            "death_rate",
            "log_likelihood",
            "optimization_result",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check types
        assert isinstance(result["birth_rate"], (float, np.float64))
        assert isinstance(result["death_rate"], (float, np.float64))
        assert isinstance(result["log_likelihood"], (float, np.float64))

    def test_process_integration_with_fitting(self):
        """Test integration between BirthDeathProcess and fitting functions."""
        # Create process
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        # Get ages and fit
        ages = process.get_manuscript_ages()
        result = fit_birth_death_processes([ages], process.final_time)

        assert result["success"]
        # Update process with fitted parameters
        process.update_parameters(result["birth_rate"], result["death_rate"], result)

        # Process should now be fitted
        assert process.fitted
        assert process.fit_results is not None
        assert process.birth_rate == result["birth_rate"]
        assert process.death_rate == result["death_rate"]

    def test_fit_from_dataframe_custom_age_column(self):
        """Test fitting with custom age column name."""
        df = pd.DataFrame(
            {
                "manuscript_age": [100.0, 200.0, 300.0, 400.0],
                "frequency": [1.0, 1.0, 1.0, 1.0],
            }
        )

        result = fit_from_dataframe(df, age_column="manuscript_age", times=1000.0)

        assert isinstance(result, dict)
        assert result["success"]
        assert "success" in result

    def test_process_state_preservation(self):
        """Test that process state is preserved when using external fitting."""
        process = BirthDeathProcess(
            creation_times=[50.0, 150.0, 250.0], final_time=500.0
        )

        original_ages = process.get_manuscript_ages().copy()
        original_creation_times = process.creation_times.copy()
        original_final_time = process.final_time

        # Fit with external data
        df = pd.DataFrame({"age": [100.0, 200.0, 300.0], "count": [1.0, 1.0, 1.0]})
        result = fit_from_dataframe(df, age_column="age", times=1000.0)

        # Original process data should be preserved
        np.testing.assert_array_equal(process.get_manuscript_ages(), original_ages)
        assert process.creation_times == original_creation_times
        assert process.final_time == original_final_time

        # But we can still update it with fit results if desired
        assert result["success"]
        process.update_parameters(result["birth_rate"], result["death_rate"], result)
        assert process.fitted
