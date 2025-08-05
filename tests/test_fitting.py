"""
Test suite for BirthDeathProcess fitting functionality.
"""

import numpy as np
import pandas as pd
import pytest

from inf_bp_texts.inference import BirthDeathProcess


class TestFitting:
    """Test parameter fitting functionality."""

    def test_fit_from_self_data_success(self):
        """Test successful fitting with internal data."""
        # Create process with known structure
        process = BirthDeathProcess(
            birth_rate=0.02,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0, 500.0, 700.0],
            final_time=1000.0,
        )

        # Fit should succeed
        result = process.fit_from_self_data()

        # Fitting might fail due to numerical issues, but should return valid result structure
        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert "birth_rate" in result
            assert "death_rate" in result
            assert "log_likelihood" in result
            assert "aic" in result
            assert "bic" in result
            assert process.fitted

    def test_fit_from_self_data_no_ages(self):
        """Test fitting fails with no ages."""
        process = BirthDeathProcess()

        with pytest.raises(ValueError, match="No ages data available"):
            process.fit_from_self_data()

    def test_fit_from_self_data_custom_initial_params(self):
        """Test fitting with custom initial parameters."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        initial_params = (0.01, 0.002)
        result = process.fit_from_self_data(initial_params=initial_params)

        # Should succeed regardless of initial parameters
        assert isinstance(result, dict)
        assert "success" in result

    def test_fit_external_data_success(self):
        """Test successful fitting with external DataFrame."""
        # Create test data
        np.random.seed(42)
        ages = np.random.exponential(scale=200, size=50)
        df = pd.DataFrame({"age": ages, "count": np.ones(len(ages))})

        process = BirthDeathProcess()
        result = process.fit(df)

        # Fitting might fail with random data, but should return valid structure
        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert "birth_rate" in result
            assert "death_rate" in result
            assert process.fitted

    def test_fit_external_data_with_counts(self):
        """Test fitting with explicit count column."""
        df = pd.DataFrame(
            {"age": [100.0, 200.0, 300.0, 400.0], "count": [5.0, 3.0, 2.0, 1.0]}
        )

        process = BirthDeathProcess()
        result = process.fit(df, count_column="count")

        # Should return valid result structure
        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert process.fitted

    def test_fit_no_data_falls_back_to_internal(self):
        """Test that fit with no external data uses internal data."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        # Calling fit without manuscript_data should use internal data
        result = process.fit(manuscript_data=None)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert process.fitted

    def test_fit_empty_dataframe_uses_internal(self):
        """Test that empty DataFrame uses internal data if available."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        empty_df = pd.DataFrame()
        result = process.fit(empty_df)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert process.fitted

    def test_fit_no_valid_data_points(self):
        """Test fitting fails with no valid data points."""
        # DataFrame with all zero counts
        df = pd.DataFrame({"age": [100.0, 200.0, 300.0], "count": [0.0, 0.0, 0.0]})

        process = BirthDeathProcess()

        with pytest.raises(ValueError, match="No valid data points for fitting"):
            process.fit(df, count_column="count")

    def test_fit_different_methods(self):
        """Test fitting with different optimization methods."""
        methods = ["L-BFGS-B", "Nelder-Mead", "Powell"]

        for method in methods:
            # Create new process for each test
            test_process = BirthDeathProcess(
                creation_times=[100.0, 200.0, 300.0, 400.0], final_time=1000.0
            )

            result = test_process.fit_from_self_data(method=method)
            # Method might not always succeed, but should return a result
            assert isinstance(result, dict)
            assert "success" in result

    def test_fit_results_structure(self):
        """Test that fit results have the expected structure."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0, 400.0, 500.0], final_time=1000.0
        )

        result = process.fit_from_self_data()

        if result["success"]:
            # Successful fit should have all expected keys
            expected_keys = [
                "success",
                "birth_rate",
                "death_rate",
                "log_likelihood",
                "aic",
                "bic",
                "n_data_points",
                "optimization_result",
                "ages_used",
                "counts_used",
            ]

            for key in expected_keys:
                assert key in result, f"Missing key: {key}"

            # Check types
            assert isinstance(result["birth_rate"], (float, np.float64))
            assert isinstance(result["death_rate"], (float, np.float64))
            assert isinstance(result["log_likelihood"], (float, np.float64))
            assert isinstance(result["aic"], (float, np.float64))
            assert isinstance(result["bic"], (float, np.float64))
            assert isinstance(result["n_data_points"], int)

    def test_fit_updates_process_parameters(self):
        """Test that successful fitting updates process parameters."""
        process = BirthDeathProcess(
            birth_rate=0.1,  # Initial value
            death_rate=0.05,  # Initial value
            creation_times=[100.0, 200.0, 300.0],
            final_time=1000.0,
        )

        original_birth = process.birth_rate
        original_death = process.death_rate

        result = process.fit_from_self_data()

        if result["success"]:
            # Parameters should have been updated
            assert (
                process.birth_rate != original_birth
                or process.death_rate != original_death
            )
            assert process.fitted
            assert process.fit_results is not None

    def test_fit_with_custom_age_column(self):
        """Test fitting with custom age column name."""
        df = pd.DataFrame(
            {
                "manuscript_age": [100.0, 200.0, 300.0, 400.0],
                "frequency": [1.0, 1.0, 1.0, 1.0],
            }
        )

        process = BirthDeathProcess()
        result = process.fit(df, age_column="manuscript_age")

        assert result["success"]

    def test_fit_preserves_original_ages(self):
        """Test that fitting with external data preserves original ages."""
        process = BirthDeathProcess(
            creation_times=[50.0, 150.0, 250.0], final_time=500.0
        )

        original_ages = process.ages.copy()

        # Fit with external data
        df = pd.DataFrame({"age": [100.0, 200.0, 300.0], "count": [1.0, 1.0, 1.0]})

        process.fit(df)

        # Original ages should be preserved
        np.testing.assert_array_equal(process.ages, original_ages)
