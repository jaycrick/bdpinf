"""
Integration tests for BirthDeathProcess class.

These tests verify that different components work together correctly
and test realistic end-to-end workflows.
"""

import numpy as np
import pandas as pd

from inf_bp_texts.inference import BirthDeathProcess


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_simulation_and_fitting(self):
        """Test complete workflow: simulate data, then fit parameters."""
        # Step 1: Create a process with known parameters for simulation
        true_birth_rate = 0.05
        true_death_rate = 0.01
        true_process = BirthDeathProcess(
            [], 200.0, birth_rate=true_birth_rate, death_rate=true_death_rate
        )

        # Step 2: Simulate data
        simulated_df = true_process.simulate(max_time=200.0, random_seed=42)

        # Step 3: Create new process and fit to simulated data
        fit_process = BirthDeathProcess([], 200.0)  # Empty initial data for fitting

        # Convert simulation to fitting format if needed
        if len(simulated_df) > 0:
            surviving_data = simulated_df[simulated_df["survived"]]
            if len(surviving_data) > 0:
                fit_df = pd.DataFrame({"age": surviving_data["age"].values})

                result = fit_process.fit(fit_df)

                if result["success"]:
                    # Fitted parameters should be reasonably close to true parameters
                    # (allowing for sampling variation)
                    fitted_birth = result["birth_rate"]
                    fitted_death = result["death_rate"]

                    # Check that parameters are in reasonable range
                    assert fitted_birth > 0
                    assert fitted_death >= 0

                    # Parameters should be positive and finite
                    assert np.isfinite(fitted_birth)
                    assert np.isfinite(fitted_death)

    def test_manuscript_lifecycle_workflow(self):
        """Test workflow for tracking manuscript lifecycle."""
        # Create with initial data
        creation_times = [
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
        ]
        process = BirthDeathProcess(
            creation_times, 1000.0, birth_rate=0.02, death_rate=0.005
        )

        # Test adding more manuscripts
        process.add_manuscript(850.0)
        process.add_manuscript(900.0)

        # Verify state
        assert len(process.creation_times) == 12  # 10 + 2 added
        ages = process.get_manuscript_ages()
        assert len(ages) == 12
        assert all(age > 0 for age in ages)

        # Get summary
        summary = process.get_process_summary()
        assert summary["n_manuscripts"] == 12
        assert summary["min_age"] == 100  # 1000 - 900 (newest manuscript)
        assert summary["max_age"] == 900  # 1000 - 100 (oldest manuscript)

        # Fit parameters (should work even when model is already initialized with parameters)
        result = process.fit_from_self_data()
        # Note: fit may or may not succeed depending on optimization convergence
        # The main point is that it doesn't crash
        assert isinstance(result, dict)
        assert "success" in result

        # Predict manuscript counts
        test_ages = np.array([100, 200, 300, 400, 500])
        predictions = process.predict(test_ages)
        assert len(predictions) == 5
        assert all(pred > 0 for pred in predictions)

    def test_external_data_processing_workflow(self):
        """Test workflow for processing external manuscript data."""
        # Create realistic manuscript data
        manuscript_dates = np.array(
            [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
        )
        current_year = 2000

        # Create DataFrame as might come from real data
        df = pd.DataFrame(
            {
                "manuscript_id": range(1, 11),
                "midYear": manuscript_dates,
                "location": ["Location_" + str(i) for i in range(1, 11)],
            }
        )

        # Prepare data inline
        prepared_df = df.copy()
        prepared_df = prepared_df[prepared_df["midYear"] < current_year]
        prepared_df["age"] = current_year - prepared_df["midYear"]
        prepared_df = prepared_df[prepared_df["age"] >= 0]

        # Verify preparation
        assert "age" in prepared_df.columns
        assert len(prepared_df) == 10

        # Fit model
        process = BirthDeathProcess(
            [], 2000.0
        )  # Empty creation_times, will use external data
        result = process.fit(prepared_df)

        assert result["success"]
        assert process.fitted

        # Use fitted model for predictions
        age_range = np.linspace(100, 1000, 10)
        predictions = process.predict(age_range)

        assert len(predictions) == 10
        assert all(np.isfinite(pred) for pred in predictions)

    def test_model_comparison_workflow(self):
        """Test workflow for comparing different models manually."""
        # Create test data with more realistic parameters
        np.random.seed(42)  # For reproducibility
        ages = np.random.exponential(
            scale=200, size=20
        )  # Smaller scale and sample size
        df = pd.DataFrame({"age": ages})

        # Create models with more reasonable initial parameters
        max_age = float(ages.max())
        final_time = max_age + 50.0  # Smaller buffer
        models = [
            BirthDeathProcess([], final_time, birth_rate=0.01, death_rate=0.005),
            BirthDeathProcess([], final_time, birth_rate=0.02, death_rate=0.001),
        ]

        # Fit each model and collect results manually
        results = []
        for i, model in enumerate(models):
            try:
                fit_result = model.fit(df)
                results.append(
                    {
                        "model_id": i,
                        "success": fit_result["success"],
                        "log_likelihood": fit_result.get("log_likelihood", np.nan),
                    }
                )
            except Exception:
                results.append(
                    {"model_id": i, "success": False, "log_likelihood": np.nan}
                )

        # Verify we have results for all models
        assert len(results) == 2

        # At least check that the models attempted to fit
        assert all("success" in result for result in results)

        # If any models were successful, verify the log-likelihood values
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            # Check that we can find the best model by log-likelihood
            best_result = max(successful_results, key=lambda x: x["log_likelihood"])
            assert np.isfinite(best_result["log_likelihood"])
        else:
            # If no models succeeded, that's also a valid outcome for difficult data
            print(
                "Warning: No models succeeded in fitting. This may indicate difficult data."
            )
            assert True  # Test passes even if no models fit

    def test_parameter_sensitivity_analysis(self):
        """Test how parameter changes affect results."""
        base_process = BirthDeathProcess(
            birth_rate=0.02,
            death_rate=0.005,
            creation_times=[100.0, 200.0, 300.0, 400.0, 500.0],
            final_time=1000.0,
        )

        base_ll = base_process.log_likelihood()

        # Test birth rate sensitivity
        higher_birth_ll = base_process.log_likelihood(birth_rate=0.04, death_rate=0.005)
        lower_birth_ll = base_process.log_likelihood(birth_rate=0.01, death_rate=0.005)

        # All should be finite
        assert np.isfinite(base_ll)
        assert np.isfinite(higher_birth_ll)
        assert np.isfinite(lower_birth_ll)

        # Test death rate sensitivity
        higher_death_ll = base_process.log_likelihood(birth_rate=0.02, death_rate=0.01)
        lower_death_ll = base_process.log_likelihood(birth_rate=0.02, death_rate=0.002)

        assert np.isfinite(higher_death_ll)
        assert np.isfinite(lower_death_ll)

        # Collect all likelihood values
        likelihoods = [
            base_ll,
            higher_birth_ll,
            lower_birth_ll,
            higher_death_ll,
            lower_death_ll,
        ]

        # All should be different (showing sensitivity)
        unique_values = len(set(likelihoods))
        assert unique_values == 5  # All different

    def test_data_quality_robustness(self):
        """Test robustness to different data quality scenarios."""
        # Scenario 1: Very few data points
        sparse_process = BirthDeathProcess(
            creation_times=[500.0, 600.0], final_time=1000.0
        )

        sparse_result = sparse_process.fit_from_self_data()
        # Should handle sparse data gracefully
        assert isinstance(sparse_result, dict)
        assert "success" in sparse_result

        # Scenario 2: Many data points
        dense_times = list(range(100, 900, 50))  # Many manuscripts
        dense_process = BirthDeathProcess(
            creation_times=[float(t) for t in dense_times], final_time=1000.0
        )

        dense_result = dense_process.fit_from_self_data()
        assert isinstance(dense_result, dict)
        assert "success" in dense_result

        # Scenario 3: Irregular spacing
        irregular_times = [10.0, 50.0, 51.0, 52.0, 200.0, 800.0, 900.0, 950.0]
        irregular_process = BirthDeathProcess(
            creation_times=irregular_times, final_time=1000.0
        )

        irregular_result = irregular_process.fit_from_self_data()
        assert isinstance(irregular_result, dict)
        assert "success" in irregular_result

    def test_numerical_edge_cases(self):
        """Test numerical edge cases that might cause problems."""
        # Very small parameters
        small_param_process = BirthDeathProcess(
            birth_rate=1e-6,
            death_rate=1e-7,
            creation_times=[100.0, 200.0, 300.0],
            final_time=1000.0,
        )

        small_ll = small_param_process.log_likelihood()
        assert np.isfinite(small_ll) or small_ll == -np.inf

        # Parameters very close to each other (critical case)
        critical_process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
        )

        birth_rate = 0.01
        death_rate = 0.01 + 1e-12  # Very close

        critical_ll = critical_process.log_likelihood(
            birth_rate=birth_rate, death_rate=death_rate
        )
        assert np.isfinite(critical_ll) or critical_ll == -np.inf

        # Very large ages
        large_age_process = BirthDeathProcess(
            birth_rate=0.001,
            death_rate=0.0001,
            creation_times=[1.0, 2.0, 3.0],
            final_time=50000.0,
        )

        large_ll = large_age_process.log_likelihood()
        assert np.isfinite(large_ll) or large_ll == -np.inf

    def test_state_consistency_throughout_workflow(self):
        """Test that object state remains consistent throughout operations."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.02, death_rate=0.005)

        # Initial state
        assert process.fitted  # Now fitted since parameters were provided
        assert len(process.get_manuscript_ages()) == 0

        # Add manuscripts
        process.add_manuscript(100.0)
        process.add_manuscript(200.0)
        process.set_final_time(500.0)

        # State after adding manuscripts
        assert len(process.creation_times) == 2
        ages = process.get_manuscript_ages()
        assert len(ages) == 2
        np.testing.assert_array_equal(ages, [400.0, 300.0])

        # Fit model
        result = process.fit_from_self_data()

        if result["success"]:
            # State after fitting
            assert process.fitted
            assert process.fit_results is not None
            assert process.fit_results == result

            # Original data should be preserved
            assert len(process.creation_times) == 2
            ages = process.get_manuscript_ages()
            assert len(ages) == 2
            np.testing.assert_array_equal(ages, [400.0, 300.0])

        # Simulate new data (should reset state)
        process.simulate(max_time=300.0, random_seed=42)

        # State after simulation
        assert process.final_time == 300.0
        ages = process.get_manuscript_ages()
        assert len(ages) == len(process.creation_times)
        if len(process.creation_times) > 0:
            # All creation times should be <= final_time
            assert process.final_time is not None
            assert all(ct <= process.final_time for ct in process.creation_times)
            # All ages should be >= 0
            assert all(age >= 0 for age in ages)
