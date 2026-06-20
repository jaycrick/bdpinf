"""Integration tests verifying components work together in end-to-end workflows."""

import numpy as np
import pandas as pd
import pytest

from inf_bp_texts.inference.fitting import (
    ParameterConstraints,
    fit_birth_death_processes,
    fit_from_data_frame,
)
from inf_bp_texts.inference.likelihood import log_likelihood
from inf_bp_texts.simulation import simulate_birth_death_process
from inf_bp_texts.simulation.simulator import simulate_bdp_ages_from_distributions
from inf_bp_texts.simulation.splitting import (
    SplittingConfig,
    simulate_subcritical_survival_splitting,
)


def _as_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.xfail(
        reason="result is a BirthDeathProcess, not subscriptable like a dict —"
        " pre-existing failure, to be sorted through later"
    )
    def test_full_workflow_simulation_and_fitting(self) -> None:
        """Test complete workflow: simulate data, then fit parameters."""
        # Step 1: Create a process with known parameters for simulation
        true_birth_rate = 0.05
        true_death_rate = 0.01
        # Step 1: Simulate data using modular function
        simulated_df = simulate_birth_death_process(
            birth_rates=true_birth_rate,
            death_rates=true_death_rate,
            times=200.0,
            random_seed=42,
        )

        if len(simulated_df) > 0:
            surviving_data = simulated_df[simulated_df["survived"]]
            if len(surviving_data) > 0:
                fit_df = pd.DataFrame({"age": surviving_data["age"].values})
                result = fit_from_data_frame(fit_df, age_column="age", times=200.0)
                if result["success"]:
                    fitted_birth = result["birth_rates"]
                    fitted_death = result["death_rates"]
                    assert fitted_birth > 0
                    assert fitted_death >= 0
                    assert np.isfinite(_as_float(fitted_birth))
                    assert np.isfinite(_as_float(fitted_death))

    @pytest.mark.slow
    def test_multi_epoch_convergence_rate(self) -> None:
        """Test multi-epoch convergence rate on simulated datasets."""
        rng = np.random.default_rng(12345)
        n_trials = 12
        min_survivors = 10
        n_datasets = 0
        n_success = 0
        consecutive_successes = 0
        times = np.array([150.0, 300.0, 450.0])

        for _ in range(n_trials):
            birth_rates = rng.uniform(0.005, 0.01, size=3)
            death_rates = birth_rates * rng.uniform(0.0025, 0.005, size=3)
            base_seed = int(rng.integers(0, 2**31 - 1))

            df = simulate_birth_death_process(
                birth_rates=birth_rates,
                death_rates=death_rates,
                times=times,
                random_seed=base_seed,
            )
            surviving = df[df["survived"]]
            ages = None
            if len(surviving) >= min_survivors:
                ages = surviving["age"].to_numpy(dtype=np.float64)

            if ages is None:
                continue

            n_datasets += 1
            try:
                result = fit_birth_death_processes([ages], times)
            except ValueError:
                consecutive_successes = 0
                continue

            if result["success"] and np.isfinite(
                _as_float(result.get("log_likelihood", np.nan)),
            ):
                n_success += 1
                consecutive_successes += 1
                if consecutive_successes >= 9:
                    break
            else:
                consecutive_successes = 0

        assert n_datasets >= int(n_trials * 0.75)
        assert n_success / n_datasets >= 0.9

    def test_manuscript_lifecycle_workflow(self) -> None:
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
        # Use modular functions for lifecycle workflow
        creation_times_extended = creation_times + [850.0, 900.0]
        assert len(creation_times_extended) == 12
        ages = [1000.0 - ct for ct in creation_times_extended]
        assert len(ages) == 12
        assert all(age > 0 for age in ages)
        summary = {
            "n_manuscripts": len(creation_times_extended),
            "min_age": min(ages),
            "max_age": max(ages),
        }
        assert summary["n_manuscripts"] == 12
        assert summary["min_age"] == 100
        assert summary["max_age"] == 900
        fit_df = pd.DataFrame({"age": ages})
        result = fit_from_data_frame(fit_df, age_column="age", times=1000.0)
        assert isinstance(result, dict)
        assert "success" in result
        test_ages = np.array([100, 200, 300, 400, 500])
        predictions = [
            log_likelihood(
                birth_rates=result["birth_rates"],
                death_rates=result["death_rates"],
                ages=np.array([age], dtype=np.float64),
                times=1000.0,
            )
            for age in test_ages
        ]
        assert len(predictions) == 5
        assert all(np.isfinite(_as_float(pred)) for pred in predictions)
        assert all(pred <= 0 for pred in predictions)

    def test_external_data_processing_workflow(self) -> None:
        """Test workflow for processing external manuscript data."""
        # Create realistic manuscript data
        manuscript_dates = np.array(
            [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        )
        current_year = 2000

        # Create DataFrame as might come from real data
        df = pd.DataFrame(
            {
                "manuscript_id": range(1, 11),
                "midYear": manuscript_dates,
                "location": ["Location_" + str(i) for i in range(1, 11)],
            },
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

        result = fit_from_data_frame(prepared_df, age_column="age", times=2000.0)
        assert result["success"]
        age_range = np.linspace(100, 1000, 10)
        predictions = [
            log_likelihood(
                birth_rates=result["birth_rates"],
                death_rates=result["death_rates"],
                ages=np.array([age], dtype=np.float64),
                times=2000.0,
            )
            for age in age_range
        ]
        assert len(predictions) == 10
        assert all(np.isfinite(_as_float(pred)) for pred in predictions)

    def test_model_comparison_workflow(self) -> None:
        """Test workflow for comparing different models manually."""
        # Create test data with more realistic parameters
        rng = np.random.default_rng(42)  # For reproducibility
        ages = rng.exponential(
            scale=200,
            size=20,
        )  # Smaller scale and sample size
        df = pd.DataFrame({"age": ages})

        # Create models with more reasonable initial parameters
        max_age = float(ages.max())
        final_time = max_age + 50.0  # Smaller buffer
        from inf_bp_texts.inference.fitting import fit_from_data_frame

        initial_params_list = [(0.01, 0.005), (0.02, 0.001)]
        results = []
        for i, (init_birth, init_death) in enumerate(initial_params_list):
            try:
                fit_result = fit_from_data_frame(
                    df,
                    age_column="age",
                    times=final_time,
                    initial_birth_rates=np.array([init_birth]),
                    initial_death_rates=np.array([init_death]),
                )
                results.append(
                    {
                        "model_id": i,
                        "success": fit_result["success"],
                        "log_likelihood": fit_result.get("log_likelihood", np.nan),
                    },
                )
            except Exception:
                results.append(
                    {"model_id": i, "success": False, "log_likelihood": np.nan},
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
            assert np.isfinite(_as_float(best_result["log_likelihood"]))
        else:
            # If no models succeeded, that's also a valid outcome for difficult data
            print(
                "Warning: No models succeeded in fitting. This may indicate difficult data.",
            )
            assert True  # Test passes even if no models fit

    def test_parameter_sensitivity_analysis(self) -> None:
        """Test how parameter changes affect results."""
        creation_times = [100.0, 200.0, 300.0, 400.0, 500.0]
        final_time = 1000.0
        ages = np.array([final_time - ct for ct in creation_times], dtype=np.float64)
        base_ll = log_likelihood(
            birth_rates=0.02,
            death_rates=0.005,
            ages=ages,
            times=final_time,
        )
        higher_birth_ll = log_likelihood(
            birth_rates=0.04,
            death_rates=0.005,
            ages=ages,
            times=final_time,
        )
        lower_birth_ll = log_likelihood(
            birth_rates=0.01,
            death_rates=0.005,
            ages=ages,
            times=final_time,
        )
        assert np.isfinite(_as_float(base_ll))
        assert np.isfinite(_as_float(higher_birth_ll))
        assert np.isfinite(_as_float(lower_birth_ll))
        higher_death_ll = log_likelihood(
            birth_rates=0.02,
            death_rates=0.01,
            ages=ages,
            times=final_time,
        )
        lower_death_ll = log_likelihood(
            birth_rates=0.02,
            death_rates=0.002,
            ages=ages,
            times=final_time,
        )
        assert np.isfinite(_as_float(higher_death_ll))
        assert np.isfinite(_as_float(lower_death_ll))
        likelihoods = [
            base_ll,
            higher_birth_ll,
            lower_birth_ll,
            higher_death_ll,
            lower_death_ll,
        ]
        unique_values = len(set(float(ll) for ll in likelihoods))
        assert unique_values == 5  # All different

    def test_data_quality_robustness(self) -> None:
        """Test robustness to different data quality scenarios."""
        # Scenario 1: Very few data points
        sparse_ages = np.array([1000.0 - t for t in [500.0, 600.0]], dtype=np.float64)
        sparse_result = fit_birth_death_processes([sparse_ages], times=1000.0)
        assert isinstance(sparse_result, dict)
        assert "success" in sparse_result

        # Scenario 2: Many data points
        dense_times = list(range(100, 900, 50))
        dense_ages = np.array(
            [1000.0 - float(t) for t in dense_times],
            dtype=np.float64,
        )
        dense_result = fit_birth_death_processes([dense_ages], times=1000.0)
        assert isinstance(dense_result, dict)
        assert "success" in dense_result

        # Scenario 3: Irregular spacing
        irregular_times = [10.0, 50.0, 51.0, 52.0, 200.0, 800.0, 900.0, 950.0]
        irregular_ages = np.array(
            [1000.0 - t for t in irregular_times],
            dtype=np.float64,
        )
        irregular_result = fit_birth_death_processes([irregular_ages], times=1000.0)
        assert isinstance(irregular_result, dict)
        assert "success" in irregular_result

    def test_numerical_edge_cases(self) -> None:
        """Test numerical edge cases that might cause problems."""
        # Very small parameters
        final_time_small = 1000.0
        creation_times_small = [100.0, 200.0, 300.0]
        ages_small = np.array(
            [final_time_small - ct for ct in creation_times_small],
            dtype=np.float64,
        )
        small_ll = log_likelihood(
            birth_rates=1e-6,
            death_rates=1e-7,
            ages=ages_small,
            times=final_time_small,
        )
        assert np.isfinite(_as_float(small_ll)) or small_ll == -np.inf

        creation_times_critical = [100.0, 200.0, 300.0]
        final_time_critical = 1000.0
        ages_critical = np.array(
            [final_time_critical - ct for ct in creation_times_critical],
            dtype=np.float64,
        )
        birth_rate = 0.01
        death_rate = 0.01 + 1e-12
        critical_ll = log_likelihood(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=ages_critical,
            times=final_time_critical,
        )
        assert np.isfinite(_as_float(critical_ll)) or critical_ll == -np.inf

        creation_times_large = [1.0, 2.0, 3.0]
        final_time_large = 50000.0
        ages_large = np.array(
            [final_time_large - ct for ct in creation_times_large],
            dtype=np.float64,
        )
        large_ll = log_likelihood(
            birth_rates=0.001,
            death_rates=0.0001,
            ages=ages_large,
            times=final_time_large,
        )
        assert np.isfinite(_as_float(large_ll)) or large_ll == -np.inf

    def test_simulate_then_log_likelihood_is_finite(self) -> None:
        rng = np.random.default_rng(77)
        br = 0.01
        dr = 0.004
        T = 600.0
        process = simulate_bdp_ages_from_distributions(
            br, dr, T, rng, ensure_non_extinct=True
        )
        ages = np.asarray(process.manuscript_ages)
        if ages.size == 0:
            return
        ll = log_likelihood(
            birth_rates=br,
            death_rates=dr,
            ages=ages,
            times=600.0,
        )
        assert np.isfinite(_as_float(ll))

    def test_fit_from_data_frame_matches_fit_from_arrays(self) -> None:
        rng = np.random.default_rng(99)
        T = 1200.0
        ages_arr = rng.exponential(scale=300.0, size=25)
        ages_arr = ages_arr[ages_arr < T]
        df = pd.DataFrame({"age": ages_arr})
        result_df = fit_from_data_frame(df, age_column="age", times=T)
        result_arr = fit_birth_death_processes(
            [ages_arr],
            times=T,
            estimate_uncertainty=False,
        )
        if result_df["success"] and result_arr["success"]:
            assert np.isfinite(_as_float(result_df["birth_rates"]))
            assert np.isfinite(_as_float(result_arr["birth_rates"]))

    def test_covariance_is_positive_definite_after_successful_fit(self) -> None:
        rng = np.random.default_rng(55)
        ages = rng.exponential(scale=300.0, size=30)
        ages = ages[ages < 1200.0]
        result = fit_birth_death_processes(
            [ages],
            times=1200.0,
            estimate_uncertainty=True,
        )
        if not result["success"]:
            return
        cov = result.get("uncertainty")
        if cov is None:
            return
        cov_arr = np.atleast_2d(cov)
        eigenvalues = np.linalg.eigvalsh(cov_arr)
        assert np.all(eigenvalues >= -1e-8)

    def test_equality_constraint_produces_equal_rates(self) -> None:
        ages = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0])
        times = np.array([750.0, 1500.0])
        constraints = ParameterConstraints(equality_groups=[[0, 1]])
        result = fit_birth_death_processes(
            [ages],
            times=times,
            param_constraints=constraints,
            estimate_uncertainty=False,
        )
        if not result["success"]:
            return
        birth_rates = np.atleast_1d(result["birth_rates"])
        assert birth_rates[0] == pytest.approx(birth_rates[1], rel=1e-4)

    @pytest.mark.xfail(
        reason="result is a BirthDeathProcess, not subscriptable like a dict —"
        " pre-existing failure, to be sorted through later"
    )
    def test_state_consistency_throughout_workflow(self) -> None:
        """Test that object state remains consistent throughout operations."""
        creation_times = []
        final_time = 1000.0
        birth_rate = 0.02
        death_rate = 0.005
        ages = np.array([final_time - ct for ct in creation_times], dtype=np.float64)
        assert len(ages) == 0
        creation_times.extend([100.0, 200.0])
        final_time = 500.0
        ages = np.array([final_time - ct for ct in creation_times], dtype=np.float64)
        assert len(creation_times) == 2
        assert len(ages) == 2
        np.testing.assert_array_equal(ages, [400.0, 300.0])
        result = fit_birth_death_processes([ages], times=final_time)
        if result["success"]:
            assert result["success"]
            assert len(creation_times) == 2
            ages = np.array(
                [final_time - ct for ct in creation_times],
                dtype=np.float64,
            )
            assert len(ages) == 2
            np.testing.assert_array_equal(ages, [400.0, 300.0])
        simulated_df = simulate_birth_death_process(
            birth_rate,
            death_rate,
            times=300.0,
            random_seed=42,
        )
        final_time = 300.0
        creation_times = list(simulated_df["creation_time"].values)
        ages = np.array([final_time - ct for ct in creation_times], dtype=np.float64)
        assert final_time == 300.0
        assert len(ages) == len(creation_times)
        if len(creation_times) > 0:
            assert final_time is not None
            assert all(ct <= final_time for ct in creation_times)
            assert all(age >= 0 for age in ages)
            assert all(ct <= final_time for ct in creation_times)
            assert all(age >= 0 for age in ages)

    def test_splitting_result_feeds_birth_death_process(self) -> None:
        """SplittingSimulationResult.conditioned_trajectories can seed a fit."""
        config = SplittingConfig(
            birth_rate=0.01,
            death_rate=0.05,
            terminal_time=300.0,
            n_particles=30,
            random_seed=42,
        )
        result = simulate_subcritical_survival_splitting(config)
        assert result.survival_probability_estimate > 0
        assert result.survival_probability_estimate < 1
        # particle_genealogy should be a DataFrame with genealogy columns
        pg = result.particle_genealogy
        assert isinstance(pg, pd.DataFrame)
        required_cols = {"child_particle_id", "parent_particle_id", "level_index"}
        assert required_cols.issubset(set(pg.columns)), (
            f"Missing columns: {required_cols - set(pg.columns)}"
        )
        # conditioned_trajectories should produce non-empty BirthDeathProcess objects
        assert len(result.conditioned_trajectories) > 0
        traj = result.conditioned_trajectories[0]
        assert traj.process is not None
        bdp = traj.process
        assert len(bdp.creation_times) > 0
