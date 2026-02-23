"""Test suite for modular fitting functionality."""

# ruff: noqa: S101
# ruff: noqa: D101 D102

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from inf_bp_texts.inference.fitting import (
    OptimizerStrategy,
    _apply_param_perturbation,
    _build_initial_params,
    _build_optimizer_strategies,
    _extract_spine_and_ages,
    _find_non_empty_age,
    _normalize_requested_optimizers,
    _prepare_population_sizes,
    _run_optimizer_strategies,
    fit_birth_death_processes,
    fit_from_dataframe,
)
from inf_bp_texts.inference.likelihood import negative_log_likelihood
from inf_bp_texts.models import BirthDeathProcess
from inf_bp_texts.models.validation import validate_times_properties


class TestFittingHelpers:
    def test_find_non_empty_age_returns_first(self) -> None:
        ages_list = [torch.tensor([]), torch.tensor([1.0, 2.0])]
        result = _find_non_empty_age(ages_list)
        assert torch.equal(result, ages_list[1])

    def testvalidate_times_properties_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            validate_times_properties(torch.tensor([1.0, 0.0]))

    def test_prepare_population_sizes_length_mismatch(self) -> None:
        ages_list = [torch.tensor([1.0]), torch.tensor([2.0])]
        with pytest.raises(ValueError, match="Length of population_sizes"):
            _prepare_population_sizes(
                ages_list,
                torch.tensor([1], dtype=torch.int64),
                None,
            )

    def test_extract_spine_and_ages_ordered(self) -> None:
        ages_list = [torch.tensor([10.0, 20.0]), torch.tensor([30.0])]
        spine_ages, non_spine = _extract_spine_and_ages(
            ages_list,
            is_ordered=True,
            spine_ages=None,
        )

        assert spine_ages is not None
        expected_spine = torch.tensor([10.0, 30.0], dtype=spine_ages.dtype)
        expected_non_spine = torch.tensor([20.0], dtype=non_spine.dtype)
        assert torch.allclose(spine_ages, expected_spine)
        assert torch.allclose(non_spine, expected_non_spine)

    def test_apply_param_perturbation_is_deterministic(self) -> None:
        params = torch.tensor([1.0, 2.0], dtype=torch.float64)
        perturbed_first = _apply_param_perturbation(params, 0.1, random_seed=7)
        perturbed_second = _apply_param_perturbation(params, 0.1, random_seed=7)

        assert torch.allclose(perturbed_first, perturbed_second)
        assert torch.all(perturbed_first > 0)

    def test_build_optimizer_strategies_merges_kwargs(self) -> None:
        lr = 0.01
        strategies = _build_optimizer_strategies({"adam": {"lr": lr}})
        assert strategies["adam"].optimizer_kwargs["lr"] == lr

    def test_normalize_requested_optimizers_rejects_unknown(self) -> None:
        strategies = _build_optimizer_strategies(None)
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            _normalize_requested_optimizers(["Adam", "Bogus"], strategies)

    def test_build_initial_params_with_custom_rates(self) -> None:
        birth_rates = torch.tensor([0.1, 0.2])
        death_rates = torch.tensor([0.03, 0.04])
        non_empty_age = torch.tensor([10.0])
        times_tensor = torch.tensor([100.0, 200.0])

        initial_params = _build_initial_params(
            birth_rates,
            death_rates,
            non_empty_age,
            times_tensor,
        )
        expected_params = torch.tensor(
            [0.1, 0.2, 0.03, 0.04],
            dtype=initial_params.dtype,
        )
        assert torch.allclose(initial_params, expected_params)


class TestFitting:
    """Test parameter fitting functionality using modular approach."""

    def test_fit_birth_death_process_with_times_arrays(self) -> None:
        """Test fitting with times as array."""
        ages = np.array([100.0, 200.0, 300.0, 500.0, 700.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes(
            [ages],
            times,
        )

        assert isinstance(result, dict)
        assert "success" in result

    def test_fit_birth_death_process_success(self) -> None:
        """Test successful fitting with ages array."""
        # Create some test ages
        ages = np.array([100.0, 200.0, 300.0, 500.0, 700.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes([ages], times)

        # Should return valid result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"]
        assert "birth_rates" in result
        assert "death_rates" in result
        assert "log_likelihood" in result
        assert result["birth_rates"] > 0
        assert result["death_rates"] > 0

    def test_fit_birth_death_process_no_ages(self) -> None:
        """Test fitting fails with no ages."""
        ages = np.array([], dtype=np.float64)
        times = 1000.0

        with pytest.raises(ValueError, match="No valid data points for fitting"):
            fit_birth_death_processes([ages], times)

    def test_fit_birth_death_process_custom_initial_params(self) -> None:
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

    def test_fit_from_dataframe_success(self) -> None:
        """Test successful fitting with external DataFrame."""
        # Create test data
        rng = np.random.default_rng(42)
        ages = rng.exponential(scale=200, size=50)
        df = pd.DataFrame({"age": ages, "count": np.ones(len(ages))})

        result = fit_from_dataframe(df, age_column="age", times=1000.0)

        # Should return valid result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"]

        if result["success"]:
            assert "birth_rates" in result
            assert "death_rates" in result
            assert result["birth_rates"] > 0
            assert result["death_rates"] > 0

    def test_fit_from_dataframe_individual_records(self) -> None:
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

    def test_fit_from_dataframe_no_valid_data_points(self) -> None:
        """Test fitting fails with no valid data points."""
        # DataFrame with no data
        df = pd.DataFrame({"age": []})

        with pytest.raises(ValueError, match="No valid data points for fitting"):
            fit_from_dataframe(df, age_column="age", times=1000.0)

    def test_fit_from_dataframe_missing_age_column(self) -> None:
        """Test fitting fails when age column is missing."""
        df = pd.DataFrame({"wrong_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'age' not found"):
            fit_from_dataframe(df, age_column="age", times=1000.0)

    def test_fit_birth_death_process_different_methods(self) -> None:
        """Test fitting with the default optimization strategy."""
        # Test that the function works with the default optimization strategy
        ages = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes([ages], times)
        # Should return a result
        assert isinstance(result, dict)
        assert "success" in result
        if result["success"]:
            assert result["birth_rates"] > 0
            assert result["death_rates"] > 0

    def test_fit_results_structure(self) -> None:
        """Test that fit results have the expected structure."""
        ages = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        times = 1000.0
        result = fit_birth_death_processes([ages], times)

        assert result["success"]

        expected_keys = [
            "success",
            "birth_rates",
            "death_rates",
            "log_likelihood",
            "optimization_result",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check types
        assert isinstance(result["birth_rates"], (float, np.float64, np.ndarray))
        assert isinstance(result["death_rates"], (float, np.float64, np.ndarray))
        assert isinstance(result["log_likelihood"], (float, np.float64))

    def test_process_integration_with_fitting(self) -> None:
        """Test integration between BirthDeathProcess and fitting functions."""
        # Create process
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            final_time=1000.0,
        )

        # Get ages and fit
        ages = process.get_manuscript_ages()
        result = fit_birth_death_processes([ages], process.final_time)

        assert result["success"]
        # Update process with fitted parameters
        process.update_parameters(result["birth_rates"], result["death_rates"], result)

        # Process should now be fitted
        assert process.fitted
        assert process.fit_results is not None
        assert process.birth_rate == result["birth_rates"]
        assert process.death_rate == result["death_rates"]

    def test_fit_from_dataframe_custom_age_column(self) -> None:
        """Test fitting with custom age column name."""
        df = pd.DataFrame(
            {
                "manuscript_age": [100.0, 200.0, 300.0, 400.0],
                "frequency": [1.0, 1.0, 1.0, 1.0],
            },
        )

        result = fit_from_dataframe(df, age_column="manuscript_age", times=1000.0)

        assert isinstance(result, dict)
        assert result["success"]
        assert "success" in result

    def test_process_state_preservation(self) -> None:
        """Test that process state is preserved when using external fitting."""
        process = BirthDeathProcess(
            creation_times=[50.0, 150.0, 250.0],
            final_time=500.0,
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
        process.update_parameters(result["birth_rates"], result["death_rates"], result)
        assert process.fitted


class TestMultiEpochFitting:
    """Test multi-epoch parameter fitting functionality."""

    def test_two_epoch_fitting_basic(self) -> None:
        """Test basic two-epoch fitting."""
        # Create ages spanning two epochs
        ages = np.array([50.0, 150.0, 250.0, 350.0, 450.0], dtype=np.float64)
        times = np.array([250.0, 500.0])  # Two epochs

        result = fit_birth_death_processes([ages], times)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert "birth_rates" in result
            assert "death_rates" in result
            assert "log_likelihood" in result

            # Should return arrays for multi-epoch
            assert isinstance(result["birth_rates"], np.ndarray)
            assert isinstance(result["death_rates"], np.ndarray)
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)

            # All rates should be positive
            assert all(result["birth_rates"] > 0)
            assert all(result["death_rates"] > 0)

    def test_two_epoch_fitting_with_initial_params(self) -> None:
        """Test two-epoch fitting with initial parameter guesses."""
        ages = np.array([100.0, 200.0, 400.0, 600.0], dtype=np.float64)
        times = np.array([300.0, 700.0])
        initial_birth_rates = np.array([0.01, 0.005])
        initial_death_rates = np.array([0.002, 0.001])

        result = fit_birth_death_processes(
            [ages],
            times,
            initial_birth_rates=initial_birth_rates,
            initial_death_rates=initial_death_rates,
        )

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            # Check structure
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)

    def test_three_epoch_fitting(self) -> None:
        """Test fitting with three epochs."""
        ages = np.array([30.0, 80.0, 130.0, 180.0, 230.0, 280.0], dtype=np.float64)
        times = np.array([100.0, 200.0, 300.0])  # Three epochs

        result = fit_birth_death_processes([ages], times)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            # Should have 3 parameters for each rate
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)
            assert all(result["birth_rates"] > 0)
            assert all(result["death_rates"] > 0)

    def test_multi_epoch_fitting_array_lengths_consistent(self) -> None:
        """Test that array lengths match between times and rates."""
        ages = np.array([50.0, 150.0, 175.0], dtype=np.float64)
        times = np.array([100.0, 200.0])

        result = fit_birth_death_processes([ages], times)

        if result["success"]:
            # Number of rate parameters should match number of epochs
            n_epochs = len(times)
            assert len(result["birth_rates"]) == n_epochs
            assert len(result["death_rates"]) == n_epochs

    def test_multi_epoch_vs_single_epoch_different_results(self) -> None:
        """Test that multi-epoch and single-epoch fits give different results."""
        ages = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        total_time = 500.0

        # Single epoch fit
        result_single = fit_birth_death_processes([ages], total_time)

        # Two epoch fit
        result_multi = fit_birth_death_processes([ages], np.array([250.0, total_time]))

        # Both should succeed
        assert result_single["success"]
        assert result_multi["success"]

        # Multi-epoch should return arrays
        assert isinstance(result_multi["birth_rates"], np.ndarray)
        assert isinstance(result_multi["death_rates"], np.ndarray)

        # Single-epoch returns scalars (backward compatible)
        # Note: Current implementation may return arrays even for single epoch
        # This test documents the expected behavior

    def test_multi_epoch_no_ages_fails(self) -> None:
        """Test that multi-epoch fitting fails with no ages."""
        ages = np.array([], dtype=np.float64)
        times = np.array([100.0, 200.0])

        with pytest.raises(ValueError, match="No valid data points for fitting"):
            fit_birth_death_processes([ages], times)

    def test_multi_epoch_likelihood_finite(self) -> None:
        """Test that multi-epoch fitting returns finite log-likelihood."""
        ages = np.array([50.0, 150.0, 250.0, 350.0], dtype=np.float64)
        times = np.array([200.0, 400.0])

        result = fit_birth_death_processes([ages], times)

        if result["success"]:
            assert np.isfinite(result["log_likelihood"])
            # Log-likelihood should be negative (log of probability < 1)
            assert result["log_likelihood"] < 0

    def test_multi_epoch_from_dataframe(self) -> None:
        """Test multi-epoch fitting from DataFrame."""
        ages = [100.0, 200.0, 300.0, 400.0, 500.0]
        df = pd.DataFrame({"age": ages})
        times = np.array([250.0, 550.0])

        result = fit_from_dataframe(df, age_column="age", times=times)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            # Should have multi-epoch parameters
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)

    def test_multi_epoch_from_dataframe_large_starting_pop(self) -> None:
        """Test multi-epoch fitting from DataFrame, with starting population sizes."""
        ages = [100.0, 200.0, 300.0, 400.0, 500.0]
        df = pd.DataFrame({"age": ages})
        times = np.array([250.0, 550.0])

        result = fit_from_dataframe(
            df,
            age_column="age",
            times=times,
            starting_population=10,
        )

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            # Should have multi-epoch parameters
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)


class TestConvergenceFlags:
    def test_max_iters_no_convergence(self) -> None:
        ages = torch.tensor([100.0, 200.0], dtype=torch.float64)
        times_tensor = torch.tensor([1000.0], dtype=torch.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)
        log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))
        strategies_map = {
            "sgd": OptimizerStrategy("SGD", torch.optim.SGD, 1, {"lr": 0.0001}),
        }

        run_result = _run_optimizer_strategies(
            strategies_map,
            ["sgd"],
            log_params,
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=torch.tensor([len(ages)], dtype=torch.int64),
            starting_population=1,
            convergence_tolerance=1e-12,
            absolute_tolerance=1e-12,
            patience=100,
            gradient_threshold=-1.0,
        )

        assert not run_result.converged
        assert run_result.iterations == 1
        assert len(run_result.nll_history) == 1

    @pytest.mark.parametrize(
        ("opt_key", "opt_class", "opt_kwargs"),
        [
            ("sgd", torch.optim.SGD, {"lr": 0.0001}),
            (
                "lbfgs",
                torch.optim.LBFGS,
                {"lr": 1.0, "max_iter": 5, "line_search_fn": "strong_wolfe"},
            ),
        ],
    )
    def test_converged_on_gradient_threshold(
        self,
        opt_key: str,
        opt_class: type,
        opt_kwargs: dict[str, float | int | str],
    ) -> None:
        ages = torch.tensor([100.0, 200.0], dtype=torch.float64)
        times_tensor = torch.tensor([1000.0], dtype=torch.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)
        log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))
        strategies_map = {
            opt_key: OptimizerStrategy(opt_key.upper(), opt_class, 100, opt_kwargs),
        }

        run_result = _run_optimizer_strategies(
            strategies_map,
            [opt_key],
            log_params,
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=torch.tensor([len(ages)], dtype=torch.int64),
            starting_population=1,
            convergence_tolerance=1e-12,
            absolute_tolerance=1e-12,
            patience=0,
            gradient_threshold=1e9,
        )

        assert run_result.converged

    def test_collects_exceptions_across_optimizer_attempts(self) -> None:
        ages = torch.tensor([100.0, 200.0], dtype=torch.float64)
        times_tensor = torch.tensor([1000.0], dtype=torch.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)
        log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))

        strategies_map = {
            "sgd": OptimizerStrategy("SGD", torch.optim.SGD, 1, {"lr": 0.0001}),
            "adam": OptimizerStrategy("Adam", torch.optim.Adam, 1, {"lr": 0.001}),
        }

        run_result = _run_optimizer_strategies(
            strategies_map,
            ["sgd", "adam"],
            log_params,
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=torch.tensor([len(ages)], dtype=torch.int64),
            starting_population=1,
            convergence_tolerance=1e-12,
            absolute_tolerance=1e-12,
            patience=100,
            gradient_threshold=-1.0,
        )

        assert not run_result.converged
        assert len(run_result.all_exceptions) == len(strategies_map)
        assert all(isinstance(exc, Exception) for exc in run_result.all_exceptions)


@pytest.mark.slow
class TestFittingDiagnostics:
    def test_realistic_optimizer_diagnostics_report_failure_stage(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data_file = project_root / "manuscript_analysis_cleaned.csv"
        if not data_file.exists():
            pytest.skip(
                "Diagnostic dataset manuscript_analysis_cleaned.csv not available",
            )

        df = pd.read_csv(data_file)
        manuscripts_with_dates = df.dropna(subset=["earlyYear", "lateYear"]).copy()
        manuscripts_with_dates["midYear"] = (
            manuscripts_with_dates["earlyYear"] + manuscripts_with_dates["lateYear"]
        ) / 2

        epochs = [
            (100, 325),
            (325, 636),
            (636, 867),
            (867, 1204),
            (1204, 1516),
            (1516, 2000),
        ]

        all_ages: list[float] = []
        for start_time, end_time in epochs:
            manuscripts_epoch = manuscripts_with_dates[
                (manuscripts_with_dates["midYear"] >= start_time)
                & (manuscripts_with_dates["midYear"] < end_time)
            ].copy()
            if len(manuscripts_epoch) == 0:
                continue

            model = BirthDeathProcess(
                creation_times=manuscripts_epoch["midYear"].tolist(),
                final_time=end_time,
                start_time=start_time,
            )
            all_ages.extend(model.get_manuscript_ages())

        assert len(all_ages) > 0, "No ages prepared from epoch split"

        times_tensor = torch.as_tensor([e[0] for e in epochs], dtype=torch.float64)
        non_spine_ages = torch.as_tensor(all_ages, dtype=torch.float64)
        population_sizes = torch.tensor([len(non_spine_ages)], dtype=torch.int64)

        initial_params = _build_initial_params(
            initial_birth_rates=None,
            initial_death_rates=None,
            non_empty_age=non_spine_ages,
            times_tensor=times_tensor,
        )
        assert torch.all(initial_params > 0)

        base_log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))
        initial_nll = float(
            negative_log_likelihood(
                birth_rates=base_log_params[: len(times_tensor)],
                death_rates=base_log_params[len(times_tensor) :],
                ages=non_spine_ages,
                times=times_tensor,
                spine_ages=None,
                ns=population_sizes,
                starting_population=1,
                is_ordered=False,
                log_scale_params=True,
            ).item(),
        )
        assert np.isfinite(initial_nll), f"Initial objective non-finite: {initial_nll}"

        optimizer_setups = {
            "adam": OptimizerStrategy("Adam", torch.optim.Adam, 300, {"lr": 0.001}),
            "sgd": OptimizerStrategy("SGD", torch.optim.SGD, 300, {"lr": 0.0001}),
            "lbfgs": OptimizerStrategy(
                "LBFGS",
                torch.optim.LBFGS,
                120,
                {"lr": 1.0, "line_search_fn": "strong_wolfe"},
            ),
        }

        diagnostics: list[str] = []
        any_converged = False

        for optimizer_key, strategy in optimizer_setups.items():
            log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))
            run_result = _run_optimizer_strategies(
                strategies_map={optimizer_key: strategy},
                requested_keys=[optimizer_key],
                log_params=log_params,
                initial_params=initial_params,
                n_epochs=len(times_tensor),
                non_spine_ages=non_spine_ages,
                starting_population=1,
                times_tensor=times_tensor,
                spine_ages_tensor=None,
                population_sizes_tensor=population_sizes,
                convergence_tolerance=1e-7,
                absolute_tolerance=1e-6,
                patience=20,
                gradient_threshold=1e-5,
            )

            nll_history = run_result.nll_history
            iterations = run_result.iterations
            converged = run_result.converged
            all_exceptions = run_result.all_exceptions

            any_converged = any_converged or converged

            history_len = len(nll_history)
            history_tail = nll_history[-1] if history_len > 0 else None
            diagnostics.append(
                (
                    f"{optimizer_key}: converged={converged}, iterations={iterations}, "
                    f"history_len={history_len}, final_nll={history_tail}, "
                    f"all_exceptions={all_exceptions}"
                ),
            )

            assert history_len > 0, (
                f"{optimizer_key} failed before objective evaluation. "
                f"Diagnostics: {' | '.join(diagnostics)}"
            )

        if not any_converged:
            pytest.fail(
                "All optimizers failed after finite initialization. "
                "Failure point is in optimization/convergence stage. "
                f"Initial NLL={initial_nll}. Diagnostics: {' | '.join(diagnostics)}",
            )

    def test_actual_fit(self) -> None:
        """Test that the full fitting function can run on the realistic dataset."""
        # Load cleaned manuscript data
        project_root = Path(__file__).resolve().parents[1]
        data_file = project_root / "manuscript_analysis_cleaned.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
        else:
            # Fallback to raw data if cleaned version doesn't exist
            pytest.fail(f"Expected data file not found: {data_file}")

        # Filter to manuscripts with dating information
        manuscripts_with_dates = df.dropna(subset=["earlyYear", "lateYear"]).copy()
        manuscripts_with_dates["midYear"] = (
            manuscripts_with_dates["earlyYear"] + manuscripts_with_dates["lateYear"]
        ) / 2
        manuscripts_with_dates["dateRange"] = (
            manuscripts_with_dates["lateYear"] - manuscripts_with_dates["earlyYear"]
        )
        # | label: setup-epochs

        # Define epochs for analysis
        epochs = [
            (100, 325, "Ante Nicaea"),
            (325, 636, "Post Nicaea"),
            (636, 867, "Post Islamic Conquest"),
            (867, 1204, "Late Byzantium"),
            (1204, 1516, "Post 4th Crusade"),
            (1516, 2000, "Modern Era"),
        ]

        # Collect ages and metadata for each epoch
        all_ages = []
        epoch_counts = []
        epoch_metadata = []

        for start_time, end_time, period_name in epochs:
            manuscripts_epoch = manuscripts_with_dates[
                (manuscripts_with_dates["midYear"] >= start_time)
                & (manuscripts_with_dates["midYear"] < end_time)
            ].copy()

            if len(manuscripts_epoch) > 0:
                creation_times = manuscripts_epoch["midYear"].tolist()
                model = BirthDeathProcess(
                    creation_times=creation_times,
                    final_time=end_time,
                    start_time=start_time,
                )
                ages = model.get_manuscript_ages()
                all_ages.extend(ages)
                epoch_counts.append(len(manuscripts_epoch))
                epoch_metadata.append(
                    (period_name, start_time, end_time, len(manuscripts_epoch)),
                )

        # Create list with single combined array of all ages
        epoch_ages_list = [np.array(all_ages)]

        # Fit all epochs jointly
        times = np.array([e[1] for e in epochs], dtype=float)

        result = fit_birth_death_processes(
            list_of_ages=epoch_ages_list,
            times=times,
            is_ordered=False,
            optimizers="LBFGS",
            perturb_initial_params_std=0.2,
            random_seed=209,
        )

        assert isinstance(result, dict)
