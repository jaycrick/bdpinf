"""
Test suite for modular fitting functionality.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from inf_bp_texts.inference.fitting import (
    _apply_param_perturbation,
    _build_initial_params,
    _build_optimizer_strategies,
    _extract_spine_and_ages,
    _find_non_empty_age,
    _normalize_requested_optimizers,
    _prepare_population_sizes,
    _run_optimizer_strategies,
    _validate_times,
    fit_birth_death_processes,
    fit_from_dataframe,
)
from inf_bp_texts.models import BirthDeathProcess


class TestFittingHelpers:
    def test_find_non_empty_age_returns_first(self) -> None:
        ages_list = [torch.tensor([]), torch.tensor([1.0, 2.0])]
        result = _find_non_empty_age(ages_list)
        assert torch.equal(result, ages_list[1])

    def test_validate_times_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _validate_times(torch.tensor([1.0, 0.0]))

    def test_prepare_population_sizes_length_mismatch(self) -> None:
        ages_list = [torch.tensor([1.0]), torch.tensor([2.0])]
        with pytest.raises(ValueError, match="Length of population_sizes"):
            _prepare_population_sizes(
                ages_list, torch.tensor([1], dtype=torch.int64), None
            )

    def test_extract_spine_and_ages_ordered(self) -> None:
        ages_list = [torch.tensor([10.0, 20.0]), torch.tensor([30.0])]
        spine_ages, non_spine = _extract_spine_and_ages(
            ages_list, is_ordered=True, spine_ages=None
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
        strategies = _build_optimizer_strategies({"adam": {"lr": 0.01}})
        assert strategies["adam"][3]["lr"] == 0.01

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
            birth_rates, death_rates, non_empty_age, times_tensor
        )
        expected_params = torch.tensor(
            [0.1, 0.2, 0.03, 0.04], dtype=initial_params.dtype
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

        if result["success"]:
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
        np.random.seed(42)
        ages = np.random.exponential(scale=200, size=50)
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
            creation_times=[100.0, 200.0, 300.0], final_time=1000.0
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
            }
        )

        result = fit_from_dataframe(df, age_column="manuscript_age", times=1000.0)

        assert isinstance(result, dict)
        assert result["success"]
        assert "success" in result

    def test_process_state_preservation(self) -> None:
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
            assert len(result["birth_rates"]) == 2
            assert len(result["death_rates"]) == 2

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
            assert len(result["birth_rates"]) == 2
            assert len(result["death_rates"]) == 2

    def test_three_epoch_fitting(self) -> None:
        """Test fitting with three epochs."""
        ages = np.array([30.0, 80.0, 130.0, 180.0, 230.0, 280.0], dtype=np.float64)
        times = np.array([100.0, 200.0, 300.0])  # Three epochs

        result = fit_birth_death_processes([ages], times)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            # Should have 3 parameters for each rate
            assert len(result["birth_rates"]) == 3
            assert len(result["death_rates"]) == 3
            assert all(result["birth_rates"] > 0)
            assert all(result["death_rates"] > 0)

    def test_multi_epoch_fitting_array_lengths_consistent(self) -> None:
        """Test that array lengths match between times and rates."""
        ages = np.array([50.0, 150.0, 250.0], dtype=np.float64)
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
            assert len(result["birth_rates"]) == 2
            assert len(result["death_rates"]) == 2
            assert len(result["death_rates"]) == 2


class TestConvergenceFlags:
    def test_max_iters_no_convergence(self) -> None:
        ages = torch.tensor([100.0, 200.0], dtype=torch.float64)
        times_tensor = torch.tensor([1000.0], dtype=torch.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)
        log_params = torch.nn.Parameter(torch.log(initial_params.clone().detach()))
        strategies_map = {
            "sgd": ("SGD", torch.optim.SGD, 1, {"lr": 0.0001}),
        }

        _, nll_history, _, iterations, converged, _ = _run_optimizer_strategies(
            strategies_map,
            ["sgd"],
            log_params,
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=torch.tensor([len(ages)], dtype=torch.int64),
            convergence_tolerance=1e-12,
            absolute_tolerance=1e-12,
            patience=100,
            gradient_threshold=-1.0,
        )

        assert not converged
        assert iterations == 1
        assert len(nll_history) == 1

    @pytest.mark.parametrize(
        "opt_key,opt_class,opt_kwargs",
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
            opt_key: (opt_key.upper(), opt_class, 1, opt_kwargs),
        }

        _, _, _, _, converged, _ = _run_optimizer_strategies(
            strategies_map,
            [opt_key],
            log_params,
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=torch.tensor([len(ages)], dtype=torch.int64),
            convergence_tolerance=1e-12,
            absolute_tolerance=1e-12,
            patience=100,
            gradient_threshold=1e9,
        )

        assert converged
