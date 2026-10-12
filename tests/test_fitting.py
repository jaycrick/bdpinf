"""Test suite for modular fitting functionality."""

import inspect
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import optax
import pandas as pd
import pytest

from bdpinf.inference import ConvergenceConfig
from bdpinf.inference import uncertainty as uncertainty_module
from bdpinf.inference.fitting import (
    OptimizerAttemptResult,
    OptimizerRunResult,
    OptimizerStrategy,
    ParameterConstraints,
    _apply_param_perturbation,
    _build_initial_params,
    _build_optimizer_strategies,
    _extract_spine_and_ages,
    _find_non_empty_age,
    _normalize_requested_optimizers,
    _prepare_population_sizes,
    _run_optimizer_strategies,
    fit_birth_death_processes,
    fit_from_data_frame,
)
from bdpinf.inference.likelihood import log_likelihood
from bdpinf.models.validation import _validate_times_properties
from bdpinf.process import BirthDeathProcess


class TestFittingHelpers:
    def test_find_non_empty_age_returns_first(self) -> None:
        ages_list = [np.array([]), np.array([1.0, 2.0])]
        result = _find_non_empty_age(ages_list)
        np.testing.assert_array_equal(np.array(result), ages_list[1])

    def test__validate_times_properties_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _validate_times_properties(np.array([1.0, 0.0]))

    def test_prepare_population_sizes_length_mismatch(self) -> None:
        ages_list = [np.array([1.0]), np.array([2.0])]
        with pytest.raises(ValueError, match="Length of population_sizes"):
            _prepare_population_sizes(
                ages_list,
                np.array([1], dtype=np.int32),
                None,
            )

    def test_extract_spine_and_ages_ordered(self) -> None:
        ages_list = [np.array([10.0, 20.0]), np.array([30.0])]
        spine_ages, non_spine = _extract_spine_and_ages(
            ages_list,
            is_ordered=True,
            spine_ages=None,
        )

        assert spine_ages is not None
        np.testing.assert_allclose(np.array(spine_ages), [10.0, 30.0])
        np.testing.assert_allclose(np.array(non_spine), [20.0])

    def test_apply_param_perturbation_is_deterministic(self) -> None:
        params = jnp.array([1.0, 2.0], dtype=jnp.float64)
        perturbed_first = _apply_param_perturbation(params, 0.1, random_seed=7)
        perturbed_second = _apply_param_perturbation(params, 0.1, random_seed=7)

        np.testing.assert_allclose(
            np.array(perturbed_first), np.array(perturbed_second)
        )
        assert bool(jnp.all(perturbed_first > 0))

    def test_build_optimizer_strategies_merges_kwargs(self) -> None:
        lr = 0.01
        strategies = _build_optimizer_strategies(
            {"adam": {"lr": lr}},
            convergence_config=ConvergenceConfig.default(),
        )
        assert strategies["adam"].solver_kwargs["lr"] == lr

    def test_normalize_requested_optimizers_rejects_unknown(self) -> None:
        strategies = _build_optimizer_strategies(
            None,
            convergence_config=ConvergenceConfig.default(),
        )
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            _normalize_requested_optimizers(["Adam", "Bogus"], strategies)

    def test_build_initial_params_with_custom_rates(self) -> None:
        birth_rates = np.array([0.1, 0.2])
        death_rates = np.array([0.03, 0.04])
        non_empty_age = np.array([10.0])
        times_tensor = jnp.array([100.0, 200.0], dtype=jnp.float64)

        initial_params = _build_initial_params(
            birth_rates,
            death_rates,
            non_empty_age,
            times_tensor,
        )
        expected = np.array([0.1, 0.2, 0.03, 0.04])
        np.testing.assert_allclose(np.array(initial_params), expected)


class TestFitting:
    """Test parameter fitting functionality using modular approach."""

    def test_fit_birth_death_processes_defaults_to_unbias_true(self) -> None:
        assert (
            inspect.signature(fit_birth_death_processes)
            .parameters["condition_on_survival"]
            .default
            is True
        )

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
        ages = np.array([100.0, 200.0, 300.0, 500.0, 700.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes([ages], times)

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

        assert isinstance(result, dict)
        assert result["success"]
        assert "success" in result

    def test_fit_from_data_frame_success(self) -> None:
        """Test successful fitting with external DataFrame."""
        rng = np.random.default_rng(42)
        ages = rng.exponential(scale=200, size=50)
        df = pd.DataFrame({"age": ages, "count": np.ones(len(ages))})

        result = fit_from_data_frame(df, age_column="age", times=1000.0)

        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"]

        if result["success"]:
            assert "birth_rates" in result
            assert "death_rates" in result
            assert result["birth_rates"] > 0
            assert result["death_rates"] > 0

    def test_fit_from_data_frame_individual_records(self) -> None:
        """Test fitting with individual individual records."""
        ages = []
        for age, count in [(100.0, 5), (200.0, 3), (300.0, 2), (400.0, 1)]:
            ages.extend([age] * count)

        df = pd.DataFrame({"age": ages})

        result = fit_from_data_frame(df, age_column="age", times=1000.0)

        assert result["success"]
        assert isinstance(result, dict)
        assert "success" in result

    def test_fit_from_data_frame_no_valid_data_points(self) -> None:
        """Test fitting fails with no valid data points."""
        df = pd.DataFrame({"age": []})

        with pytest.raises(ValueError, match="No valid data points for fitting"):
            fit_from_data_frame(df, age_column="age", times=1000.0)

    def test_fit_from_data_frame_missing_age_column(self) -> None:
        """Test fitting fails when age column is missing."""
        df = pd.DataFrame({"wrong_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'age' not found"):
            fit_from_data_frame(df, age_column="age", times=1000.0)

    def test_fit_birth_death_process_different_methods(self) -> None:
        """Test that the function works with the default optimization strategy."""
        ages = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        times = 1000.0

        result = fit_birth_death_processes([ages], times)
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

        assert isinstance(result["birth_rates"], (float, np.float64, np.ndarray))
        assert isinstance(result["death_rates"], (float, np.float64, np.ndarray))
        assert isinstance(result["log_likelihood"], (float, np.float64))

        optimization_result = result["optimization_result"]
        assert "attempts" in optimization_result
        assert isinstance(optimization_result["attempts"], list)
        assert "selected_attempt_index" in optimization_result
        assert "best_attempt_index" in optimization_result
        assert "nll_history" not in result

    def test_fit_failure_uses_attempt_centric_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Failure payload should expose all attempts and drop legacy error fields."""

        def fake_run_optimizer_strategies(
            *args: object,
            **kwargs: object,
        ) -> OptimizerRunResult:
            del args, kwargs
            attempts = [
                OptimizerAttemptResult(
                    optimizer="LBFGS",
                    attempt_index=0,
                    converged=False,
                    iterations=2,
                    nll_history=[10.0, 9.5],
                    exception_type="RuntimeError",
                    exception_message="LBFGS failed",
                ),
                OptimizerAttemptResult(
                    optimizer="Adam",
                    attempt_index=1,
                    converged=False,
                    iterations=3,
                    nll_history=[9.8, 9.7, 9.7],
                    exception_type="RuntimeError",
                    exception_message="Adam failed",
                ),
            ]
            fake_log_params = jnp.log(jnp.array([0.1, 0.1], dtype=jnp.float64))
            return OptimizerRunResult(
                log_params=fake_log_params,
                converged=False,
                selected_attempt_index=0,
                attempts=attempts,
            )

        monkeypatch.setattr(
            "bdpinf.inference.fitting._run_optimizer_strategies",
            fake_run_optimizer_strategies,
        )

        ages = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        result = fit_birth_death_processes([ages], 1000.0)

        assert result["success"] is False
        assert "birth_rates" in result
        assert "death_rates" in result
        assert "log_likelihood" in result
        assert "error" not in result
        assert "error_type" not in result
        assert "errors" not in result
        assert "failed_optimizer_histories" not in result
        assert "nll_history" not in result

        assert np.all(np.asarray(result["birth_rates"]) > 0)
        assert np.all(np.asarray(result["death_rates"]) > 0)

        optimization_result = result["optimization_result"]
        assert optimization_result["selected_attempt_index"] == 0
        assert optimization_result["first_attempt_index"] == 0
        assert optimization_result["last_attempt_index"] == 1
        expected_attempt_count = 2
        assert len(optimization_result["attempts"]) == expected_attempt_count
        assert optimization_result["attempts"][0]["nll_history"] == [10.0, 9.5]
        assert optimization_result["attempts"][1]["nll_history"] == [9.8, 9.7, 9.7]

    def test_process_integration_with_fitting(self) -> None:
        """Test integration between BirthDeathProcess and fitting functions."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        ages = process.ages
        result = fit_birth_death_processes([ages], process.epoch_end_times[-1])

        assert result["success"]
        process.update_parameters(result["birth_rates"], result["death_rates"], result)

        assert process.fitted
        assert process.fit_results is not None
        assert process.birth_rates == result["birth_rates"]
        assert process.death_rates == result["death_rates"]

    def test_fit_from_data_frame_custom_age_column(self) -> None:
        """Test fitting with custom age column name."""
        df = pd.DataFrame(
            {
                "creation_age": [100.0, 200.0, 300.0, 400.0],
                "frequency": [1.0, 1.0, 1.0, 1.0],
            },
        )

        result = fit_from_data_frame(df, age_column="creation_age", times=1000.0)

        assert isinstance(result, dict)
        assert result["success"]
        assert "success" in result

    def test_process_state_preservation(self) -> None:
        """Test that process state is preserved when using external fitting."""
        process = BirthDeathProcess(
            creation_times=[50.0, 150.0, 250.0],
            epoch_end_times=500.0,
        )

        original_ages = np.array(process.ages)
        original_creation_times = process.creation_times.copy()
        original_epoch_end_times = np.array(process.epoch_end_times)

        df = pd.DataFrame({"age": [100.0, 200.0, 300.0], "count": [1.0, 1.0, 1.0]})
        result = fit_from_data_frame(df, age_column="age", times=1000.0)

        np.testing.assert_array_equal(np.array(process.ages), original_ages)
        assert process.creation_times == original_creation_times
        np.testing.assert_array_equal(
            np.array(process.epoch_end_times), original_epoch_end_times
        )

        assert result["success"]
        process.update_parameters(result["birth_rates"], result["death_rates"], result)
        assert process.fitted


class TestMultiEpochFitting:
    """Test multi-epoch parameter fitting functionality."""

    def test_two_epoch_fitting_basic(self) -> None:
        """Test basic two-epoch fitting."""
        ages = np.array([50.0, 150.0, 250.0, 350.0, 450.0], dtype=np.float64)
        times = np.array([250.0, 500.0])

        result = fit_birth_death_processes([ages], times)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert "birth_rates" in result
            assert "death_rates" in result
            assert "log_likelihood" in result
            assert isinstance(result["birth_rates"], np.ndarray)
            assert isinstance(result["death_rates"], np.ndarray)
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)
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
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)

    def test_three_epoch_fitting(self) -> None:
        """Test fitting with three epochs."""
        ages = np.array([30.0, 80.0, 130.0, 180.0, 230.0, 280.0], dtype=np.float64)
        times = np.array([100.0, 200.0, 300.0])

        result = fit_birth_death_processes([ages], times)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
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
            n_epochs = len(times)
            assert len(result["birth_rates"]) == n_epochs
            assert len(result["death_rates"]) == n_epochs

    def test_multi_epoch_vs_single_epoch_different_results(self) -> None:
        """Test that multi-epoch and single-epoch fits give different results."""
        ages = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        total_time = 500.0

        result_single = fit_birth_death_processes([ages], total_time)
        result_multi = fit_birth_death_processes([ages], np.array([250.0, total_time]))

        assert result_single["success"]
        assert result_multi["success"]

        assert isinstance(result_multi["birth_rates"], np.ndarray)
        assert isinstance(result_multi["death_rates"], np.ndarray)

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
            assert result["log_likelihood"] < 0

    def test_multi_epoch_from_data_frame(self) -> None:
        """Test multi-epoch fitting from DataFrame."""
        ages = [100.0, 200.0, 300.0, 400.0, 500.0]
        df = pd.DataFrame({"age": ages})
        times = np.array([250.0, 550.0])

        result = fit_from_data_frame(df, age_column="age", times=times)

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)

    def test_multi_epoch_from_data_frame_large_initial_pop(self) -> None:
        """Test multi-epoch fitting from DataFrame, with initial population sizes."""
        ages = [100.0, 200.0, 300.0, 400.0, 500.0]
        df = pd.DataFrame({"age": ages})
        times = np.array([250.0, 550.0])

        result = fit_from_data_frame(
            df,
            age_column="age",
            times=times,
            initial_population=10,
        )

        assert isinstance(result, dict)
        assert "success" in result

        if result["success"]:
            assert len(result["birth_rates"]) == len(times)
            assert len(result["death_rates"]) == len(times)


class TestConvergenceFlags:
    def test_fit_birth_death_process_includes_uncertainty_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ages = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        sentinel_covariance = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=np.float64)

        def fake_estimate_uncertainty(*args: object, **kwargs: object) -> np.ndarray:
            del args, kwargs
            return sentinel_covariance

        monkeypatch.setattr(
            "bdpinf.inference.fitting.estimate_mle_covariance",
            fake_estimate_uncertainty,
        )

        result = fit_birth_death_processes([ages], 1000.0)

        assert result["success"]
        assert "uncertainty" in result
        uncertainty = np.asarray(result["uncertainty"], dtype=np.float64)
        assert uncertainty.shape == sentinel_covariance.shape
        np.testing.assert_allclose(uncertainty, sentinel_covariance)

    def test_fit_birth_death_process_can_skip_uncertainty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ages = np.array([100.0, 200.0, 300.0], dtype=np.float64)

        called = {"value": False}

        def fake_estimate_uncertainty(
            *args: object,
            **kwargs: object,
        ) -> np.ndarray:
            del args, kwargs
            called["value"] = True
            return np.eye(2, dtype=np.float64)

        monkeypatch.setattr(
            "bdpinf.inference.fitting.estimate_mle_covariance",
            fake_estimate_uncertainty,
        )

        result = fit_birth_death_processes([ages], 1000.0, estimate_uncertainty=False)

        assert result["success"]
        assert "uncertainty" not in result
        assert called["value"] is False

    def test_uncertainty_helper_sets_fixed_param_se_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        free_log_params = jnp.array([0.0], dtype=jnp.float64)
        non_spine_ages = jnp.array([100.0], dtype=jnp.float64)
        times_tensor = jnp.array([1000.0], dtype=jnp.float64)
        population_sizes_tensor = jnp.array([1], dtype=jnp.int32)

        def full_log_params_fn(theta_free: jax.Array) -> jax.Array:
            return jnp.stack(
                [
                    theta_free[0] + jnp.log(jnp.array(2.0, dtype=jnp.float64)),
                    jnp.log(jnp.array(0.5, dtype=jnp.float64)),
                ],
            )

        fake_hess_value = jnp.array([[2.0]], dtype=jnp.float64)
        fake_jac_value = jnp.array([[1.0], [0.0]], dtype=jnp.float64)

        def fake_hessian(fn: object) -> object:
            del fn

            def _inner(x: object) -> jax.Array:
                del x
                return fake_hess_value

            return _inner

        def fake_jacobian(fn: object) -> object:
            del fn

            def _inner(x: object) -> jax.Array:
                del x
                return fake_jac_value

            return _inner

        monkeypatch.setattr(jax, "hessian", fake_hessian)
        monkeypatch.setattr(jax, "jacobian", fake_jacobian)

        covariance = uncertainty_module.estimate_mle_covariance(
            free_log_params=free_log_params,
            n_epochs=1,
            full_log_params_fn=full_log_params_fn,
            non_spine_ages=non_spine_ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=population_sizes_tensor,
            initial_population=1,
        )

        assert covariance.shape == (2, 2)
        assert float(covariance[0, 0]) == pytest.approx(0.5)
        assert float(covariance[1, 1]) == pytest.approx(0.0, abs=1e-12)
        assert float(covariance[1, 0]) == pytest.approx(0.0, abs=1e-12)
        assert float(covariance[0, 1]) == pytest.approx(0.0, abs=1e-12)

    def test_log_space_confidence_intervals_are_exponentiated(self) -> None:
        log_parameter_estimates = jnp.array(
            [np.log(2.0), np.log(4.0)],
            dtype=jnp.float64,
        )
        covariance_log = jnp.diag(
            jnp.array([0.25, 0.0], dtype=jnp.float64),
        )

        intervals = uncertainty_module.confidence_intervals_from_log_covariance(
            log_parameter_estimates,
            covariance_log,
        )

        z_value = math.sqrt(2.0) * float(
            jax.scipy.special.erfinv(jnp.array(0.95, dtype=jnp.float64))
        )
        ses = np.array([0.5, 0.0])
        log_ests = np.array([np.log(2.0), np.log(4.0)])
        expected_lower = np.exp(log_ests - z_value * ses)
        expected_upper = np.exp(log_ests + z_value * ses)

        assert intervals.shape == (2, 2)
        np.testing.assert_allclose(
            np.array(intervals[:, 0]), expected_lower, rtol=1e-10
        )
        np.testing.assert_allclose(
            np.array(intervals[:, 1]), expected_upper, rtol=1e-10
        )

    def test_max_iters_no_convergence(self) -> None:
        ages = jnp.array([100.0, 200.0], dtype=jnp.float64)
        times_tensor = jnp.array([1000.0], dtype=jnp.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)
        strategies_map = {
            "sgd": OptimizerStrategy(
                name="SGD",
                solver_factory=optax.sgd,
                max_iters=1,
                solver_kwargs={"lr": 0.0001},
            ),
        }

        convergence_config = ConvergenceConfig(
            gradient_threshold=-1.0,
            absolute_tolerance=1e-12,
            patience=100,
        )

        run_result = _run_optimizer_strategies(
            strategies_map,
            ["sgd"],
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=jnp.array([len(ages)], dtype=jnp.int32),
            initial_population=1,
            convergence_config=convergence_config,
        )

        assert not run_result.converged
        assert len(run_result.attempts) == 1
        attempt = run_result.attempts[0]
        assert attempt.iterations == 1, (
            f"Expected 1 iteration, got {attempt.iterations}"
        )
        assert len(attempt.nll_history) == 1

    @pytest.mark.parametrize(
        ("opt_key", "solver_factory", "solver_kwargs"),
        [
            ("sgd", optax.sgd, {"lr": 0.0001}),
            (
                "lbfgs",
                lambda **_: None,
                {"lr": 1.0},
            ),
        ],
    )
    def test_converged_on_gradient_threshold(
        self,
        opt_key: str,
        solver_factory: object,
        solver_kwargs: dict[str, float],
    ) -> None:
        ages = jnp.array([100.0, 200.0], dtype=jnp.float64)
        times_tensor = jnp.array([1000.0], dtype=jnp.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)
        strategies_map = {
            opt_key: OptimizerStrategy(
                name=opt_key.upper(),
                solver_factory=solver_factory,
                max_iters=100,
                solver_kwargs=solver_kwargs,
            ),
        }

        convergence_config = ConvergenceConfig(
            gradient_threshold=1e9,
            absolute_tolerance=1e-12,
            patience=0,
        )

        run_result = _run_optimizer_strategies(
            strategies_map,
            [opt_key],
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=jnp.array([len(ages)], dtype=jnp.int32),
            initial_population=1,
            convergence_config=convergence_config,
        )

        assert run_result.converged

    def test_collects_exceptions_across_optimizer_attempts(self) -> None:
        ages = jnp.array([100.0, 200.0], dtype=jnp.float64)
        times_tensor = jnp.array([1000.0], dtype=jnp.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)

        strategies_map = {
            "sgd": OptimizerStrategy(
                name="SGD",
                solver_factory=optax.sgd,
                max_iters=1,
                solver_kwargs={"lr": 0.0001},
            ),
            "adam": OptimizerStrategy(
                name="Adam",
                solver_factory=optax.adam,
                max_iters=1,
                solver_kwargs={"lr": 0.001},
            ),
        }
        convergence_config = ConvergenceConfig(
            gradient_threshold=-1.0,
            absolute_tolerance=1e-12,
            patience=100,
        )

        run_result = _run_optimizer_strategies(
            strategies_map,
            ["sgd", "adam"],
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=jnp.array([len(ages)], dtype=jnp.int32),
            initial_population=1,
            convergence_config=convergence_config,
        )

        assert not run_result.converged
        assert len(run_result.attempts) == len(strategies_map)
        assert all(
            attempt.exception_type is not None for attempt in run_result.attempts
        )

    def test_lbfgs_failed_run_reports_nonzero_iterations_regression(self) -> None:
        """Regression test: failed LBFGS runs should report executed iterations."""
        ages = jnp.array([100.0, 200.0], dtype=jnp.float64)
        times_tensor = jnp.array([1000.0], dtype=jnp.float64)
        initial_params = _build_initial_params(None, None, ages, times_tensor)

        strategies_map = {
            "lbfgs": OptimizerStrategy(
                name="LBFGS",
                solver_factory=lambda **_: None,
                max_iters=1,
                solver_kwargs={"lr": 1.0},
            ),
        }
        convergence_config = ConvergenceConfig(
            gradient_threshold=-1.0,
            absolute_tolerance=1e-12,
            patience=100,
        )

        run_result = _run_optimizer_strategies(
            strategies_map,
            ["lbfgs"],
            initial_params,
            n_epochs=len(times_tensor),
            non_spine_ages=ages,
            times_tensor=times_tensor,
            spine_ages_tensor=None,
            population_sizes_tensor=jnp.array([len(ages)], dtype=jnp.int32),
            initial_population=1,
            convergence_config=convergence_config,
        )

        assert not run_result.converged
        assert len(run_result.attempts) == 1
        attempt = run_result.attempts[0]
        assert len(attempt.nll_history) >= 1
        assert attempt.iterations == 1


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
        individuals_with_dates = df.dropna(subset=["earlyYear", "lateYear"]).copy()
        individuals_with_dates["midYear"] = (
            individuals_with_dates["earlyYear"] + individuals_with_dates["lateYear"]
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
            individuals_epoch = individuals_with_dates[
                (individuals_with_dates["midYear"] >= start_time)
                & (individuals_with_dates["midYear"] < end_time)
            ].copy()
            if len(individuals_epoch) == 0:
                continue

            model = BirthDeathProcess(
                creation_times=individuals_epoch["midYear"].tolist(),
                epoch_end_times=end_time,
            )
            all_ages.extend(model.ages)

        assert len(all_ages) > 0, "No ages prepared from epoch split"

        times_tensor = jnp.array([e[0] for e in epochs], dtype=jnp.float64)
        non_spine_ages = jnp.array(all_ages, dtype=jnp.float64)
        population_sizes = jnp.array([len(non_spine_ages)], dtype=jnp.int32)

        initial_params = _build_initial_params(
            initial_birth_rates=None,
            initial_death_rates=None,
            non_empty_age=non_spine_ages,
            times_tensor=times_tensor,
        )
        assert bool(jnp.all(initial_params > 0))

        log_params_initial = jnp.log(initial_params)
        constraints = ParameterConstraints()
        initial_nll = float(
            -log_likelihood(
                birth_rates=log_params_initial[: len(times_tensor)],
                death_rates=log_params_initial[len(times_tensor) :],
                ages=non_spine_ages,
                times=times_tensor,
                spine_ages=None,
                ns=population_sizes,
                initial_population=1,
                is_ordered=False,
                log_scale_params=True,
            ),
        )
        assert np.isfinite(initial_nll), f"Initial objective non-finite: {initial_nll}"

        optimizer_setups = {
            "adam": OptimizerStrategy(
                name="Adam",
                solver_factory=optax.adam,
                max_iters=300,
                solver_kwargs={"lr": 0.001},
            ),
            "sgd": OptimizerStrategy(
                name="SGD",
                solver_factory=optax.sgd,
                max_iters=300,
                solver_kwargs={"lr": 0.0001},
            ),
            "lbfgs": OptimizerStrategy(
                name="LBFGS",
                solver_factory=lambda **_: None,
                max_iters=120,
                solver_kwargs={"lr": 1.0},
            ),
        }

        diagnostics: list[str] = []
        any_converged = False

        for optimizer_key, strategy in optimizer_setups.items():
            convergence_config = ConvergenceConfig(
                gradient_threshold=1e-5,
                absolute_tolerance=1e-6,
                patience=20,
            )
            run_result = _run_optimizer_strategies(
                strategies_map={optimizer_key: strategy},
                requested_keys=[optimizer_key],
                initial_params=initial_params,
                n_epochs=len(times_tensor),
                non_spine_ages=non_spine_ages,
                initial_population=1,
                times_tensor=times_tensor,
                spine_ages_tensor=None,
                population_sizes_tensor=population_sizes,
                convergence_config=convergence_config,
                param_constraints=constraints,
            )

            assert len(run_result.attempts) == 1
            attempt = run_result.attempts[0]
            nll_history = attempt.nll_history
            iterations = attempt.iterations
            converged = run_result.converged
            exception_message = attempt.exception_message

            any_converged = any_converged or converged

            history_len = len(nll_history)
            history_tail = nll_history[-1] if history_len > 0 else None
            diagnostics.append(
                (
                    f"{optimizer_key}: converged={converged}, iterations={iterations}, "
                    f"history_len={history_len}, final_nll={history_tail}, "
                    f"exception={exception_message}"
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
        """Test that the full fitting function can run on a realistic dataset."""
        project_root = Path(__file__).resolve().parents[1]
        data_file = project_root / "manuscript_analysis_cleaned.csv"
        if not data_file.exists():
            pytest.skip(f"Diagnostic dataset not available: {data_file}")
        df = pd.read_csv(data_file)

        individuals_with_dates = df.dropna(subset=["earlyYear", "lateYear"]).copy()
        individuals_with_dates["midYear"] = (
            individuals_with_dates["earlyYear"] + individuals_with_dates["lateYear"]
        ) / 2
        individuals_with_dates["dateRange"] = (
            individuals_with_dates["lateYear"] - individuals_with_dates["earlyYear"]
        )

        epochs = [
            (100, 325, "Ante Nicaea"),
            (325, 636, "Post Nicaea"),
            (636, 867, "Post Islamic Conquest"),
            (867, 1204, "Late Byzantium"),
            (1204, 1516, "Post 4th Crusade"),
            (1516, 2000, "Modern Era"),
        ]

        all_ages = []

        for start_time, end_time, _period_name in epochs:
            individuals_epoch = individuals_with_dates[
                (individuals_with_dates["midYear"] >= start_time)
                & (individuals_with_dates["midYear"] < end_time)
            ].copy()

            if len(individuals_epoch) > 0:
                creation_times = individuals_epoch["midYear"].tolist()
                model = BirthDeathProcess(
                    creation_times=creation_times,
                    epoch_end_times=end_time,
                )
                ages = model.ages
                all_ages.extend(ages)

        epoch_ages_list = [np.array(all_ages)]
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
