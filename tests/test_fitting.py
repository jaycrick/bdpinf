"""Test maximum likelihood fitting of birth-death rate models."""

import json
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from bdsurv import PiecewiseConstant

from bdpinf.analysis.multi_epoch_recovery import simulate_zero_birth_second_epoch
from bdpinf.inference import (
    AgeData,
    BirthDeathModel,
    Constraints,
    FitResult,
    fit,
    get_crude_parameter_estimates,
    moment_initial_rates,
    profile_likelihood_interval,
)
from bdpinf.inference.likelihood import log_likelihood_rates
from bdpinf.models.rate_functions import PiecewiseLinearRate
from bdpinf.simulation import simulate_birth_death


def _simulate(
    birth: list[float], death: list[float], times: list[float], seed: int
) -> np.ndarray:
    process = simulate_birth_death(
        PiecewiseConstant(np.array(birth), np.array(times)),
        PiecewiseConstant(np.array(death), np.array(times)),
        random_seed=seed,
        ensure_non_extinct=True,
    )
    return np.asarray(process.ages)


@pytest.fixture(scope="module")
def constant_case() -> tuple[AgeData, BirthDeathModel, FitResult]:
    """Return a single-epoch supercritical dataset and its fit."""
    ages = _simulate([1.0], [0.6], [9.0], seed=2)
    data = AgeData.from_groups([ages], 9.0)
    model = BirthDeathModel.piecewise_constant([9.0])
    return data, model, fit(data, model)


class TestFit:
    """Test the single-start piecewise-constant fit."""

    def test_converges_to_interior_optimum(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        assert result.converged
        assert result.success
        assert result.at_interior_optimum
        assert result.score_norm < 1e-4
        assert result.min_hessian_eig is not None
        assert result.min_hessian_eig > 0
        assert result.method == "trust-exact"
        assert result.starts[0].converged

    def test_log_likelihood_matches_direct_evaluation(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        data, _, result = constant_case
        np.testing.assert_allclose(
            result.log_likelihood,
            float(log_likelihood_rates(data, result.birth, result.death)),
            rtol=1e-10,
        )

    def test_is_a_local_maximum(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        data, model, result = constant_case
        best = result.log_likelihood
        for scale in (0.9, 1.1):
            for i in range(2):
                natural = jnp.asarray(result.natural).at[i].mul(scale)
                moved = float(log_likelihood_rates(data, *model.rates(natural)))
                assert moved <= best + 1e-9

    def test_matches_recorded_reference_fits(self) -> None:
        """Maximised log-likelihoods of the previous optimiser on seeded data."""
        cases = [
            ([0.8], [0.3], [8.0], 2, -38.11206893044985),
            ([1.0], [0.6], [9.0], 2, -23.017548163371135),
            ([0.8, 0.6], [0.3, 0.4], [5.0, 10.0], 2, -49.882908238505465),
        ]
        for birth, death, times, seed, reference in cases:
            ages = _simulate(birth, death, times, seed)
            result = fit(
                AgeData.from_groups([ages], times[-1]),
                BirthDeathModel.piecewise_constant(times),
                compute_covariance=False,
            )
            assert result.log_likelihood >= reference - 1e-4

    def test_result_views(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, model, result = constant_case
        assert set(result.params) == set(model.names)
        np.testing.assert_allclose(result.birth_values, [result.params["birth.v0"]])
        np.testing.assert_allclose(result.death_values, [result.params["death.v0"]])
        assert all(v > 0 for v in result.params.values())

    def test_to_dict_is_json_serialisable(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["converged"] is True
        assert payload["param_names"] == ["birth.v0", "death.v0"]


class TestUncertainty:
    """Test covariance, standard errors and intervals."""

    def test_covariance_is_symmetric_positive_definite(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        assert result.covariance is not None
        np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(result.covariance) > 0)

    def test_standard_errors_and_intervals(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        _, _, result = constant_case
        errors = result.standard_errors
        intervals = result.confidence_intervals(0.95)
        assert errors is not None
        assert intervals is not None
        for name, (lo, hi) in intervals.items():
            assert 0 < lo < result.params[name] < hi
            assert errors[name] > 0
        assert (
            result.confidence_intervals(0.5)["birth.v0"][1] < intervals["birth.v0"][1]
        )  # pyright: ignore[reportOptionalSubscript]

    def test_covariance_can_be_skipped(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=3)
        result = fit(
            AgeData.from_groups([ages], 9.0),
            BirthDeathModel.piecewise_constant([9.0]),
            compute_covariance=False,
        )
        assert result.covariance is None
        assert result.standard_errors is None
        assert result.confidence_intervals() is None

    def test_profile_interval_brackets_estimate(
        self, constant_case: tuple[AgeData, BirthDeathModel, FitResult]
    ) -> None:
        data, _, result = constant_case
        lo, hi = profile_likelihood_interval(result, data, "birth.v0")
        assert lo < result.params["birth.v0"] < hi
        wald_lo, wald_hi = result.confidence_intervals()["birth.v0"]  # pyright: ignore[reportOptionalSubscript]
        assert wald_lo / 1.5 < lo < wald_hi
        assert wald_lo < hi < wald_hi * 1.5


class TestConstraints:
    """Test fitting under constraints."""

    def test_equal_and_fixed_parameters(self) -> None:
        times = np.array([5.0, 10.0])
        simulated = simulate_zero_birth_second_epoch(
            first_epoch_seed=4,
            death_time_seed=5,
            birth_rate_epoch_1=1.0,
            death_rate=0.3,
            times=times,
        )
        alive = simulated.loc[simulated["death_time"].isna()]
        ages = 10.0 - alive["creation_time"].to_numpy(dtype=np.float64)
        data = AgeData.from_groups([ages], 10.0, is_ordered=False)
        model = BirthDeathModel.piecewise_constant(times)
        constraints = Constraints(
            fixed={"birth.v1": 0.0}, equal=[["death.v0", "death.v1"]]
        )
        result = fit(data, model, constraints=constraints)
        assert result.converged
        assert result.params["birth.v1"] == 0.0
        assert result.params["death.v0"] == pytest.approx(result.params["death.v1"])
        assert result.compiled.n_free == 2
        assert result.standard_errors is not None
        assert result.standard_errors["birth.v1"] == 0.0
        assert 0.0 < result.params["death.v0"] < 1.0

    def test_all_parameters_fixed(self) -> None:
        data = AgeData.from_groups([jnp.array([0.3, 0.9, 1.5])], 2.0)
        model = BirthDeathModel.piecewise_constant([2.0])
        result = fit(
            data,
            model,
            constraints=Constraints(fixed={"birth.v0": 1.0, "death.v0": 0.5}),
        )
        assert result.converged
        assert result.compiled.n_free == 0
        assert result.params == {"birth.v0": 1.0, "death.v0": 0.5}
        np.testing.assert_allclose(
            result.log_likelihood,
            float(log_likelihood_rates(data, result.birth, result.death)),
        )


class TestNesting:
    """Test that piecewise constant is a special case of piecewise linear."""

    @pytest.fixture(scope="module")
    def constant_fit(self) -> tuple[AgeData, FitResult]:
        """Return a well-conditioned single-epoch dataset and its constant fit."""
        ages = _simulate([1.0], [0.6], [9.0], seed=1)
        data = AgeData.from_groups([ages], 9.0)
        return data, fit(
            data, BirthDeathModel.piecewise_constant([9.0]), compute_covariance=False
        )

    def test_flat_jump_model_reproduces_piecewise_constant_fit(
        self, constant_fit: tuple[AgeData, FitResult]
    ) -> None:
        data, pc = constant_fit
        pl = BirthDeathModel.piecewise_linear([9.0], continuous=False)
        constraints = Constraints(
            equal=pl.birth.flat_groups("birth.") + pl.death.flat_groups("death.")  # pyright: ignore[reportAttributeAccessIssue]
        )
        flat = fit(data, pl, constraints=constraints, compute_covariance=False)
        assert flat.converged
        np.testing.assert_allclose(flat.log_likelihood, pc.log_likelihood, atol=1e-6)
        np.testing.assert_allclose(
            flat.params["birth.start0"], pc.params["birth.v0"], rtol=1e-4
        )

    def test_free_piecewise_linear_never_fits_worse(
        self, constant_fit: tuple[AgeData, FitResult]
    ) -> None:
        data, pc = constant_fit
        jumps = fit(
            data,
            BirthDeathModel.piecewise_linear([9.0], continuous=False),
            init=pc,
            compute_covariance=False,
        )
        continuous = fit(
            data,
            BirthDeathModel.piecewise_linear([9.0]),
            init=pc,
            compute_covariance=False,
        )
        assert jumps.log_likelihood >= pc.log_likelihood - 1e-6
        assert continuous.log_likelihood >= pc.log_likelihood - 1e-6

    def test_quadrature_matches_exact_at_fitted_constant_rates(
        self, constant_fit: tuple[AgeData, FitResult]
    ) -> None:
        data, pc = constant_fit
        times = jnp.array([9.0])
        birth = PiecewiseLinearRate.from_constant(jnp.asarray(pc.birth_values), times)
        death = PiecewiseLinearRate.from_constant(jnp.asarray(pc.death_values), times)
        np.testing.assert_allclose(
            float(log_likelihood_rates(data, birth, death)),
            pc.log_likelihood,
            atol=1e-8,
        )


class TestInitialisation:
    """Test the forms of ``init`` and the starting heuristics."""

    @pytest.fixture(scope="module")
    def case(self) -> tuple[AgeData, BirthDeathModel]:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        return AgeData.from_groups([ages], 9.0), BirthDeathModel.piecewise_constant(
            [9.0]
        )

    def test_mapping_array_and_rate_function_inits(
        self, case: tuple[AgeData, BirthDeathModel]
    ) -> None:
        data, model = case
        reference = fit(data, model, compute_covariance=False)
        by_name = fit(data, model, init={"birth.v0": 2.0}, compute_covariance=False)
        by_array = fit(data, model, init=[0.5, 0.2], compute_covariance=False)
        by_rates = fit(
            data,
            model,
            init=(reference.birth, reference.death),
            compute_covariance=False,
        )
        for other in (by_name, by_array, by_rates):
            np.testing.assert_allclose(
                other.log_likelihood, reference.log_likelihood, atol=1e-6
            )
        assert by_rates.n_iter <= 1

    def test_bad_init_inputs(self, case: tuple[AgeData, BirthDeathModel]) -> None:
        data, model = case
        with pytest.raises(ValueError, match="Unknown parameter"):
            fit(data, model, init={"nope": 1.0})
        with pytest.raises(ValueError, match="init has shape"):
            fit(data, model, init=[1.0, 2.0, 3.0])
        impossible = Constraints(fixed={"birth.v0": 0.0})
        with pytest.raises(ValueError, match="finite likelihood"):
            fit(data, model, constraints=impossible)

    def test_moment_estimators(self, case: tuple[AgeData, BirthDeathModel]) -> None:
        data, _ = case
        rates = moment_initial_rates(data)
        assert rates is not None
        assert rates[0] > 0
        assert rates[1] > 0
        assert (
            moment_initial_rates(AgeData.from_groups([jnp.array([1.0])], 2.0)) is None
        )

    def test_crude_estimates_validation(self) -> None:
        birth, death = get_crude_parameter_estimates(jnp.array([1.0, 2.0, 3.0]), 10.0)
        assert float(birth[0]) > 0
        assert float(death[0]) > 0
        with pytest.raises(ValueError, match="Cannot provide both"):
            get_crude_parameter_estimates(
                jnp.array([1.0]), 10.0, birth_rates=0.1, death_rates=0.1
            )
        with pytest.raises(ValueError, match="empty age data"):
            get_crude_parameter_estimates(jnp.array([]), 10.0)


class TestMultiStart:
    """Test restarts."""

    def test_second_start_is_swapped_and_best_is_kept(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        result = fit(data, model, n_starts=3, seed=0, compute_covariance=False)
        assert len(result.starts) == 3
        assert [s.swapped for s in result.starts] == [False, True, False]
        assert result.log_likelihood == pytest.approx(
            max(s.log_likelihood for s in result.starts)
        )

    def test_thorough_uses_five_starts_and_keeps_the_best(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        result = fit(data, model, thorough=True, seed=0, compute_covariance=False)
        assert len(result.starts) == 5
        assert result.log_likelihood == pytest.approx(
            max(s.log_likelihood for s in result.starts)
        )

    def test_seeded_restarts_are_reproducible(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        a = fit(data, model, n_starts=3, seed=5, compute_covariance=False)
        b = fit(data, model, n_starts=3, seed=5, compute_covariance=False)
        assert [s.log_likelihood for s in a.starts] == [
            s.log_likelihood for s in b.starts
        ]


class TestPaddingAndCompilation:
    """Test that bucket padding is exact and shares compiled code."""

    def test_padded_and_unpadded_fits_agree(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=1)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        padded = fit(data, model)
        plain = fit(data, model, pad_to_bucket=False)
        np.testing.assert_allclose(
            padded.log_likelihood, plain.log_likelihood, rtol=1e-10
        )
        for name in model.names:
            np.testing.assert_allclose(
                padded.params[name], plain.params[name], rtol=1e-5
            )
        assert padded.covariance is not None
        assert plain.covariance is not None
        np.testing.assert_allclose(padded.covariance, plain.covariance, rtol=1e-4)

    def test_objective_is_reused_across_sample_sizes_in_a_bucket(self) -> None:
        from bdpinf.inference import fitting

        model = BirthDeathModel.piecewise_constant([9.0])
        first = _simulate([1.0], [0.6], [9.0], seed=1)
        second = first[:150]
        assert fitting._bucket(len(first)) == fitting._bucket(len(second))
        fit(AgeData.from_groups([first], 9.0), model, compute_covariance=False)
        cached = len(fitting._OBJECTIVES)
        fit(AgeData.from_groups([second], 9.0), model, compute_covariance=False)
        assert len(fitting._OBJECTIVES) == cached

    @pytest.mark.parametrize(
        ("n", "expected"), [(1, 32), (32, 32), (33, 64), (100, 128), (1000, 1024)]
    )
    def test_bucket_sizes(self, n: int, expected: int) -> None:
        from bdpinf.inference import fitting

        assert fitting._bucket(n) == expected


class TestOptimiserChoice:
    """Test method selection and failure handling."""

    def test_lbfgs_agrees_with_newton(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        data = AgeData.from_groups([ages], 9.0)
        model = BirthDeathModel.piecewise_constant([9.0])
        newton = fit(data, model, compute_covariance=False)
        lbfgs = fit(data, model, method="L-BFGS-B", compute_covariance=False)
        assert lbfgs.converged
        np.testing.assert_allclose(
            lbfgs.log_likelihood, newton.log_likelihood, atol=1e-6
        )

    def test_rejects_unknown_method_and_window_mismatch(self) -> None:
        data = AgeData.from_groups([jnp.array([0.5, 1.0])], 2.0)
        with pytest.raises(ValueError, match="Unsupported method"):
            fit(data, BirthDeathModel.piecewise_constant([2.0]), method="Adam")
        with pytest.raises(ValueError, match="does not match the data"):
            fit(data, BirthDeathModel.piecewise_constant([3.0]))

    def test_unconverged_fit_warns_about_covariance(self) -> None:
        ages = _simulate([1.0], [0.6], [9.0], seed=2)
        with pytest.warns(UserWarning, match="did not converge"):
            result = fit(
                AgeData.from_groups([ages], 9.0),
                BirthDeathModel.piecewise_constant([9.0]),
                init=[5.0, 4.0],
                max_iter=1,
                tol=1e-14,
            )
        assert not result.converged


class TestDataFrame:
    """Test building data from a frame."""

    def test_age_column_detection(self) -> None:
        frame = pd.DataFrame(
            {"age": [100.0, 200.0, 300.0, 400.0], "other": [1, 2, 3, 4]}
        )
        data = AgeData.from_data_frame(frame, 1000.0)
        assert data.n_observations == 4
        result = fit(data, BirthDeathModel.piecewise_constant([1000.0]))
        assert result.params["birth.v0"] > 0

    def test_custom_and_missing_columns(self) -> None:
        frame = pd.DataFrame({"creation_age": [100.0, 200.0, 300.0]})
        assert (
            AgeData.from_data_frame(
                frame, 500.0, age_column="creation_age"
            ).n_observations
            == 3
        )
        with pytest.raises(ValueError, match="not found"):
            AgeData.from_data_frame(frame, 500.0)
        ambiguous = pd.DataFrame({"age_a": [1.0], "age_b": [2.0]})
        with pytest.raises(ValueError, match="Multiple age columns"):
            AgeData.from_data_frame(ambiguous, 500.0)


def test_fit_emits_no_warnings_on_clean_data() -> None:
    """Test that a well-posed fit is silent."""
    ages = _simulate([1.0], [0.6], [9.0], seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit(
            AgeData.from_groups([ages], 9.0),
            BirthDeathModel.piecewise_constant([9.0]),
        )
