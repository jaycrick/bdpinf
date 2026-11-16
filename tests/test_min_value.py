"""Test lower bounds on rate families (min_birth, min_death)."""

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference import (
    AgeData,
    BayesianMCMCConfig,
    BayesianPriorConfig,
    BirthDeathModel,
    Constraints,
    FreeKnots,
    Param,
    PiecewiseConstantModel,
    PiecewiseLinearModel,
    fit,
    fit_bayesian,
    fit_knots,
    profile_likelihood_interval,
    sample_prior,
)
from bdpinf.models.rate_functions import PiecewiseConstantRate
from bdpinf.simulation import simulate_ages_from_rates

_T = 100.0
_BOUND = 0.02


def _simulate(n_groups: int, birth: float, death: float, seed: int) -> AgeData:
    times = jnp.array([_T])
    rng = np.random.default_rng(seed)
    groups = [
        simulate_ages_from_rates(
            PiecewiseConstantRate(jnp.array([birth]), times),
            PiecewiseConstantRate(jnp.array([death]), times),
            rng,
        )
        for _ in range(n_groups)
    ]
    return AgeData.from_groups(
        [np.asarray(g.non_spine_ages) for g in groups],
        _T,
        spine_ages=[float(g.spine_age) for g in groups],
    )


@pytest.fixture(scope="module")
def low_death_data() -> AgeData:
    """Ages whose unconstrained death-rate MLE is far below ``_BOUND``."""
    return _simulate(30, 0.03, 0.001, seed=1)


class TestMinValueSpec:
    """Test the model spec and the shifted log transform."""

    def test_defaults_are_zero(self) -> None:
        model = BirthDeathModel.piecewise_constant([50.0, _T])
        assert all(p.lower == 0.0 for p in model.params)
        assert PiecewiseConstantModel([_T]).min_value == 0.0
        assert PiecewiseLinearModel([_T]).min_value == 0.0

    def test_bounds_land_on_the_right_family(self) -> None:
        model = BirthDeathModel.piecewise_linear(
            [50.0, _T], min_birth=0.1, min_death=0.2, continuous=False
        )
        lowers = {p.name: p.lower for p in model.params}
        assert all(v == 0.1 for k, v in lowers.items() if k.startswith("birth."))
        assert all(v == 0.2 for k, v in lowers.items() if k.startswith("death."))

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_rejects_bad_minimum(self, bad: float) -> None:
        with pytest.raises(ValueError, match="min_value"):
            PiecewiseConstantModel([_T], bad)
        with pytest.raises(ValueError, match="min_value"):
            PiecewiseLinearModel([_T], True, bad)
        with pytest.raises(ValueError, match="minimum value"):
            Param("x", lower=bad)

    def test_round_trip_with_lower(self) -> None:
        model = BirthDeathModel.piecewise_constant(
            [50.0, _T], min_birth=0.05, min_death=0.3
        )
        compiled = Constraints().compile(model.params)
        natural = jnp.array([0.06, 0.5, 0.31, 2.0])
        free = compiled.reduce(natural)
        np.testing.assert_allclose(
            free, np.log(np.asarray(natural) - [0.05, 0.05, 0.3, 0.3])
        )
        np.testing.assert_allclose(compiled.expand(free), natural, rtol=1e-12)
        assert np.all(
            np.asarray(compiled.expand(jnp.full(4, -50.0))) >= [0.05] * 2 + [0.3] * 2
        )

    def test_reduce_lifts_values_at_or_below_the_bound(self) -> None:
        model = BirthDeathModel.piecewise_constant([_T], min_death=_BOUND)
        compiled = Constraints().compile(model.params)
        free = np.asarray(compiled.reduce(jnp.array([0.01, 0.0])))
        assert np.all(np.isfinite(free))
        assert float(compiled.expand(jnp.asarray(free))[1]) > _BOUND

    def test_starting_values_are_above_the_bound(self) -> None:
        model = BirthDeathModel.piecewise_linear([50.0, _T], min_death=_BOUND)
        natural = np.asarray(model.constant_natural(0.01, 0.001))
        assert np.all(natural[model.n_birth :] > _BOUND)
        zero = PiecewiseConstantRate(jnp.array([0.0, 0.0]), jnp.array([50.0, _T]))
        warm = np.asarray(model.natural_from_rates(zero, zero))
        assert np.all(warm[model.n_birth :] > _BOUND)
        assert model.swapped(jnp.asarray(natural)) is not None

    def test_constraint_validation(self) -> None:
        model = BirthDeathModel.piecewise_constant([50.0, _T], min_death=_BOUND)
        with pytest.raises(ValueError, match="domain"):
            Constraints(fixed={"death.v0": 0.01}).compile(model.params)
        compiled = Constraints(fixed={"death.v0": _BOUND}).compile(model.params)
        assert float(compiled.expand(jnp.zeros(3))[2]) == _BOUND
        with pytest.raises(ValueError, match="minimum values"):
            Constraints(equal=[["birth.v0", "death.v0"]]).compile(model.params)
        Constraints(equal=[["death.v0", "death.v1"]]).compile(model.params)


class TestFitWithMinValue:
    """Test fit, fit_knots and fit_bayesian with a bounded family."""

    def test_default_is_unchanged(self, low_death_data: AgeData) -> None:
        for build in (
            BirthDeathModel.piecewise_constant,
            BirthDeathModel.piecewise_linear,
        ):
            plain = fit(low_death_data, build([_T]))
            zero = fit(low_death_data, build([_T], min_birth=0.0, min_death=0.0))
            assert plain.log_likelihood == zero.log_likelihood
            assert plain.params == zero.params

    def test_bounded_fit_respects_the_bound(self, low_death_data: AgeData) -> None:
        free = fit(low_death_data, BirthDeathModel.piecewise_constant([_T]))
        bounded = fit(
            low_death_data,
            BirthDeathModel.piecewise_constant([_T], min_death=_BOUND),
        )
        assert free.params["death.v0"] < _BOUND
        assert bounded.params["death.v0"] >= _BOUND
        assert bounded.log_likelihood <= free.log_likelihood + 1e-8
        assert not bounded.at_interior_optimum
        assert bounded.params["death.v0"] == pytest.approx(_BOUND, abs=1e-5)
        intervals = bounded.confidence_intervals()
        assert intervals is not None
        lo, hi = intervals["death.v0"]
        assert lo >= _BOUND
        assert not np.isnan(hi)
        assert bounded.covariance is not None
        assert np.all(np.isfinite(bounded.covariance[:1, :1]))
        low, _ = profile_likelihood_interval(bounded, low_death_data, "death.v0")
        assert low == _BOUND

    def test_linear_fit_respects_the_bound(self, low_death_data: AgeData) -> None:
        model = BirthDeathModel.piecewise_linear([_T], min_death=_BOUND)
        result = fit(low_death_data, model, n_starts=2, seed=0)
        assert np.all(result.death_values >= _BOUND)

    def test_fit_knots_respects_the_bound(self) -> None:
        data = _simulate(25, 0.05, 0.001, seed=3)
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(1, _T, bounds=(20.0, 80.0)), min_death=_BOUND
        )
        result = fit_knots(data, model, n_candidates=8, seed=0)
        assert np.all(result.fit.death_values >= _BOUND)
        assert np.all(np.isfinite(result.knot_times))

    def test_bayesian_draws_respect_the_bound(self, low_death_data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant([_T], min_death=_BOUND)
        result = fit_bayesian(
            low_death_data,
            model,
            mcmc=BayesianMCMCConfig(
                num_samples=40, warmup_steps=40, disable_progressbar=True
            ),
            rng_seed=0,
        )
        assert np.all(result.death_samples >= _BOUND)

    def test_sample_prior_and_exponential_validation(self) -> None:
        model = BirthDeathModel.piecewise_constant([50.0, _T], min_death=_BOUND)
        draws = sample_prior(model, n=500, seed=0)
        assert np.all(draws["death.v0"] >= _BOUND)
        assert np.all(draws["birth.v0"] > 0)
        with pytest.raises(ValueError, match="minimum value"):
            sample_prior(model, BayesianPriorConfig(exponential={"death.v0": 0.1}), 10)
        ok = sample_prior(model, BayesianPriorConfig(exponential={"birth.v0": 0.1}), 10)
        assert ok["birth.v0"].shape == (10,)
