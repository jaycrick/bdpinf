"""Test free knot times: the spec, the interval transform, MLE and Bayesian fits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.bayesian import BayesianMCMCConfig, fit_bayesian
from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import fit
from bdpinf.inference.knots import KNOT_METHODS, fit_knots
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import BirthDeathModel, FreeKnots
from bdpinf.models.rate_functions import PiecewiseConstantRate
from bdpinf.simulation.age_sampling import simulate_ages_from_rates

_T = 100.0
_TRUE_KNOT = 50.0


def _simulate(n_groups: int = 25, seed: int = 3) -> AgeData:
    """Simulate groups whose birth rate drops at ``_TRUE_KNOT``."""
    times = jnp.array([_TRUE_KNOT, _T])
    birth = PiecewiseConstantRate(jnp.array([0.09, 0.02]), times)
    death = PiecewiseConstantRate(jnp.array([0.01, 0.01]), times)
    rng = np.random.default_rng(seed)
    groups = [simulate_ages_from_rates(birth, death, rng) for _ in range(n_groups)]
    return AgeData.from_groups(
        [np.asarray(g.non_spine_ages) for g in groups],
        _T,
        spine_ages=[float(g.spine_age) for g in groups],
    )


@pytest.fixture(scope="module")
def data() -> AgeData:
    """Simulate groups with a rate change at ``_TRUE_KNOT``."""
    return _simulate()


class TestFreeKnots:
    """Test free knot times."""

    def test_defaults_and_names(self) -> None:
        knots = FreeKnots(2, end_time=90.0)
        assert knots.names == ("tau0", "tau1")
        assert knots.initial_times == pytest.approx((30.0, 60.0))
        assert knots.ranges == ((0.0, 90.0), (0.0, 90.0))

    def test_shared_and_per_knot_ranges(self) -> None:
        shared = FreeKnots(2, 100.0, bounds=(20.0, 80.0))
        assert shared.ranges == ((20.0, 80.0),) * 2
        per_knot = FreeKnots(2, 100.0, bounds=[(10.0, 40.0), (50.0, 90.0)])
        assert per_knot.initial_times == pytest.approx((25.0, 70.0))

    @pytest.mark.parametrize(
        "bounds",
        [(50.0, 20.0), (-1.0, 50.0), (10.0, 120.0), [(10.0, 60.0), (50.0, 90.0)]],
    )
    def test_rejects_bad_bounds(self, bounds: object) -> None:
        with pytest.raises(ValueError, match=r"range|pair"):
            FreeKnots(2, 100.0, bounds=bounds)  # type: ignore[arg-type]

    def test_rejects_bad_initial(self) -> None:
        with pytest.raises(ValueError, match="initial"):
            FreeKnots(1, 100.0, bounds=(10.0, 20.0), initial=[50.0])

    def test_model_layout(self) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(2, 100.0))
        assert model.names[-2:] == ("tau0", "tau1")
        assert model.n_knots == 2
        assert len(model.names) == 3 + 3 + 2

    def test_epoch_count_mismatch(self) -> None:
        fixed = BirthDeathModel.piecewise_constant([50.0, 100.0])
        with pytest.raises(ValueError, match="epochs"):
            BirthDeathModel(fixed.birth, fixed.death, FreeKnots(2, 100.0))

    def test_interval_transform_round_trips(self) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(1, _T, bounds=(20.0, 80.0))
        )
        compiled = Constraints().compile(model.params)
        natural = jnp.array([0.1, 0.2, 0.05, 0.03, 35.0])
        back = compiled.expand(compiled.reduce(natural))
        np.testing.assert_allclose(back, natural, rtol=1e-9)
        assert 20.0 < float(compiled.expand(jnp.full(5, 2.0))[-1]) < 80.0

    def test_constraints_reject_knot_issues(self) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=(20.0, 80.0))
        )
        with pytest.raises(ValueError, match="domain"):
            Constraints(fixed={"tau0": 90.0}).compile(model.params)
        with pytest.raises(ValueError, match="knot time"):
            Constraints(equal=[["tau0", "tau1"]]).compile(model.params)

    @pytest.mark.parametrize("continuous", [True, False])
    def test_matches_fixed_epochs(self, data: AgeData, continuous: bool) -> None:
        free = BirthDeathModel.piecewise_linear(
            FreeKnots(1, _T, initial=[40.0]), continuous=continuous
        )
        fixed = BirthDeathModel.piecewise_linear([40.0, _T], continuous=continuous)
        natural = np.array(free.constant_natural(0.05, 0.02))
        n_rates = free.n_birth + free.n_death
        value = fixed.constant_natural(0.05, 0.02)
        b_free, d_free = free.rates(jnp.asarray(natural))
        b_fix, d_fix = fixed.rates(value)
        np.testing.assert_allclose(natural[n_rates:], [40.0])
        np.testing.assert_allclose(
            log_likelihood_rates(data, b_free, d_free),
            log_likelihood_rates(data, b_fix, d_fix),
            rtol=1e-10,
        )

    def test_gradient_wrt_knot_is_finite(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_linear(FreeKnots(1, _T))
        base = np.array(model.constant_natural(0.05, 0.02))
        base[:3] = [0.09, 0.05, 0.02]

        def ll(tau: jax.Array) -> jax.Array:
            natural = jnp.asarray(base).at[-1].set(tau[0])
            return log_likelihood_rates(data, *model.rates(natural))

        grad = jax.grad(ll)(jnp.array([37.3]))
        assert np.all(np.isfinite(grad))
        eps = 1e-5
        numeric = (ll(jnp.array([37.3 + eps])) - ll(jnp.array([37.3 - eps]))) / (
            2 * eps
        )
        assert abs(grad[0]) > 0
        np.testing.assert_allclose(grad[0], numeric, rtol=1e-4)


class TestFitKnots:
    """Test the profile-likelihood search over knot times."""

    def test_recovers_knot(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        result = fit_knots(data, model, seed=0)
        assert abs(result.knot_times[0] - _TRUE_KNOT) < 8.0
        assert result.fit.params["birth.v0"] > result.fit.params["birth.v1"]
        assert result.profile_points.shape[1] == 1
        assert result.log_likelihood == pytest.approx(result.fit.log_likelihood)

    def test_respects_bounds(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(1, _T, bounds=(60.0, 90.0))
        )
        result = fit_knots(data, model, seed=0)
        assert 60.0 <= result.knot_times[0] <= 90.0

    def test_beats_fixed_wrong_knot(self, data: AgeData) -> None:
        wrong = fit(data, BirthDeathModel.piecewise_constant([80.0, _T]))
        free = fit_knots(
            data, BirthDeathModel.piecewise_constant(FreeKnots(1, _T)), seed=0
        )
        assert free.log_likelihood > wrong.log_likelihood

    def test_two_knots_sorted(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(2, _T))
        result = fit_knots(data, model, n_candidates=16, n_refine=1, seed=0)
        assert result.knot_times.shape == (2,)
        assert result.knot_times[0] < result.knot_times[1]

    def test_requires_free_knots_and_free_constraints(self, data: AgeData) -> None:
        with pytest.raises(ValueError, match="free knots"):
            fit_knots(data, BirthDeathModel.piecewise_constant([50.0, _T]))
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        with pytest.raises(ValueError, match="Knot times"):
            fit_knots(data, model, constraints=Constraints(fixed={"tau0": 50.0}))


class TestKnotMethods:
    """Test the search methods of fit_knots."""

    @pytest.mark.parametrize("method", ["brent", "de-profile", "grid-newton"])
    def test_each_method_recovers_knot(self, data: AgeData, method: str) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        result = fit_knots(data, model, methods=method, seed=0)
        assert result.method == method
        assert list(result.method_results) == [method]
        assert abs(result.knot_times[0] - _TRUE_KNOT) < 10.0

    def test_nelder_mead_runs(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(1, _T, bounds=(30.0, 70.0))
        )
        result = fit_knots(data, model, methods="nelder-mead", seed=0)
        assert 30.0 <= result.knot_times[0] <= 70.0

    def test_several_methods_keep_the_best(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        names = ["brent", "de-profile", "grid-newton"]
        result = fit_knots(data, model, methods=names, seed=0)
        assert list(result.method_results) == names
        top = max(r.log_likelihood for r in result.method_results.values())
        assert result.method_results[result.method].log_likelihood == top
        assert result.log_likelihood >= top - 1e-6

    def test_unknown_method_raises(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        with pytest.raises(ValueError, match="Unknown knot methods"):
            fit_knots(data, model, methods=["brent", "nope"])
        assert "brent" in KNOT_METHODS

    def test_fixed_knot_is_left_out_of_the_search(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=[(10.0, 80.0), (80.0, 99.0)])
        )
        result = fit_knots(
            data, model, constraints=Constraints(fixed={"tau1": 90.0}), seed=0
        )
        assert result.knot_times[1] == pytest.approx(90.0)
        assert result.profile_points.shape[1] == 1

    @pytest.mark.parametrize("method", ["brent", "de-profile"])
    def test_two_free_knots(self, data: AgeData, method: str) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=[(10.0, 80.0), (80.0, 99.0)])
        )
        result = fit_knots(data, model, methods=method, seed=0)
        assert result.knot_times[0] < result.knot_times[1]
        assert result.profile_points.shape[1] == 2


class TestJointFit:
    """Test fit() and fit_bayesian() with free knots."""

    def test_joint_fit_reports_sorted_knots(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_linear(FreeKnots(2, _T))
        result = fit(data, model, compute_covariance=False)
        knots = result.knot_times
        assert knots is not None
        assert knots[0] < knots[1]
        assert np.isfinite(result.log_likelihood)

    def test_fixed_epoch_fit_has_no_knot_times(self, data: AgeData) -> None:
        result = fit(data, BirthDeathModel.piecewise_constant([50.0, _T]))
        assert result.knot_times is None

    def test_bayesian_knot_draws(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_linear(FreeKnots(2, _T, bounds=(10.0, 90.0)))
        start = fit(data, model, compute_covariance=False)
        result = fit_bayesian(
            data,
            model,
            init=start,
            mcmc=BayesianMCMCConfig(num_samples=60, warmup_steps=60),
            rng_seed=0,
        )
        draws = result.knot_samples
        assert draws is not None
        assert np.all(np.diff(draws, axis=1) >= 0)
        assert np.all((draws > 10.0) & (draws < 90.0))
        out = result.to_dict()
        assert out["knot_times"].shape == (2,)
        assert out["posterior_samples"]["death_rates"].shape[1] == model.n_death
