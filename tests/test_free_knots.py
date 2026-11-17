"""Test free knot times: the spec, the interval transform, MLE and Bayesian fits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.bayesian import BayesianMCMCConfig, fit_bayesian
from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import fit
from bdpinf.inference.knots import _candidates, _run_methods, _Search, fit_knots
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
        assert result.method == "brent"
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
        result = fit_knots(data, model, n_candidates=16, seed=0)
        assert result.method == "grid-newton"
        assert result.knot_times.shape == (2,)
        assert result.knot_times[0] < result.knot_times[1]

    def test_requires_free_knots_and_free_constraints(self, data: AgeData) -> None:
        with pytest.raises(ValueError, match="free knots"):
            fit_knots(data, BirthDeathModel.piecewise_constant([50.0, _T]))
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        with pytest.raises(ValueError, match="Knot times"):
            fit_knots(data, model, constraints=Constraints(fixed={"tau0": 50.0}))


class TestKnotMethods:
    """Test the default and thorough searches of fit_knots."""

    @pytest.mark.slow
    def test_thorough_runs_three_methods_and_keeps_the_best(
        self, data: AgeData
    ) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        default = fit_knots(data, model, seed=0)
        serial = fit_knots(data, model, thorough=True, workers=1, seed=0)
        parallel = fit_knots(data, model, thorough=True, workers=3, seed=0)
        names = ["brent", "de-profile", "grid-newton"]
        assert list(serial.method_results) == names
        assert list(parallel.method_results) == names
        top = max(r.log_likelihood for r in serial.method_results.values())
        assert serial.method_results[serial.method].log_likelihood == top
        assert serial.log_likelihood >= default.log_likelihood - 1e-6
        assert parallel.method == serial.method
        assert parallel.log_likelihood == pytest.approx(serial.log_likelihood, abs=1e-4)

    def test_fixed_knot_is_left_out_of_the_search(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=[(10.0, 80.0), (80.0, 99.0)])
        )
        result = fit_knots(
            data, model, constraints=Constraints(fixed={"tau1": 90.0}), seed=0
        )
        assert result.method == "brent"
        assert result.knot_times[1] == pytest.approx(90.0)

    def test_two_free_knots_default_to_grid_newton(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=[(10.0, 80.0), (80.0, 99.0)])
        )
        result = fit_knots(data, model, n_candidates=16, seed=0)
        assert result.method == "grid-newton"
        assert result.knot_times[0] < result.knot_times[1]


class TestJointFit:
    """Test fit() and fit_bayesian() with free knots."""

    @pytest.mark.slow
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

    @pytest.mark.slow
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


def _gaps(knots: np.ndarray) -> np.ndarray:
    return np.diff(np.sort(knots))


class TestMinSpacing:
    """Test the minimum distance between knots."""

    def test_default_is_zero_and_validated(self) -> None:
        assert FreeKnots(2, _T).min_spacing == 0.0
        with pytest.raises(ValueError, match="min_spacing"):
            FreeKnots(2, _T, min_spacing=-1.0)
        with pytest.raises(ValueError, match="do not fit"):
            FreeKnots(3, _T, bounds=(10.0, 40.0), min_spacing=20.0)
        with pytest.raises(ValueError, match="min_spacing"):
            FreeKnots(2, _T, initial=[40.0, 45.0], min_spacing=20.0)

    def test_default_initial_is_nudged_apart(self) -> None:
        knots = FreeKnots(3, _T, bounds=(40.0, 60.0), min_spacing=10.0)
        assert np.all(_gaps(np.array(knots.initial_times)) >= 10.0 - 1e-9)
        per_knot = FreeKnots(
            2, _T, bounds=[(40.0, 50.0), (50.0, 60.0)], min_spacing=12.0
        )
        assert np.all(_gaps(np.array(per_knot.initial_times)) >= 12.0 - 1e-9)

    def test_candidates_are_valid_and_redrawn(self) -> None:
        ranges = [(0.0, 100.0)] * 3
        rows = _candidates(ranges, 16, 0, min_spacing=45.0)
        assert len(rows) >= 1
        assert np.all(np.diff(rows, axis=1) >= 45.0 - 1e-6)
        assert rows.min() >= 0.0
        assert rows.max() <= 100.0
        with pytest.raises(ValueError, match="No placement"):
            _candidates([(40.0, 60.0)], 16, 0, min_spacing=30.0, fixed=np.array([50.0]))

    @pytest.mark.slow
    @pytest.mark.parametrize("n_knots", [2, 3])
    def test_grid_newton_respects_spacing(self, data: AgeData, n_knots: int) -> None:
        spacing = 25.0 if n_knots == 2 else 18.0
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(n_knots, _T, bounds=(5.0, 95.0), min_spacing=spacing)
        )
        result = fit_knots(data, model, n_candidates=16, seed=0)
        assert np.all(_gaps(result.knot_times) >= spacing - 1e-6)
        for found in result.method_results.values():
            assert np.all(_gaps(found.knot_times) >= spacing - 1e-6)

    def test_binding_spacing(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=(5.0, 95.0), min_spacing=60.0)
        )
        result = fit_knots(data, model, n_candidates=16, seed=1)
        assert np.all(_gaps(result.knot_times) >= 60.0 - 1e-6)

    @pytest.mark.slow
    def test_thorough_methods_respect_spacing(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=(5.0, 95.0), min_spacing=30.0)
        )
        result = fit_knots(data, model, thorough=True, workers=1, n_candidates=16)
        assert set(result.method_results) == {"brent", "de-profile", "grid-newton"}
        for found in result.method_results.values():
            assert np.all(_gaps(found.knot_times) >= 30.0 - 1e-6)
        assert np.all(_gaps(result.knot_times) >= 30.0 - 1e-6)

    def test_fixed_knot_spacing_is_enforced(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=[(10.0, 80.0), (80.0, 99.0)], min_spacing=15.0)
        )
        result = fit_knots(
            data, model, constraints=Constraints(fixed={"tau1": 90.0}), seed=0
        )
        assert (
            abs(result.knot_times[0] - 90.0) >= 15.0 - 1e-6
            or abs(result.knot_times[1] - 90.0) >= 15.0 - 1e-6
        )
        assert np.all(_gaps(result.knot_times) >= 15.0 - 1e-6)
        with pytest.raises(ValueError, match="closer"):
            fit_knots(
                data,
                BirthDeathModel.piecewise_constant(FreeKnots(3, _T, min_spacing=15.0)),
                constraints=Constraints(fixed={"tau0": 50.0, "tau1": 60.0}),
            )

    def test_zero_spacing_matches_default(self, data: AgeData) -> None:
        plain = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        zero = BirthDeathModel.piecewise_constant(FreeKnots(1, _T, min_spacing=0.0))
        first = fit_knots(data, plain, seed=0)
        second = fit_knots(data, zero, seed=0)
        assert first.log_likelihood == second.log_likelihood
        np.testing.assert_array_equal(first.knot_times, second.knot_times)


# Values recorded from the pre-refactor implementation of ``_run_methods``:
# (knot_times, log_likelihood, n_fits) per method, for ``_simulate(10, 3)``,
# ``n_candidates=8`` and ``seed=0``.  The "all" entries run the three methods in
# one call, so later methods reuse earlier fits and warm starts.
_PINNED: dict[str, dict[str, tuple[list[float], float, int]]] = {
    "one|brent": {"brent": ([49.68474595362213], -2286.9352445255236, 16)},
    "one|de-profile": {"de-profile": ([49.985008724337895], -2286.5643459072053, 62)},
    "one|grid-newton": {"grid-newton": ([49.359355717985096], -2296.371629361889, 8)},
    "one|all": {
        "brent": ([49.68474595362213], -2286.9352445255236, 16),
        "de-profile": ([49.985008724337895], -2286.5643458600493, 62),
        "grid-newton": ([49.3593557179851], -2296.3717228883947, 8),
    },
    "two|brent": {
        "brent": ([20.109787866068245, 49.57838475202295], -2280.832578092461, 67)
    },
    "two|de-profile": {
        "de-profile": (
            [19.951116533402843, 50.023326849841496],
            -2278.173414235784,
            237,
        )
    },
    "two|grid-newton": {
        "grid-newton": ([24.54478628, 49.54478628], -2284.5064624201955, 8)
    },
    "two|all": {
        "brent": ([20.109787866068245, 49.57838475202295], -2280.832578092461, 67),
        "de-profile": (
            [19.951116533402843, 50.023326849841496],
            -2278.1734142357845,
            237,
        ),
        "grid-newton": ([49.64605434219583, 86.44230429003188], -2289.233440098662, 0),
    },
}
_ALL = ("brent", "de-profile", "grid-newton")


def _pinned_job(case: str) -> _Search:
    if case == "one":
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
    else:
        model = BirthDeathModel.piecewise_constant(
            FreeKnots(2, _T, bounds=(5.0, 95.0), min_spacing=25.0)
        )
    return _Search(
        _simulate(10, 3),
        model,
        Constraints(),
        list(range(model.n_knots)),
        8,
        0,
        None,
        {},
    )


class TestRunMethodsPinned:
    """Pin the knot searches (recorded before they were refactored)."""

    @pytest.mark.parametrize("case", ["one", "two"])
    @pytest.mark.parametrize("which", ["brent", "de-profile", "grid-newton", "all"])
    def test_method_results_are_unchanged(self, case: str, which: str) -> None:
        names = _ALL if which == "all" else (which,)
        found = _run_methods(names, _pinned_job(case))
        expected = _PINNED[f"{case}|{which}"]
        assert list(found) == list(expected)
        for name, (result, _) in found.items():
            times, value, n_fits = expected[name]
            np.testing.assert_allclose(result.knot_times, times, rtol=1e-9)
            assert result.log_likelihood == pytest.approx(value, rel=1e-9)
            assert result.n_fits == n_fits
