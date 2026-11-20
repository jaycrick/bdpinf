"""Test the hardened fit_knots: starts, effort, neighbours, nesting, diagnostics."""

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import fit
from bdpinf.inference.knots import (
    _default_effort,
    _diagnostics,
    fit_knots,
    neighbour_placements,
    nested_start,
)
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import BirthDeathModel, FreeKnots, SplineKind
from bdpinf.models.rate_functions import PiecewiseConstantRate
from bdpinf.simulation.age_sampling import simulate_ages_from_rates

_T = 100.0


@pytest.fixture(scope="module")
def data() -> AgeData:
    """Simulate groups whose birth rate drops at 50."""
    times = jnp.array([50.0, _T])
    birth = PiecewiseConstantRate(jnp.array([0.09, 0.02]), times)
    death = PiecewiseConstantRate(jnp.array([0.01, 0.01]), times)
    rng = np.random.default_rng(3)
    groups = [simulate_ages_from_rates(birth, death, rng) for _ in range(12)]
    return AgeData.from_groups(
        [np.asarray(g.non_spine_ages) for g in groups],
        _T,
        spine_ages=[float(g.spine_age) for g in groups],
    )


def _profile(data: AgeData, model: BirthDeathModel, knots: list[float]) -> float:
    fixed = {f"tau{i}": t for i, t in enumerate(knots)}
    result = fit(
        data,
        model,
        constraints=Constraints(fixed=fixed),
        compute_covariance=False,
    )
    return result.log_likelihood


class TestDefaultEffort:
    """Test the scaling of the search effort with the number of free knots."""

    @pytest.mark.parametrize("k", [1, 2])
    def test_old_defaults_for_few_knots(self, k: int) -> None:
        assert _default_effort(k) == (64, 3, 15)

    @pytest.mark.parametrize(
        ("k", "expected"),
        [(3, (128, 3, 15)), (4, (128, 4, 30)), (5, (256, 5, 30)), (13, (512, 13, 30))],
    )
    def test_scales_with_knots(self, k: int, expected: tuple[int, int, int]) -> None:
        assert _default_effort(k) == expected

    def test_recorded_on_result(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        result = fit_knots(data, model, seed=0)
        assert (result.n_candidates, result.n_polish, result.de_maxiter) == (64, 3, 15)
        explicit = fit_knots(data, model, n_candidates=8, n_polish=2, seed=0)
        assert (explicit.n_candidates, explicit.n_polish) == (8, 2)


class TestStarts:
    """Test the extra starting placements."""

    def test_result_is_at_least_the_best_start(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(2, _T))
        starts = [[20.0, 50.0], [35.0, 75.0]]
        best = max(_profile(data, model, s) for s in starts)
        result = fit_knots(data, model, n_candidates=4, starts=starts, seed=0)
        assert result.log_likelihood >= best - 1e-9
        assert result.n_starts == 2

    def test_starts_win_when_the_search_is_poor(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(2, _T))
        start = [20.0, 50.0]
        poor = fit_knots(data, model, n_candidates=4, n_polish=1, seed=5)
        helped = fit_knots(
            data, model, n_candidates=4, n_polish=1, seed=5, starts=[start]
        )
        assert helped.log_likelihood >= poor.log_likelihood - 1e-9
        assert helped.log_likelihood >= _profile(data, model, start) - 1e-9

    @pytest.mark.slow
    def test_start_rates_warm_start(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_linear(FreeKnots(1, _T))
        lower = fit(
            data, BirthDeathModel.piecewise_linear([_T]), compute_covariance=False
        )
        natural = nested_start(model, lower, [50.0])
        result = fit_knots(data, model, starts=[[50.0]], start_rates=[natural], seed=0)
        assert result.log_likelihood >= lower.log_likelihood - 1e-6

    def test_invalid_starts_raise(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(2, _T))
        with pytest.raises(ValueError, match="free knot"):
            fit_knots(data, model, starts=[[50.0]])
        with pytest.raises(ValueError, match="outside"):
            fit_knots(data, model, starts=[[50.0, 150.0]])
        with pytest.raises(ValueError, match="start_rates"):
            fit_knots(data, model, starts=[[40.0, 60.0]], start_rates=[])


class TestNeighbourPlacements:
    """Test the neighbour placements of a fit with one knot more or fewer."""

    ranges = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0))

    def test_insertions_cover_every_gap(self) -> None:
        rows = neighbour_placements([30.0, 70.0], ranges=self.ranges)
        assert len(rows) == 3
        expected = [[15.0, 30.0, 70.0], [30.0, 50.0, 70.0], [30.0, 70.0, 85.0]]
        np.testing.assert_allclose(sorted(r.tolist() for r in rows), expected)

    def test_deletions(self) -> None:
        rows = neighbour_placements(upper=[10.0, 40.0, 60.0, 90.0], ranges=self.ranges)
        assert len(rows) == 4
        assert all(len(r) == 3 for r in rows)

    def test_spacing_drops_and_repairs(self) -> None:
        rows = neighbour_placements([30.0, 70.0], ranges=self.ranges, min_spacing=25.0)
        for row in rows:
            assert np.all(np.diff(row) >= 25.0 - 1e-9)
        crowded = neighbour_placements(
            upper=[10.0, 20.0, 60.0, 90.0], ranges=self.ranges, min_spacing=25.0
        )
        assert len(crowded) < 4

    def test_random_and_duplicates(self) -> None:
        rows = neighbour_placements([30.0, 70.0], ranges=self.ranges, n_random=5, rng=1)
        assert 3 < len(rows) <= 8
        keys = {tuple(np.round(r, 9)) for r in rows}
        assert len(keys) == len(rows)
        again = neighbour_placements(
            [30.0, 70.0], ranges=self.ranges, n_random=5, rng=1
        )
        np.testing.assert_array_equal(np.array(rows), np.array(again))

    def test_fixed_knots_and_ranges_filter(self) -> None:
        rows = neighbour_placements(
            [30.0, 70.0], ranges=self.ranges, min_spacing=10.0, fixed=[50.0]
        )
        assert all(np.all(np.abs(r - 50.0) >= 10.0 - 1e-9) for r in rows)
        narrow = neighbour_placements([30.0], ranges=[(0.0, 40.0), (60.0, 100.0)])
        assert all(r[0] <= 40.0 and r[1] >= 60.0 for r in narrow)

    def test_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="lower"):
            neighbour_placements([30.0], ranges=self.ranges)
        with pytest.raises(ValueError, match="upper"):
            neighbour_placements(upper=[30.0], ranges=self.ranges)
        with pytest.raises(ValueError, match="lower and/or upper"):
            neighbour_placements(ranges=self.ranges)


_NESTED_KINDS = ["linear", "quadratic", "cubic_natural", "cubic_clamped"]


def _model(kind: str, knots: FreeKnots) -> BirthDeathModel:
    if kind == "linear":
        return BirthDeathModel.piecewise_linear(knots)
    return BirthDeathModel.spline(knots, kind)  # type: ignore[arg-type]


class TestNestedStart:
    """Test that inserting a knot through nested_start keeps the likelihood."""

    @pytest.mark.parametrize("kind", _NESTED_KINDS)
    def test_exact_nesting(self, kind: SplineKind | str, data: AgeData) -> None:
        lower_model = _model(kind, FreeKnots(1, _T))
        lower = fit(
            data,
            lower_model,
            constraints=Constraints(fixed={"tau0": 45.0}),
            compute_covariance=False,
        )
        upper_model = _model(kind, FreeKnots(2, _T))
        natural = nested_start(upper_model, lower, [45.0, 80.0])
        birth, death = upper_model.rates(natural)
        value = float(log_likelihood_rates(data, birth, death))
        assert value == pytest.approx(lower.log_likelihood, abs=1e-6)
        # A fit with the knots fixed from that start cannot be worse.
        again = fit(
            data,
            upper_model,
            constraints=Constraints(fixed={"tau0": 45.0, "tau1": 80.0}),
            init=natural,
            compute_covariance=False,
        )
        assert again.log_likelihood >= lower.log_likelihood - 1e-8

    def test_accepts_knot_fit_result(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_linear(FreeKnots(1, _T))
        found = fit_knots(data, model, n_candidates=8, seed=0)
        upper = BirthDeathModel.piecewise_linear(FreeKnots(2, _T))
        natural = nested_start(upper, found, [*found.knot_times, 95.0])
        assert natural.shape == (len(upper.params),)


class TestDiagnostics:
    """Test the multimodality diagnostics."""

    def test_counts_distinct_placements(self) -> None:
        tried = {
            (50.0,): -100.0,
            (50.0001,): -100.001,
            (20.0,): -100.005,
            (80.0,): -105.0,
            (30.0,): -np.inf,
        }
        tol = np.array([0.1])
        count, gap = _diagnostics(tried, (50.0,), -100.0, tol)
        assert count == 2
        assert gap == pytest.approx(0.005)

    def test_no_runner_up(self) -> None:
        count, gap = _diagnostics({(50.0,): -1.0}, (50.0,), -1.0, np.array([0.1]))
        assert count == 1
        assert gap == np.inf

    def test_reported_on_result(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_constant(FreeKnots(1, _T))
        result = fit_knots(data, model, seed=0)
        assert result.n_distinct_optima >= 1
        assert result.runner_up_gap >= 0.0
