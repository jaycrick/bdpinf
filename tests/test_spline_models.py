"""Test the spline rate families: parameters, nesting, guard, counts and fits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import fit
from bdpinf.inference.knots import fit_knots
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import (
    BirthDeathModel,
    FreeKnots,
    SplineKind,
    SplineModel,
)
from bdpinf.models.rate_functions import (
    PiecewiseConstantRate,
    PiecewiseCubicRate,
    PiecewiseLinearRate,
)
from bdpinf.simulation.age_sampling import simulate_ages_from_rates

T = 1200.0
EPOCHS = [400.0, 800.0, T]  # K = 3 epochs
K = 3
KINDS: list[SplineKind] = [
    "quadratic",
    "cubic_natural",
    "cubic_clamped",
    "pchip",
    "cubic_hermite",
]
# Number of spline parameters per kind, as a function of K (the plan's table).
SPLINE_COUNT = {
    "quadratic": K + 2,
    "cubic_natural": K + 1,
    "cubic_clamped": K + 3,
    "pchip": K + 1,
    "cubic_hermite": 2 * K + 2,
}
LINEAR_COUNT = K + 1
GRID = np.linspace(0.0, T, 241)


@pytest.fixture(scope="module")
def data() -> AgeData:
    """Small simulated data set with a smooth, positive birth rate."""
    times = jnp.array(EPOCHS)
    birth = PiecewiseCubicRate.natural_spline(
        jnp.array([0.004, 0.006, 0.003, 0.002]), times
    )
    death = PiecewiseLinearRate.continuous(jnp.full(4, 0.002), times)
    rng = np.random.default_rng(1)
    groups = [simulate_ages_from_rates(birth, death, rng) for _ in range(25)]
    return AgeData.from_groups(
        [np.asarray(g.non_spine_ages) for g in groups],
        T,
        spine_ages=[float(g.spine_age) for g in groups],
    )


def _ll(model: BirthDeathModel, natural: jax.Array, data: AgeData) -> float:
    return float(log_likelihood_rates(data, *model.rates(natural)))


class TestSplineModelLayout:
    """Test parameter names, counts and the scaling of slope parameters."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_parameter_count_matches_table(self, kind: SplineKind) -> None:
        model = SplineModel(EPOCHS, kind)
        assert len(model.params) == SPLINE_COUNT[kind]

    def test_names_and_transforms(self) -> None:
        names = {k: [p.name for p in SplineModel(EPOCHS, k).params] for k in KINDS}
        knots = ["knot0", "knot1", "knot2", "knot3"]
        assert names["quadratic"] == [*knots, "slope0"]
        assert names["cubic_natural"] == knots
        assert names["cubic_clamped"] == [*knots, "slope0", "slopeT"]
        assert names["pchip"] == knots
        assert names["cubic_hermite"] == [*knots, *(f"slope{i}" for i in range(4))]
        params = SplineModel(EPOCHS, "cubic_hermite", min_value=0.1).params
        assert {p.transform for p in params[:4]} == {"log"}
        assert {p.lower for p in params[:4]} == {0.1}
        assert {p.transform for p in params[4:]} == {"identity"}

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            SplineModel(EPOCHS, "cubic")  # type: ignore[arg-type]

    def test_slopes_are_scaled_by_observation_time(self) -> None:
        model = SplineModel(EPOCHS, "cubic_hermite")
        slopes = np.array([0.001, -0.002, 0.0, 0.0005])
        values = jnp.concatenate([jnp.full(4, 0.5), jnp.asarray(slopes * T)])
        rate = model.build(values)
        np.testing.assert_allclose(rate.start_slopes, slopes[:-1])
        np.testing.assert_allclose(rate.end_slopes, slopes[1:])

    def test_clamped_end_slopes_are_scaled(self) -> None:
        model = SplineModel(EPOCHS, "cubic_clamped")
        values = jnp.array([0.5, 0.6, 0.4, 0.5, 0.6, -0.3])
        rate = model.build(values)
        assert float(rate.start_slopes[0]) == pytest.approx(0.6 / T)
        assert float(rate.end_slopes[-1]) == pytest.approx(-0.3 / T)

    @pytest.mark.parametrize("kind", KINDS)
    def test_constant_values_give_a_flat_rate(self, kind: SplineKind) -> None:
        model = SplineModel(EPOCHS, kind)
        rate = model.build(model.constant_values(0.3))
        np.testing.assert_allclose(rate(jnp.asarray(GRID)), 0.3, atol=1e-14)

    @pytest.mark.parametrize("kind", ["quadratic", "cubic_natural", "cubic_clamped"])
    def test_values_from_rate_reproduces_a_spline_on_refined_knots(
        self, kind: SplineKind
    ) -> None:
        coarse = SplineModel([500.0, T], kind)
        n = len(coarse.params)
        values = np.array([0.004, 0.006, 0.003, 0.002, -0.001, 0.001, 0.0005][:n])
        rate = coarse.build(jnp.asarray(values))
        fine = SplineModel([200.0, 500.0, 900.0, T], kind)
        refined = fine.build(fine.values_from_rate(rate))
        np.testing.assert_allclose(
            refined(jnp.asarray(GRID)), rate(jnp.asarray(GRID)), atol=1e-10
        )
        np.testing.assert_allclose(
            refined.integral(0.0, jnp.asarray(GRID)),
            rate.integral(0.0, jnp.asarray(GRID)),
            atol=1e-10,
        )

    def test_values_from_rate_of_a_linear_rate_gives_matching_slopes(self) -> None:
        linear = PiecewiseLinearRate.continuous(jnp.array([1.0, 2.0, 1.0]), [500.0, T])
        model = SplineModel([500.0, T], "cubic_hermite")
        values = model.values_from_rate(linear)
        np.testing.assert_allclose(values[:3], [1.0, 2.0, 1.0])
        np.testing.assert_allclose(values[3:] / T, [1 / 500, 1 / 500, -1 / 700])

    def test_values_from_rate_of_a_generic_rate_uses_differences(self) -> None:
        flat = PiecewiseConstantRate(jnp.array([0.5, 0.5]), jnp.array([500.0, T]))
        values = SplineModel([500.0, T], "cubic_clamped").values_from_rate(flat)
        np.testing.assert_allclose(values[:3], 0.5)
        np.testing.assert_allclose(values[3:], 0.0, atol=1e-9)

    def test_min_value_lifts_warm_starts(self) -> None:
        model = SplineModel(EPOCHS, "cubic_natural", min_value=0.1)
        zero = PiecewiseConstantRate(jnp.zeros(1), jnp.array([T]))
        assert np.all(np.asarray(model.values_from_rate(zero)) > 0.1)


class TestBirthDeathSpline:
    """Test the BirthDeathModel.spline constructor."""

    def test_default_is_spline_birth_and_linear_death(self) -> None:
        model = BirthDeathModel.spline(EPOCHS)
        assert isinstance(model.birth, SplineModel)
        assert model.birth.kind == "cubic_natural"
        assert model.n_death == LINEAR_COUNT
        assert isinstance(
            model.rates(model.constant_natural(0.01, 0.01))[1], PiecewiseLinearRate
        )

    def test_minimum_values_are_applied(self) -> None:
        model = BirthDeathModel.spline(
            EPOCHS, "pchip", "quadratic", min_birth=0.1, min_death=0.2
        )
        assert {p.lower for p in model.params[: model.n_birth]} == {0.1}
        assert {p.lower for p in model.params[model.n_birth : -1]} == {0.2}

    def test_same_kind_models_can_swap_and_mixed_cannot(self) -> None:
        same = BirthDeathModel.spline(EPOCHS, "cubic_clamped", "cubic_clamped")
        natural = same.constant_natural(0.01, 0.02)
        swapped = same.swapped(natural)
        assert swapped is not None
        np.testing.assert_allclose(swapped[: same.n_birth], natural[same.n_birth :])
        mixed = BirthDeathModel.spline(EPOCHS, "cubic_clamped")
        assert mixed.swapped(mixed.constant_natural(0.01, 0.02)) is None

    def test_natural_from_rates_with_free_knots(self) -> None:
        model = BirthDeathModel.spline(FreeKnots(2, T), "cubic_natural")
        birth = PiecewiseCubicRate.natural_spline(
            jnp.array([1.0, 2.0, 1.5]), jnp.array([600.0, T])
        )
        death = PiecewiseConstantRate(jnp.array([0.1]), jnp.array([T]))
        natural = model.natural_from_rates(birth, death, [300.0, 700.0])
        assert natural.shape[0] == len(model.params)
        np.testing.assert_allclose(natural[-2:], [300.0, 700.0])
        rebuilt = model.rates(natural)[0]
        np.testing.assert_allclose(
            rebuilt(jnp.array([300.0, 700.0])), natural[1:3], atol=1e-12
        )


class TestLikelihoodNesting:
    """Test the likelihood against simpler models it contains."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_equal_knot_values_give_the_constant_rate_likelihood(
        self, kind: SplineKind, data: AgeData
    ) -> None:
        model = BirthDeathModel.spline(EPOCHS, kind, kind)
        constant = _ll(model, model.constant_natural(0.004, 0.002), data)
        reference = float(
            log_likelihood_rates(
                data,
                PiecewiseConstantRate(jnp.array([0.004]), jnp.array([T])),
                PiecewiseConstantRate(jnp.array([0.002]), jnp.array([T])),
            )
        )
        assert constant == pytest.approx(reference, rel=1e-8)

    def test_hermite_with_linear_slopes_is_the_linear_likelihood(
        self, data: AgeData
    ) -> None:
        death = PiecewiseLinearRate.continuous(jnp.full(4, 0.002), jnp.array(EPOCHS))
        model = BirthDeathModel.spline(EPOCHS, "cubic_hermite")
        line = np.interp([0.0, *EPOCHS], [0.0, T], [0.004, 0.001])
        natural = jnp.concatenate(
            [jnp.asarray(line), jnp.full(4, (0.001 - 0.004)), jnp.full(4, 0.002)]
        )
        straight = PiecewiseLinearRate.continuous(jnp.asarray(line), jnp.array(EPOCHS))
        expected = float(log_likelihood_rates(data, straight, death))
        assert _ll(model, natural, data) == pytest.approx(expected, abs=1e-8)

    def test_hermite_with_per_epoch_linear_slopes_is_the_linear_likelihood(
        self, data: AgeData
    ) -> None:
        values = np.array([0.004, 0.005, 0.003, 0.002])
        linear = PiecewiseLinearRate.continuous(jnp.asarray(values), jnp.array(EPOCHS))
        death = PiecewiseLinearRate.continuous(jnp.full(4, 0.002), jnp.array(EPOCHS))
        slope = np.diff(values) / np.diff([0.0, *EPOCHS])
        rate = PiecewiseCubicRate(
            jnp.asarray(values[:-1]),
            jnp.asarray(values[1:]),
            jnp.asarray(slope),
            jnp.asarray(slope),
            jnp.array(EPOCHS),
        )
        reference = float(log_likelihood_rates(data, linear, death))
        assert float(log_likelihood_rates(data, rate, death)) == pytest.approx(
            reference, abs=1e-8
        )


class TestNonNegativityGuard:
    """Test the -inf guard and the gradient near it."""

    def _bad(self) -> tuple[BirthDeathModel, jax.Array]:
        model = BirthDeathModel.spline(EPOCHS, "cubic_hermite")
        natural = jnp.array(
            [0.004, 0.0001, 0.006, 0.002, -0.5, 0.5, -0.5, 0.5, *([0.002] * 4)]
        )
        return model, natural

    def test_negative_rate_gives_minus_infinity(self, data: AgeData) -> None:
        model, natural = self._bad()
        assert float(model.rates(natural)[0].min_values().min()) < 0.0
        assert _ll(model, natural, data) == -np.inf

    def test_gradient_and_hessian_stay_finite_when_infeasible(
        self, data: AgeData
    ) -> None:
        model, natural = self._bad()

        def objective(x: jax.Array) -> jax.Array:
            return log_likelihood_rates(data, *model.rates(x))

        assert np.all(np.isfinite(jax.grad(objective)(natural)))
        assert np.all(np.isfinite(jax.jit(jax.hessian(objective))(natural)))

    def test_guard_applies_to_range_data_with_finite_gradient(self) -> None:
        model, natural = self._bad()
        lower = np.array([10.0, 100.0, 300.0, 600.0, 900.0])
        ranges = AgeData.from_ranges(lower, lower + 50.0, T, population_size=6)

        def objective(x: jax.Array) -> jax.Array:
            return log_likelihood_rates(ranges, *model.rates(x))

        assert float(objective(natural)) == -np.inf
        assert np.all(np.isfinite(jax.grad(objective)(natural)))

    def test_guard_applies_under_jit(self, data: AgeData) -> None:
        model, natural = self._bad()
        value = jax.jit(lambda x: log_likelihood_rates(data, *model.rates(x)))(natural)
        assert float(value) == -np.inf

    def test_gradient_is_finite_just_inside_the_boundary(self, data: AgeData) -> None:
        model = BirthDeathModel.spline(EPOCHS, "cubic_hermite")
        feasible = jnp.array(
            [0.004, 0.003, 0.006, 0.002, 0.0, 0.0, 0.0, 0.0, *([0.002] * 4)]
        )
        bad = feasible.at[4:8].set(jnp.array([-0.5, 0.5, -0.5, 0.5]))

        def lowest(x: jax.Array) -> float:
            return float(model.rates(x)[0].min_values().min())

        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if lowest(feasible + mid * (bad - feasible)) > 0:
                lo = mid
            else:
                hi = mid
        inside = feasible + lo * (bad - feasible)
        assert 0.0 < lowest(inside) < 1e-9
        grad = jax.grad(lambda x: log_likelihood_rates(data, *model.rates(x)))(inside)
        assert np.all(np.isfinite(grad))

    def test_other_families_are_not_guarded(self, data: AgeData) -> None:
        model = BirthDeathModel.piecewise_linear(EPOCHS)
        assert np.isfinite(_ll(model, model.constant_natural(0.004, 0.002), data))

    def test_pchip_is_always_feasible(self, data: AgeData) -> None:
        model = BirthDeathModel.spline(EPOCHS, "pchip")
        rng = np.random.default_rng(0)
        for _ in range(10):
            natural = jnp.asarray(rng.uniform(1e-4, 8e-3, len(model.params)))
            assert np.isfinite(_ll(model, natural, data))


def _lowest_birth(model: BirthDeathModel, natural: np.ndarray) -> float:
    return float(model.rates(jnp.asarray(natural))[0].min_values().min())


class TestFitting:
    """Test that the Newton fit respects the guard."""

    @pytest.mark.parametrize("kind", ["cubic_natural", "cubic_hermite"])
    def test_fit_from_next_to_the_boundary_reaches_a_feasible_optimum(
        self, kind: SplineKind, data: AgeData
    ) -> None:
        model = BirthDeathModel.spline(EPOCHS, kind)
        reference = fit(data, model)
        assert reference.success
        natural = np.asarray(reference.natural)
        bad = natural.copy()
        bad[:4] = [0.02, 1e-6, 0.02, 1e-6]
        if kind == "cubic_hermite":
            bad[4:8] = [0.5, -0.5, 0.5, -0.5]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _lowest_birth(model, natural + mid * (bad - natural)) > 0:
                lo = mid
            else:
                hi = mid
        start = natural + lo * (bad - natural)
        assert 0.0 < _lowest_birth(model, start) < 1e-8
        result = fit(data, model, init=start, n_starts=1)
        assert result.success
        assert _lowest_birth(model, np.asarray(result.natural)) > 0.0
        assert result.log_likelihood == pytest.approx(
            reference.log_likelihood, abs=1e-4
        )

    def test_fixed_end_value_of_zero_works(self, data: AgeData) -> None:
        model = BirthDeathModel.spline(EPOCHS, "cubic_clamped")
        constraints = Constraints(fixed={"birth.knot3": 0.0})
        result = fit(data, model, constraints=constraints)
        natural = np.asarray(result.natural)
        assert natural[3] == 0.0
        assert np.isfinite(result.log_likelihood)
        assert _lowest_birth(model, natural) >= -1e-12

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", KINDS)
    def test_every_kind_fits(self, kind: SplineKind, data: AgeData) -> None:
        result = fit(data, BirthDeathModel.spline(EPOCHS, kind))
        assert result.success
        assert np.isfinite(result.log_likelihood)

    @pytest.mark.parametrize(
        "kind",
        [
            pytest.param(
                kind, marks=[] if kind == "cubic_natural" else pytest.mark.slow
            )
            for kind in KINDS
        ],
    )
    def test_fit_knots_runs_for_every_kind(
        self, kind: SplineKind, data: AgeData
    ) -> None:
        model = BirthDeathModel.spline(FreeKnots(1, T, min_spacing=100.0), kind)
        result = fit_knots(data, model, n_candidates=8)
        assert np.isfinite(result.fit.log_likelihood)
        assert 0.0 < result.fit.knot_times[0] < T  # type: ignore[index]
        assert _lowest_birth(model, np.asarray(result.fit.natural)) >= -1e-12


class TestFreeParameterCount:
    """Test BirthDeathModel.n_free against hand counts."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_unconstrained_count(self, kind: SplineKind) -> None:
        model = BirthDeathModel.spline(EPOCHS, kind)
        assert model.n_free() == SPLINE_COUNT[kind] + LINEAR_COUNT
        assert model.n_free(Constraints()) == model.n_free()

    @pytest.mark.parametrize("kind", KINDS)
    def test_fixed_end_value_removes_one(self, kind: SplineKind) -> None:
        model = BirthDeathModel.spline(EPOCHS, kind)
        constraints = Constraints(fixed={f"birth.knot{K}": 0.0})
        assert model.n_free(constraints) == SPLINE_COUNT[kind] + LINEAR_COUNT - 1

    @pytest.mark.parametrize("kind", KINDS)
    def test_death_groups_and_fixed_values(self, kind: SplineKind) -> None:
        model = BirthDeathModel.spline(EPOCHS, kind)
        constraints = Constraints(
            fixed={f"birth.knot{K}": 0.0},
            equal=[["death.knot0", "death.knot1"], ["death.knot2", "death.knot3"]],
        )
        assert model.n_free(constraints) == SPLINE_COUNT[kind] + LINEAR_COUNT - 3

    def test_slope_parameters_can_be_tied_and_fixed(self) -> None:
        model = BirthDeathModel.spline(EPOCHS, "cubic_hermite")
        constraints = Constraints(
            fixed={"birth.slope0": 0.0},
            equal=[["birth.slope1", "birth.slope2", "birth.slope3"]],
        )
        assert model.n_free(constraints) == (2 * K + 2) + LINEAR_COUNT - 1 - 2

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("n_knots", [1, 2, 3])
    def test_free_knots_are_counted(self, kind: SplineKind, n_knots: int) -> None:
        epochs = n_knots + 1
        model = BirthDeathModel.spline(FreeKnots(n_knots, T), kind)
        spline = {
            "quadratic": epochs + 2,
            "cubic_natural": epochs + 1,
            "cubic_clamped": epochs + 3,
            "pchip": epochs + 1,
            "cubic_hermite": 2 * epochs + 2,
        }[kind]
        assert model.n_free() == spline + (epochs + 1) + n_knots
        constraints = Constraints(fixed={f"birth.knot{epochs}": 0.0, "tau0": 100.0})
        assert model.n_free(constraints) == spline + (epochs + 1) + n_knots - 2

    def test_other_families(self) -> None:
        model = BirthDeathModel.piecewise_constant(EPOCHS)
        assert model.n_free() == 2 * K
        linear = BirthDeathModel.piecewise_linear(EPOCHS)
        assert linear.n_free() == 2 * (K + 1)

    def test_matches_the_compiled_constraint_map(self) -> None:
        model = BirthDeathModel.spline(EPOCHS, "cubic_clamped")
        constraints = Constraints(fixed={"birth.knot3": 0.0})
        assert model.n_free(constraints) == constraints.compile(model.params).n_free


class TestQuadraticValidationUnderTracing:
    """Test that the quadratic spline is guarded rather than raising when traced."""

    def test_traced_dip_is_minus_infinity(self, data: AgeData) -> None:
        model = BirthDeathModel.spline(EPOCHS, "quadratic")
        natural = jnp.array([0.004, 1e-6, 0.004, 1e-6, 0.0, *([0.002] * 4)])
        # Concretely the quadratic constructor rejects the dip ...
        with pytest.raises(ValueError, match="non-negative"):
            model.rates(natural)
        # ... and under jit the likelihood guard returns -inf.
        value = jax.jit(lambda x: log_likelihood_rates(data, *model.rates(x)))(natural)
        assert float(value) == -np.inf
