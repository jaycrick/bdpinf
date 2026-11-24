"""Test the constrained post-step of fit for spline rates on the non-negativity wall."""

import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from bdpinf.inference import fitting
from bdpinf.inference.constraints import Constraints
from bdpinf.inference.fitting import FitResult, fit
from bdpinf.inference.knots import fit_knots
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import BirthDeathModel, FreeKnots, SplineKind
from bdpinf.models.rate_functions import PiecewiseCubicRate, PiecewiseLinearRate
from bdpinf.simulation.age_sampling import simulate_ages_from_rates

_T = 1200.0
_EPOCHS = [400.0, 800.0, _T]
_D15 = Constraints(fixed={"birth.knot3": 0.0})
_GRID = jnp.linspace(0.0, _T, 4001)


def _data(seed: int, values: list[float], slope0: float = 0.0) -> AgeData:
    """Fifteen groups from a clamped-cubic birth rate (end slope zero)."""
    times = jnp.array(_EPOCHS)
    birth = PiecewiseCubicRate.clamped_spline(jnp.array(values), slope0, 0.0, times)
    death = PiecewiseLinearRate.continuous(jnp.full(4, 0.002), times)
    rng = np.random.default_rng(seed)
    groups = [simulate_ages_from_rates(birth, death, rng) for _ in range(15)]
    return AgeData.from_groups(
        [np.asarray(g.non_spine_ages) for g in groups],
        _T,
        spine_ages=[float(g.spine_age) for g in groups],
    )


@pytest.fixture(scope="module")
def declining() -> dict[int, AgeData]:
    """Simulate a birth rate falling to zero at T (seeds chosen to hit the wall)."""
    return {
        seed: _data(seed, [0.006, 0.004, 0.0015, 0.0], -0.000005) for seed in (0, 1, 8)
    }


@pytest.fixture(scope="module")
def dipping() -> AgeData:
    """Simulate a birth rate that dips almost to zero in the middle epoch."""
    return _data(9, [0.005, 0.0004, 0.0002, 0.005])


def _reevaluated(model: BirthDeathModel, result: FitResult, data: AgeData) -> float:
    return float(log_likelihood_rates(data, *model.rates(jnp.asarray(result.natural))))


def _min_birth(result: FitResult) -> float:
    return float(jnp.min(result.birth(_GRID)))


class TestEndSlopeWall:
    """Test an optimum on the end-slope condition w'(T-) <= 0 under D15."""

    @pytest.mark.parametrize("kind", ["cubic_clamped", "quadratic"])
    def test_wall_optimum_verified(
        self, declining: dict[int, AgeData], kind: SplineKind
    ) -> None:
        data = declining[8]
        model = BirthDeathModel.spline(_EPOCHS, kind)
        plain = fit(
            data, model, constraints=_D15, constrained=False, compute_covariance=False
        )
        result = fit(data, model, constraints=_D15)
        assert not plain.converged
        assert result.converged
        assert result.at_guard
        assert result.constrained_optimum
        assert not result.at_interior_optimum
        assert result.kkt_residual is not None
        assert result.kkt_residual <= 1e-6 * data.n_observations
        assert "birth: slope at end of epoch 2" in result.active_constraints
        assert result.log_likelihood >= plain.log_likelihood
        assert result.log_likelihood > plain.log_likelihood + 1.0
        assert np.isfinite(_reevaluated(model, result, data))
        assert np.isclose(
            _reevaluated(model, result, data), result.log_likelihood, atol=1e-8
        )
        assert _min_birth(result) >= -1e-12

    def test_clamped_end_slope_is_not_positive(
        self, declining: dict[int, AgeData]
    ) -> None:
        model = BirthDeathModel.spline(_EPOCHS, "cubic_clamped")
        result = fit(declining[1], model, constraints=_D15)
        assert result.converged
        assert result.params["birth.slopeT"] <= 0.0
        assert result.active_constraints == ("birth: slope at end of epoch 2",)
        assert result.reduced_hessian_min_eig is not None
        assert result.reduced_hessian_min_eig > 0

    def test_result_is_strictly_feasible_and_round_trips(
        self, declining: dict[int, AgeData]
    ) -> None:
        model = BirthDeathModel.spline(_EPOCHS, "cubic_clamped")
        result = fit(declining[1], model, constraints=_D15)
        # Re-evaluating from the stored dictionary gives the same finite value,
        # i.e. the result does not sit on the -1e-12 guard tolerance.
        natural = jnp.array([result.params[n] for n in model.names])
        value = float(log_likelihood_rates(declining[1], *model.rates(natural)))
        assert np.isfinite(value)
        assert value == pytest.approx(result.log_likelihood, abs=1e-8)
        assert result.params["birth.slopeT"] < 0.0

    def test_diagnostics_in_to_dict(self, declining: dict[int, AgeData]) -> None:
        model = BirthDeathModel.spline(_EPOCHS, "cubic_clamped")
        summary = fit(declining[1], model, constraints=_D15).to_dict()
        assert summary["at_guard"] is True
        assert summary["constrained_optimum"] is True
        assert summary["active_constraints"] == ["birth: slope at end of epoch 2"]
        assert summary["kkt_residual"] is not None


class TestInteriorTangency:
    """Test an optimum where an interior segment touches zero."""

    def test_tangency_optimum_verified(self, dipping: AgeData) -> None:
        model = BirthDeathModel.spline(_EPOCHS, "cubic_natural")
        plain = fit(dipping, model, constrained=False, compute_covariance=False)
        result = fit(dipping, model)
        assert result.converged
        assert result.at_guard
        assert result.constrained_optimum
        assert result.active_constraints == ("birth: interior minimum of epoch 1",)
        assert result.log_likelihood >= plain.log_likelihood
        assert np.isfinite(_reevaluated(model, result, dipping))
        assert _min_birth(result) >= -1e-12


class TestUnchanged:
    """Test that fits clear of the wall, and non-spline models, are unchanged."""

    @staticmethod
    def _forbid(monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("constrained post-step must not run")

        monkeypatch.setattr(fitting, "refine_constrained", boom)

    def test_interior_spline_fit_skips_post_step(
        self, declining: dict[int, AgeData], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = BirthDeathModel.spline(_EPOCHS, "cubic_clamped")
        data = declining[0]
        plain = fit(
            data, model, constraints=_D15, constrained=False, compute_covariance=False
        )
        assert plain.converged
        self._forbid(monkeypatch)
        result = fit(data, model, constraints=_D15)
        assert np.allclose(result.natural, plain.natural, rtol=0, atol=1e-10)
        assert result.log_likelihood == plain.log_likelihood
        assert not result.at_guard
        assert not result.constrained_optimum
        assert result.kkt_residual is None
        assert result.active_constraints == ()
        assert result.at_interior_optimum == plain.at_interior_optimum

    @pytest.mark.parametrize("family", ["linear", "constant"])
    def test_non_spline_models_never_build_the_wall(
        self,
        declining: dict[int, AgeData],
        monkeypatch: pytest.MonkeyPatch,
        family: str,
    ) -> None:
        model = (
            BirthDeathModel.piecewise_linear(_EPOCHS)
            if family == "linear"
            else BirthDeathModel.piecewise_constant(_EPOCHS)
        )

        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("no wall for this model")

        before = fit(declining[0], model)
        monkeypatch.setattr(fitting, "wall_constraints", boom)
        self._forbid(monkeypatch)
        after = fit(declining[0], model)
        assert np.array_equal(before.natural, after.natural)
        assert after.log_likelihood == before.log_likelihood
        assert not after.at_guard
        assert not after.constrained_optimum

    def test_constrained_false_reproduces_plain_fit(
        self, declining: dict[int, AgeData]
    ) -> None:
        model = BirthDeathModel.spline(_EPOCHS, "cubic_clamped")
        a = fit(
            declining[8],
            model,
            constraints=_D15,
            constrained=False,
            compute_covariance=False,
        )
        assert not a.constrained_optimum
        assert a.kkt_residual is None
        assert a.method != "SLSQP+KKT"


class TestFitResultDefaults:
    """Test that the new fields have backward-compatible defaults."""

    def test_defaults(self, declining: dict[int, AgeData]) -> None:
        result = fit(declining[0], BirthDeathModel.piecewise_constant(_EPOCHS))
        assert result.at_guard is False
        assert result.kkt_residual is None
        assert result.reduced_hessian_min_eig is None
        assert result.active_constraints == ()
        assert result.constrained_optimum is False
        summary = result.to_dict()
        assert summary["active_constraints"] == []


class TestKnotSearch:
    """Test the constrained step inside fit_knots (the final and joint fits)."""

    @pytest.mark.slow
    def test_fit_knots_clamped_spline_reports_verified_diagnostics(self) -> None:
        data = _data(1, [0.006, 0.004, 0.0015, 0.0], -0.000005)
        free = FreeKnots(2, _T, bounds=(100.0, 1100.0), min_spacing=100.0)
        model = BirthDeathModel.spline(free, "cubic_clamped")
        result = fit_knots(
            data,
            model,
            constraints=_D15,
            seed=0,
        )
        assert np.isfinite(result.log_likelihood)
        assert result.converged
        assert result.fit.converged == result.converged
        assert result.at_guard == result.fit.at_guard
        assert result.kkt_residual is None or result.kkt_residual < 1e-3
        natural = jnp.asarray(result.fit.natural)
        assert np.isfinite(float(log_likelihood_rates(data, *model.rates(natural))))

    @pytest.mark.slow
    def test_joint_candidates_are_constrained_and_profiles_plain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bdpinf.inference.knots as knots_module

        calls: list[tuple[bool, object, float]] = []
        real = knots_module.fit

        def spy(*args: object, **kwargs: object) -> FitResult:
            result = real(*args, **kwargs)  # type: ignore[arg-type]
            constraints = kwargs["constraints"]
            joint = not any(
                k.startswith("tau")
                for k in constraints.fixed  # type: ignore[attr-defined]
            )
            calls.append(
                (joint, kwargs.get("constrained", True), result.log_likelihood)
            )
            return result

        monkeypatch.setattr(knots_module, "fit", spy)
        data = _data(1, [0.006, 0.004, 0.0015, 0.0], -0.000005)
        free = FreeKnots(2, _T, bounds=(100.0, 1100.0), min_spacing=100.0)
        model = BirthDeathModel.spline(free, "cubic_clamped")
        result = fit_knots(data, model, constraints=_D15, seed=0)
        joint = [c for c in calls if c[0]]
        assert joint
        assert all(c[1] is not False for c in joint)
        assert any(c[1] is False for c in calls if not c[0])
        assert result.log_likelihood >= max(c[2] for c in joint) - 1e-9


def _year_data(weights: list[float], n: int = 200, seed: int = 3) -> AgeData:
    """Interval-censored years 100-1100 whose density follows ``weights``."""
    rng = np.random.default_rng(seed)
    grid = np.arange(100, 1100)
    density = np.interp(grid, [100, 400, 700, 1100], weights)
    early = rng.choice(grid, size=n, p=density / density.sum())
    frame = pd.DataFrame({"early": early, "late": early + rng.integers(0, 80, size=n)})
    return AgeData.from_year_ranges(
        frame["early"], frame["late"], start_year=100, observation_year=1100
    )


class TestRobustness:
    """Test the 0.8.3 fixes: non-finite gradients, pchip at_guard, retries."""

    @pytest.mark.parametrize("weights", [[3, 3, 0.1, 0.1], [1, 3, 0.2, 0.05]])
    def test_clamped_two_knots_does_not_crash(self, weights: list[float]) -> None:
        # SLSQP used to drift a log coordinate at its bound to -150, where the
        # likelihood is finite but the gradient is NaN, and nnls then raised.
        data = _year_data(weights)
        model = BirthDeathModel.spline([300.0, 600.0, 1000.0], "cubic_clamped")
        result = fit(
            data,
            model,
            constraints=Constraints(fixed={"birth.knot3": 0.0}),
            compute_covariance=False,
        )
        assert np.isfinite(result.log_likelihood)
        assert result.converged
        assert result.constrained_optimum
        assert np.isfinite(_reevaluated(model, result, data))

    def test_hard_wall_case_reaches_verification(self) -> None:
        data = _year_data([1, 3, 0.2, 0.05])
        model = BirthDeathModel.spline([400.0, 1000.0], "cubic_clamped")
        d15 = Constraints(fixed={"birth.knot2": 0.0})
        plain = fit(
            data, model, constraints=d15, constrained=False, compute_covariance=False
        )
        result = fit(data, model, constraints=d15, compute_covariance=False)
        assert result.converged
        assert result.log_likelihood > plain.log_likelihood + 10.0
        assert result.log_likelihood == pytest.approx(-1287.4848, abs=1e-3)

    @pytest.mark.parametrize("kind", ["pchip", "cubic_natural"])
    def test_interior_d15_fit_is_not_at_guard(self, kind: SplineKind) -> None:
        data = _year_data([3, 3, 0.1, 0.1])
        model = BirthDeathModel.spline([400.0, 1000.0], kind)
        result = fit(
            data,
            model,
            constraints=Constraints(fixed={"birth.knot2": 0.0}),
            compute_covariance=False,
        )
        assert not result.at_guard
        assert not result.constrained_optimum
        assert result.active_constraints == ()

    def test_post_step_exception_falls_back_to_plain(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("boom")

        data = _year_data([1, 3, 0.2, 0.05])
        model = BirthDeathModel.spline([400.0, 1000.0], "cubic_clamped")
        d15 = Constraints(fixed={"birth.knot2": 0.0})
        plain = fit(
            data, model, constraints=d15, constrained=False, compute_covariance=False
        )
        monkeypatch.setattr(fitting, "refine_constrained", boom)
        with caplog.at_level(logging.WARNING):
            result = fit(data, model, constraints=d15, compute_covariance=False)
        assert result.log_likelihood == plain.log_likelihood
        assert not result.constrained_optimum
        assert "post-step failed" in caplog.text
