"""Test the parametric rate families and the birth-death model wrapper."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import (
    BirthDeathModel,
    FreeKnots,
    PiecewiseConstantModel,
    PiecewiseLinearModel,
)
from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate

_TIMES = (1.0, 2.5)


class TestPiecewiseConstant:
    """Test the piecewise-constant family."""

    def test_params_and_build(self) -> None:
        family = PiecewiseConstantModel(_TIMES)
        assert [p.name for p in family.params] == ["v0", "v1"]
        assert family.end_time == 2.5
        rate = family.build(jnp.array([0.4, 0.9]))
        assert isinstance(rate, PiecewiseConstantRate)
        np.testing.assert_allclose(rate(jnp.array([0.5, 2.0])), [0.4, 0.9])

    def test_constant_values_and_values_from_rate(self) -> None:
        family = PiecewiseConstantModel(_TIMES)
        np.testing.assert_allclose(family.constant_values(0.3), [0.3, 0.3])
        rate = family.build(jnp.array([0.4, 0.9]))
        np.testing.assert_allclose(family.values_from_rate(rate), [0.4, 0.9])

    @pytest.mark.parametrize("times", [[], [0.0, 1.0], [2.0, 1.0], [1.0, 1.0]])
    def test_rejects_bad_epochs(self, times: list[float]) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            PiecewiseConstantModel(times)


class TestPiecewiseLinear:
    """Test the continuous and jump-form piecewise-linear families."""

    def test_continuous_parameters(self) -> None:
        family = PiecewiseLinearModel(_TIMES)
        assert [p.name for p in family.params] == ["knot0", "knot1", "knot2"]
        rate = family.build(jnp.array([1.0, 2.0, 0.5]))
        assert isinstance(rate, PiecewiseLinearRate)
        np.testing.assert_allclose(
            rate(jnp.array([0.0, 1.0, 2.5])), [1.0, 2.0, 0.5], atol=1e-12
        )

    def test_jump_parameters(self) -> None:
        family = PiecewiseLinearModel(_TIMES, continuous=False)
        assert [p.name for p in family.params] == ["start0", "start1", "end0", "end1"]
        rate = family.build(jnp.array([1.0, 3.0, 2.0, 0.5]))
        np.testing.assert_allclose(rate(jnp.array([0.0, 1.0])), [1.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(rate(jnp.array([1.0 + 1e-9])), [3.0], atol=1e-6)

    def test_values_from_rate_round_trip(self) -> None:
        for continuous, values in (
            (True, [1.0, 2.0, 0.5]),
            (False, [1.0, 3.0, 2.0, 0.5]),
        ):
            family = PiecewiseLinearModel(_TIMES, continuous=continuous)
            rate = family.build(jnp.array(values))
            np.testing.assert_allclose(family.values_from_rate(rate), values, rtol=1e-6)

    def test_warm_start_from_piecewise_constant(self) -> None:
        constant = PiecewiseConstantModel(_TIMES).build(jnp.array([1.0, 3.0]))
        knots = PiecewiseLinearModel(_TIMES).values_from_rate(constant)
        np.testing.assert_allclose(knots, [1.0, 2.0, 3.0], rtol=1e-6)

    def test_flat_groups_only_for_jump_form(self) -> None:
        assert PiecewiseLinearModel(_TIMES, continuous=False).flat_groups("birth.") == [
            ["birth.start0", "birth.end0"],
            ["birth.start1", "birth.end1"],
        ]
        with pytest.raises(ValueError, match="continuous=False"):
            PiecewiseLinearModel(_TIMES).flat_groups()

    def test_flat_jump_model_is_piecewise_constant(self) -> None:
        """Tying start and end values makes a jump-form linear model constant."""
        pc = BirthDeathModel.piecewise_constant(_TIMES)
        pl = BirthDeathModel.piecewise_linear(_TIMES, continuous=False)
        constraints = Constraints(
            equal=pl.birth.flat_groups("birth.") + pl.death.flat_groups("death.")  # pyright: ignore[reportAttributeAccessIssue]
        )
        compiled = constraints.compile(pl.params)
        assert compiled.n_free == 4
        free = jnp.log(jnp.array([0.7, 1.1, 0.4, 0.9]))
        pl_rates = pl.rates(compiled.expand(free))
        pc_rates = pc.rates(jnp.exp(free))
        data = AgeData.from_groups([jnp.array([0.2, 0.9, 1.4, 2.0, 2.2])], 2.5)
        np.testing.assert_allclose(
            log_likelihood_rates(data, *pl_rates),
            log_likelihood_rates(data, *pc_rates),
            rtol=1e-9,
        )


class TestBirthDeathModel:
    """Test the combined model."""

    def test_names_and_layout(self) -> None:
        model = BirthDeathModel.piecewise_constant(_TIMES)
        assert model.names == ("birth.v0", "birth.v1", "death.v0", "death.v1")
        assert model.birth_names == ["birth.v0", "birth.v1"]
        assert model.death_names == ["death.v0", "death.v1"]
        assert model.n_birth == 2
        assert model.end_time == 2.5

    def test_rates_split_birth_and_death(self) -> None:
        model = BirthDeathModel.piecewise_constant(_TIMES)
        birth, death = model.rates(jnp.array([0.1, 0.2, 0.3, 0.4]))
        np.testing.assert_allclose(birth(jnp.array([0.5, 2.0])), [0.1, 0.2])
        np.testing.assert_allclose(death(jnp.array([0.5, 2.0])), [0.3, 0.4])

    def test_swapped_and_constant_natural(self) -> None:
        model = BirthDeathModel.piecewise_constant(_TIMES)
        natural = model.constant_natural(1.0, 2.0)
        np.testing.assert_allclose(natural, [1.0, 1.0, 2.0, 2.0])
        np.testing.assert_allclose(model.swapped(natural), [2.0, 2.0, 1.0, 1.0])

    def test_swapped_is_none_for_mismatched_layouts(self) -> None:
        mixed = BirthDeathModel(
            PiecewiseConstantModel(_TIMES), PiecewiseLinearModel(_TIMES)
        )
        assert mixed.swapped(jnp.ones(5)) is None

    def test_natural_from_rates(self) -> None:
        model = BirthDeathModel.piecewise_linear(_TIMES)
        birth = PiecewiseConstantModel(_TIMES).build(jnp.array([1.0, 3.0]))
        death = PiecewiseConstantModel(_TIMES).build(jnp.array([0.5, 0.5]))
        natural = model.natural_from_rates(birth, death)
        np.testing.assert_allclose(natural, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5], rtol=1e-6)

    def test_rejects_mismatched_windows(self) -> None:
        with pytest.raises(ValueError, match="share the observation time"):
            BirthDeathModel(
                PiecewiseConstantModel([1.0]), PiecewiseConstantModel([2.0])
            )


# Recorded from the pre-refactor ``FreeKnots.__post_init__``.
_BAD_KNOTS: dict[str, tuple[dict[str, Any], str]] = {
    "zero_knots": (dict(n_knots=0, end_time=100.0), "n_knots must be at least 1"),
    "zero_knots_bad_end": (
        dict(n_knots=0, end_time=-1.0),
        "n_knots must be at least 1",
    ),
    "end_zero": (dict(n_knots=1, end_time=0.0), "end_time must be positive"),
    "end_negative": (dict(n_knots=1, end_time=-5.0), "end_time must be positive"),
    "end_nan": (dict(n_knots=1, end_time=float("nan")), "end_time must be positive"),
    "bad_end_bad_bounds": (
        dict(n_knots=2, end_time=-1.0, bounds=(50.0, 20.0)),
        "end_time must be positive",
    ),
    "bounds_reversed": (
        dict(n_knots=2, end_time=100.0, bounds=(50.0, 20.0)),
        "knot ranges must satisfy 0 <= low < high <= end_time (100.0)",
    ),
    "bounds_negative": (
        dict(n_knots=2, end_time=100.0, bounds=(-1.0, 50.0)),
        "knot ranges must satisfy 0 <= low < high <= end_time (100.0)",
    ),
    "bounds_beyond_end": (
        dict(n_knots=2, end_time=100.0, bounds=(10.0, 120.0)),
        "knot ranges must satisfy 0 <= low < high <= end_time (100.0)",
    ),
    "bounds_overlap": (
        dict(n_knots=2, end_time=100.0, bounds=[(10.0, 60.0), (50.0, 90.0)]),
        "per-knot ranges must be ascending and must not overlap",
    ),
    "bounds_wrong_count": (
        dict(n_knots=2, end_time=100.0, bounds=[(10.0, 60.0)]),
        "bounds must be one (low, high) pair or 2 pairs",
    ),
    "spacing_negative": (
        dict(n_knots=2, end_time=100.0, min_spacing=-1.0),
        "min_spacing must be a non-negative finite number",
    ),
    "spacing_nan": (
        dict(n_knots=2, end_time=100.0, min_spacing=float("nan")),
        "min_spacing must be a non-negative finite number",
    ),
    "spacing_inf": (
        dict(n_knots=2, end_time=100.0, min_spacing=float("inf")),
        "min_spacing must be a non-negative finite number",
    ),
    "spacing_infeasible": (
        dict(n_knots=3, end_time=100.0, bounds=(10.0, 40.0), min_spacing=20.0),
        "3 knots at least 20.0 apart do not fit in the knot ranges ((10.0, 40.0), (10.0, 40.0), (10.0, 40.0))",
    ),
    "bad_bounds_bad_spacing": (
        dict(n_knots=2, end_time=100.0, bounds=(50.0, 20.0), min_spacing=-1.0),
        "knot ranges must satisfy 0 <= low < high <= end_time (100.0)",
    ),
    "bad_spacing_bad_initial": (
        dict(n_knots=2, end_time=100.0, min_spacing=-1.0, initial=[1.0]),
        "min_spacing must be a non-negative finite number",
    ),
    "initial_wrong_length": (
        dict(n_knots=2, end_time=100.0, initial=[50.0]),
        "initial has 1 times, expected 2",
    ),
    "initial_single_outside": (
        dict(n_knots=1, end_time=100.0, bounds=(10.0, 20.0), initial=[50.0]),
        "initial must be strictly increasing and inside the knot ranges",
    ),
    "initial_on_edge": (
        dict(n_knots=1, end_time=100.0, bounds=(10.0, 20.0), initial=[10.0]),
        "initial must be strictly increasing and inside the knot ranges",
    ),
    "initial_decreasing": (
        dict(n_knots=2, end_time=100.0, initial=[60.0, 40.0]),
        "initial must be strictly increasing and inside the knot ranges",
    ),
    "initial_equal": (
        dict(n_knots=2, end_time=100.0, initial=[40.0, 40.0]),
        "initial must be strictly increasing and inside the knot ranges",
    ),
    "initial_outside_per_knot": (
        dict(
            n_knots=2,
            end_time=100.0,
            bounds=[(10.0, 40.0), (50.0, 90.0)],
            initial=[45.0, 60.0],
        ),
        "initial must be strictly increasing and inside the knot ranges",
    ),
    "initial_too_close": (
        dict(n_knots=2, end_time=100.0, initial=[40.0, 45.0], min_spacing=20.0),
        "initial knots must be at least min_spacing (20.0) apart",
    ),
    "initial_wrong_length_bad_order": (
        dict(n_knots=3, end_time=100.0, initial=[60.0, 40.0]),
        "initial has 2 times, expected 3",
    ),
    "initial_bad_order_too_close": (
        dict(n_knots=2, end_time=100.0, initial=[40.0, 30.0], min_spacing=20.0),
        "initial must be strictly increasing and inside the knot ranges",
    ),
}

_GOOD_KNOTS: dict[str, tuple[dict[str, Any], tuple[Any, ...]]] = {
    "defaults": (
        dict(n_knots=2, end_time=90),
        (((0.0, 90.0), (0.0, 90.0)), (30.0, 60.0), 0.0, 90.0),
    ),
    "one_default": (
        dict(n_knots=1, end_time=1200.0, bounds=(300.0, 900.0)),
        (((300.0, 900.0),), (600.0,), 0.0, 1200.0),
    ),
    "shared_bounds": (
        dict(n_knots=2, end_time=100.0, bounds=(20.0, 80.0)),
        (((20.0, 80.0), (20.0, 80.0)), (40.0, 60.0), 0.0, 100.0),
    ),
    "per_knot": (
        dict(n_knots=2, end_time=100.0, bounds=[(10.0, 40.0), (50.0, 90.0)]),
        (((10.0, 40.0), (50.0, 90.0)), (25.0, 70.0), 0.0, 100.0),
    ),
    "initial_given": (
        dict(n_knots=2, end_time=100.0, initial=(20, 70), min_spacing=5),
        (((0.0, 100.0), (0.0, 100.0)), (20.0, 70.0), 5.0, 100.0),
    ),
    "nudged_shared": (
        dict(n_knots=3, end_time=100.0, bounds=(40.0, 60.0), min_spacing=10.0),
        (((40.0, 60.0), (40.0, 60.0), (40.0, 60.0)), (40.0, 50.0, 60.0), 10.0, 100.0),
    ),
    "nudged_per_knot": (
        dict(
            n_knots=2,
            end_time=100.0,
            bounds=[(40.0, 50.0), (50.0, 60.0)],
            min_spacing=12.0,
        ),
        (((40.0, 50.0), (50.0, 60.0)), (45.0, 57.0), 12.0, 100.0),
    ),
    "spacing_spread_default_ok": (
        dict(n_knots=3, end_time=100.0, min_spacing=25.0),
        (((0.0, 100.0), (0.0, 100.0), (0.0, 100.0)), (25.0, 50.0, 75.0), 25.0, 100.0),
    ),
    "spacing_exact_initial": (
        dict(n_knots=2, end_time=100.0, initial=[40.0, 60.0], min_spacing=20.0),
        (((0.0, 100.0), (0.0, 100.0)), (40.0, 60.0), 20.0, 100.0),
    ),
    "spacing_feasible_per_knot": (
        dict(
            n_knots=2,
            end_time=100.0,
            bounds=[(10.0, 20.0), (25.0, 30.0)],
            min_spacing=15.0,
        ),
        (((10.0, 20.0), (25.0, 30.0)), (15.0, 30.0), 15.0, 100.0),
    ),
}


class TestFreeKnotsValidationPinned:
    """Pin the error messages and resulting fields of ``FreeKnots``."""

    @pytest.mark.parametrize("case", list(_BAD_KNOTS))
    def test_error_messages_are_unchanged(self, case: str) -> None:
        kwargs, message = _BAD_KNOTS[case]
        with pytest.raises(ValueError, match=r".") as caught:
            FreeKnots(**kwargs)
        assert str(caught.value) == message

    @pytest.mark.parametrize("case", list(_GOOD_KNOTS))
    def test_valid_fields_are_unchanged(self, case: str) -> None:
        kwargs, expected = _GOOD_KNOTS[case]
        knots = FreeKnots(**kwargs)
        assert (
            knots.bounds,
            knots.initial,
            knots.min_spacing,
            knots.end_time,
        ) == expected
