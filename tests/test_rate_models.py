"""Test the parametric rate families and the birth-death model wrapper."""

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import (
    BirthDeathModel,
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
