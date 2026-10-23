"""Test name-keyed parameter constraints."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.rate_models import BirthDeathModel, Param

_MODEL = BirthDeathModel.piecewise_constant([1.0, 2.0, 3.0])
_PARAMS = _MODEL.params


class TestValidation:
    """Test compile-time validation."""

    def test_empty_constraints_are_valid(self) -> None:
        compiled = Constraints().compile(_PARAMS)
        assert compiled.n_free == 6
        assert compiled.free_names == _MODEL.names

    def test_unknown_name_in_fixed(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown parameter 'birth\.v9'"):
            Constraints(fixed={"birth.v9": 0.5}).compile(_PARAMS)

    def test_unknown_name_in_equal(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter 'nope'"):
            Constraints(equal=[["birth.v0", "nope"]]).compile(_PARAMS)

    def test_fixed_and_equal_overlap(self) -> None:
        with pytest.raises(ValueError, match="both fixed and equal"):
            Constraints(
                fixed={"birth.v0": 0.5}, equal=[["birth.v0", "birth.v1"]]
            ).compile(_PARAMS)

    def test_parameter_in_two_groups(self) -> None:
        with pytest.raises(ValueError, match="more than one group"):
            Constraints(
                equal=[["birth.v0", "birth.v1"], ["birth.v1", "birth.v2"]]
            ).compile(_PARAMS)

    def test_single_element_group(self) -> None:
        with pytest.raises(ValueError, match="at least 2 elements"):
            Constraints(equal=[["birth.v0"]]).compile(_PARAMS)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside its domain"):
            Constraints(fixed={"birth.v0": -0.1}).compile(_PARAMS)

    def test_logit_domain(self) -> None:
        params = (Param("amplitude", "logit"), Param("phase", "identity"))
        with pytest.raises(ValueError, match="outside its domain"):
            Constraints(fixed={"amplitude": 1.5}).compile(params)
        Constraints(fixed={"phase": -3.0}).compile(params)

    def test_mixed_transforms_in_group(self) -> None:
        params = (Param("a", "log"), Param("b", "identity"))
        with pytest.raises(ValueError, match="mixes parameter transforms"):
            Constraints(equal=[["a", "b"]]).compile(params)


class TestExpandReduce:
    """Test the free <-> natural maps."""

    def test_no_constraints_is_exp(self) -> None:
        compiled = Constraints().compile(_PARAMS)
        free = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        np.testing.assert_allclose(compiled.expand(free), jnp.exp(free))

    def test_fixed_values_including_zero(self) -> None:
        compiled = Constraints(fixed={"birth.v1": 0.0, "death.v0": 0.25}).compile(
            _PARAMS
        )
        assert compiled.n_free == 4
        natural = compiled.expand(jnp.zeros(4))
        assert float(natural[1]) == 0.0
        assert float(natural[3]) == pytest.approx(0.25)
        assert float(natural[0]) == pytest.approx(1.0)

    def test_equality_groups_share_one_free_parameter(self) -> None:
        compiled = Constraints(
            equal=[["birth.v0", "birth.v1", "birth.v2"], ["death.v0", "death.v1"]]
        ).compile(_PARAMS)
        assert compiled.free_names == ("birth.v0", "death.v0", "death.v2")
        natural = np.asarray(compiled.expand(jnp.log(jnp.array([0.2, 0.7, 0.9]))))
        np.testing.assert_allclose(natural, [0.2, 0.2, 0.2, 0.7, 0.7, 0.9])

    def test_round_trip(self) -> None:
        compiled = Constraints(
            fixed={"birth.v0": 0.1}, equal=[["birth.v1", "birth.v2"]]
        ).compile(_PARAMS)
        natural = jnp.array([0.1, 0.2, 0.2, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(compiled.expand(compiled.reduce(natural)), natural)

    def test_reduce_picks_group_representative(self) -> None:
        compiled = Constraints(equal=[["birth.v0", "birth.v1"]]).compile(_PARAMS)
        reduced = compiled.reduce(jnp.array([0.2, 99.0, 1.0, 1.0, 1.0, 1.0]))
        assert float(jnp.exp(reduced[0])) == pytest.approx(0.2)

    def test_logit_and_identity_transforms(self) -> None:
        params = (Param("amp", "logit"), Param("shift", "identity"))
        compiled = Constraints().compile(params)
        natural = compiled.expand(jnp.array([0.0, -2.5]))
        np.testing.assert_allclose(natural, [0.5, -2.5])
        np.testing.assert_allclose(compiled.reduce(natural), [0.0, -2.5], atol=1e-12)

    def test_gradient_flows_through_expand(self) -> None:
        compiled = Constraints(
            fixed={"birth.v0": 0.1}, equal=[["birth.v1", "birth.v2"]]
        ).compile(_PARAMS)
        grad = jax.grad(lambda u: compiled.expand(u).sum())(jnp.zeros(compiled.n_free))
        # birth.v1 and birth.v2 share a slot, so that slot receives two contributions
        assert float(grad[0]) == pytest.approx(2.0)
        assert bool(jnp.all(jnp.isfinite(grad)))
