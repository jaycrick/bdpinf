"""Test name-keyed parameter constraints."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.constraints import Constraints
from bdpinf.inference.rate_models import BirthDeathModel, FreeKnots, Param

_MODEL = BirthDeathModel.piecewise_constant([1.0, 2.0, 3.0])
_PARAMS = _MODEL.params
_KNOT_PARAMS = BirthDeathModel.piecewise_constant(
    FreeKnots(2, 100.0, bounds=(20.0, 80.0))
).params
_MIN_PARAMS = BirthDeathModel.piecewise_constant(
    [1.0, 2.0, 3.0], min_birth=0.1, min_death=0.2
).params
_LOGIT_PARAMS = (
    Param("amp", "logit"),
    Param("shift", "identity"),
    Param("a", "log"),
    Param("b", "log"),
)


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


_CASES: dict[str, tuple[tuple[Param, ...], dict[str, float], list[list[str]]]] = {
    "empty": (_PARAMS, {}, []),
    "fixed_only": (_PARAMS, {"birth.v1": 0.5, "death.v0": 0.0}, []),
    "equal_only": (
        _PARAMS,
        {},
        [["birth.v2", "birth.v0"], ["death.v0", "death.v1", "death.v2"]],
    ),
    "mixed": (
        _PARAMS,
        {"birth.v0": 0.1},
        [["birth.v1", "birth.v2"], ["death.v2", "death.v0"]],
    ),
    "knots_fixed": (_KNOT_PARAMS, {"tau1": 50.0}, []),
    "knots_mixed": (
        _KNOT_PARAMS,
        {"tau0": 30.0, "birth.v0": 0.3},
        [["death.v0", "death.v2"]],
    ),
    "knots_equal_rates": (
        _KNOT_PARAMS,
        {},
        [["birth.v0", "birth.v1", "birth.v2"], ["death.v0", "death.v1", "death.v2"]],
    ),
    "min_values": (
        _MIN_PARAMS,
        {"birth.v0": 0.1, "death.v1": 0.5},
        [["birth.v1", "birth.v2"], ["death.v0", "death.v2"]],
    ),
    "logit_identity": (
        _LOGIT_PARAMS,
        {"amp": 0.25, "shift": -1.5},
        [["a", "b"]],
    ),
    "all_fixed": (
        _LOGIT_PARAMS,
        {"amp": 0.25, "shift": -1.5, "a": 1.0, "b": 2.0},
        [],
    ),
}

_BAD_CASES: dict[str, tuple[tuple[Param, ...], dict[str, float], list[list[str]]]] = {
    "unknown_fixed": (_PARAMS, {"birth.v9": 0.5}, []),
    "unknown_equal": (_PARAMS, {}, [["birth.v0", "nope"]]),
    "fixed_and_equal": (_PARAMS, {"birth.v0": 0.5}, [["birth.v0", "birth.v1"]]),
    "two_groups": (
        _PARAMS,
        {},
        [["birth.v0", "birth.v1"], ["birth.v1", "birth.v2"]],
    ),
    "duplicate_in_group": (_PARAMS, {}, [["birth.v0", "birth.v0"]]),
    "single": (_PARAMS, {}, [["birth.v0"]]),
    "negative": (_PARAMS, {"birth.v0": -0.1}, []),
    "nan": (_PARAMS, {"birth.v0": float("nan")}, []),
    "inf": (_PARAMS, {"birth.v0": float("inf")}, []),
    "inf_identity": (_LOGIT_PARAMS, {"shift": float("inf")}, []),
    "logit_high": (_LOGIT_PARAMS, {"amp": 1.5}, []),
    "logit_negative": (_LOGIT_PARAMS, {"amp": -0.5}, []),
    "below_min": (_MIN_PARAMS, {"birth.v0": 0.05}, []),
    "mixed_transforms": (_LOGIT_PARAMS, {}, [["a", "shift"]]),
    "mixed_minimums": (_MIN_PARAMS, {}, [["birth.v0", "death.v0"]]),
    "knot_low": (_KNOT_PARAMS, {"tau0": 10.0}, []),
    "knot_high": (_KNOT_PARAMS, {"tau0": 90.0}, []),
    "knot_tied": (_KNOT_PARAMS, {}, [["tau0", "tau1"]]),
    "knot_tied_with_rate": (_KNOT_PARAMS, {}, [["birth.v0", "tau1"]]),
    "dup_names": ((Param("a"), Param("a")), {}, []),
    "fixed_first_error": (_PARAMS, {"birth.v9": 1.0, "birth.v0": -1.0}, [["x", "y"]]),
    "fixed_before_equal": (_PARAMS, {"birth.v0": -1.0}, [["x", "y"]]),
    "single_before_unknown": (_PARAMS, {}, [["x"]]),
    "small_group_after_ok": (
        _PARAMS,
        {},
        [["birth.v0", "birth.v1"], ["birth.v2"]],
    ),
}

# Recorded from the pre-refactor ``Constraints.compile``.
_COMPILED = {
    "empty": {
        "free_indices": [0, 1, 2, 3, 4, 5],
        "source": [0, 1, 2, 3, 4, 5],
        "is_fixed": [False, False, False, False, False, False],
        "fixed_natural": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "codes": [1, 1, 1, 1, 1, 1],
        "lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "fixed_only": {
        "free_indices": [0, 2, 4, 5],
        "source": [0, 0, 1, 0, 2, 3],
        "is_fixed": [False, True, False, True, False, False],
        "fixed_natural": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        "codes": [1, 1, 1, 1, 1, 1],
        "lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "equal_only": {
        "free_indices": [0, 1, 3],
        "source": [0, 1, 0, 2, 2, 2],
        "is_fixed": [False, False, False, False, False, False],
        "fixed_natural": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "codes": [1, 1, 1, 1, 1, 1],
        "lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "mixed": {
        "free_indices": [1, 3, 4],
        "source": [0, 0, 0, 1, 2, 1],
        "is_fixed": [True, False, False, False, False, False],
        "fixed_natural": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        "codes": [1, 1, 1, 1, 1, 1],
        "lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "knots_fixed": {
        "free_indices": [0, 1, 2, 3, 4, 5, 6],
        "source": [0, 1, 2, 3, 4, 5, 6, 0],
        "is_fixed": [False, False, False, False, False, False, False, True],
        "fixed_natural": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        "codes": [1, 1, 1, 1, 1, 1, 3, 3],
        "lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 80.0, 80.0],
    },
    "knots_mixed": {
        "free_indices": [1, 2, 3, 4, 7],
        "source": [0, 0, 1, 2, 3, 2, 0, 4],
        "is_fixed": [True, False, False, False, False, False, True, False],
        "fixed_natural": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0],
        "codes": [1, 1, 1, 1, 1, 1, 3, 3],
        "lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 80.0, 80.0],
    },
    "knots_equal_rates": {
        "free_indices": [0, 3, 6, 7],
        "source": [0, 0, 0, 1, 1, 1, 2, 3],
        "is_fixed": [False, False, False, False, False, False, False, False],
        "fixed_natural": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "codes": [1, 1, 1, 1, 1, 1, 3, 3],
        "lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 80.0, 80.0],
    },
    "min_values": {
        "free_indices": [1, 3],
        "source": [0, 0, 0, 1, 0, 1],
        "is_fixed": [True, False, False, False, True, False],
        "fixed_natural": [0.1, 0.0, 0.0, 0.0, 0.5, 0.0],
        "codes": [1, 1, 1, 1, 1, 1],
        "lower": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    "logit_identity": {
        "free_indices": [2],
        "source": [0, 0, 0, 0],
        "is_fixed": [True, True, False, False],
        "fixed_natural": [0.25, -1.5, 0.0, 0.0],
        "codes": [2, 0, 1, 1],
        "lower": [0.0, 0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0, 1.0],
    },
    "all_fixed": {
        "free_indices": [],
        "source": [0, 0, 0, 0],
        "is_fixed": [True, True, True, True],
        "fixed_natural": [0.25, -1.5, 1.0, 2.0],
        "codes": [2, 0, 1, 1],
        "lower": [0.0, 0.0, 0.0, 0.0],
        "upper": [1.0, 1.0, 1.0, 1.0],
    },
}

_ERRORS = {
    "unknown_fixed": "Unknown parameter 'birth.v9' in fixed",
    "unknown_equal": "Unknown parameter 'nope' in equal",
    "fixed_and_equal": "Parameter 'birth.v0' appears in both fixed and equal",
    "two_groups": "Parameter 'birth.v1' appears in more than one group",
    "duplicate_in_group": "Parameter 'birth.v0' appears in more than one group",
    "single": "Equality groups must have at least 2 elements",
    "negative": "Fixed value for 'birth.v0' is outside its domain: -0.1",
    "nan": "Fixed value for 'birth.v0' is outside its domain: nan",
    "inf": "Fixed value for 'birth.v0' is outside its domain: inf",
    "inf_identity": "Fixed value for 'shift' is outside its domain: inf",
    "logit_high": "Fixed value for 'amp' is outside its domain: 1.5",
    "logit_negative": "Fixed value for 'amp' is outside its domain: -0.5",
    "below_min": "Fixed value for 'birth.v0' is outside its domain: 0.05",
    "mixed_transforms": "Equality group ['a', 'shift'] mixes parameter transforms",
    "mixed_minimums": "Equality group ['birth.v0', 'death.v0'] mixes different minimum "
    "values",
    "knot_low": "Fixed value for 'tau0' is outside its domain: 10.0",
    "knot_high": "Fixed value for 'tau0' is outside its domain: 90.0",
    "knot_tied": "Parameter 'tau0' is a knot time and cannot be tied",
    "knot_tied_with_rate": "Parameter 'tau1' is a knot time and cannot be tied",
    "dup_names": "Parameter names must be unique",
    "fixed_first_error": "Unknown parameter 'birth.v9' in fixed",
    "fixed_before_equal": "Fixed value for 'birth.v0' is outside its domain: -1.0",
    "single_before_unknown": "Equality groups must have at least 2 elements",
    "small_group_after_ok": "Equality groups must have at least 2 elements",
}


class TestCompilePinned:
    """Pin every field of the compiled constraints and every error message."""

    @pytest.mark.parametrize("case", list(_CASES))
    def test_compiled_fields_are_unchanged(self, case: str) -> None:
        params, fixed, equal = _CASES[case]
        compiled = Constraints(fixed=fixed, equal=equal).compile(params)
        expected = _COMPILED[case]
        assert compiled.params == params
        assert compiled.free_indices.tolist() == expected["free_indices"]
        assert compiled.source.tolist() == expected["source"]
        assert compiled.is_fixed.tolist() == expected["is_fixed"]
        assert compiled.fixed_natural.tolist() == expected["fixed_natural"]
        assert compiled.codes.tolist() == expected["codes"]
        assert compiled.lower.tolist() == expected["lower"]
        assert compiled.upper.tolist() == expected["upper"]
        assert compiled.free_indices.dtype == np.int64
        assert compiled.source.dtype == np.int64
        assert compiled.codes.dtype == np.int64
        assert compiled.fixed_natural.dtype == np.float64

    @pytest.mark.parametrize("case", list(_BAD_CASES))
    def test_error_messages_are_unchanged(self, case: str) -> None:
        params, fixed, equal = _BAD_CASES[case]
        with pytest.raises(ValueError, match=r".") as caught:
            Constraints(fixed=fixed, equal=equal).compile(params)
        assert str(caught.value) == _ERRORS[case]
