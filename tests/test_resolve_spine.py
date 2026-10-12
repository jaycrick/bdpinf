"""Test shared spine resolution and its use by the likelihood and the process."""

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.likelihood import log_likelihood, resolve_spine_ages
from bdpinf.process import BirthDeathProcess

_T = 10.0


class TestResolveSpineAges:
    def test_age_at_final_time_is_snapped_to_exactly_t(self) -> None:
        result = resolve_spine_ages([2.0, _T - 1e-7, 5.0], _T)
        assert result.spine_ages is not None
        assert float(result.spine_ages[0]) == _T
        np.testing.assert_array_equal(result.non_spine_ages, [2.0, 5.0])
        assert not result.use_unknown_spine

    def test_single_age_is_the_spine(self) -> None:
        result = resolve_spine_ages([3.0], _T)
        assert result.spine_ages is not None
        np.testing.assert_array_equal(result.spine_ages, [3.0])
        assert result.non_spine_ages.size == 0
        assert not result.use_unknown_spine

    def test_unknown_spine_when_no_age_at_t(self) -> None:
        result = resolve_spine_ages([1.0, 2.0, 3.0], _T)
        assert result.spine_ages is None
        assert result.use_unknown_spine

    def test_empty_ages(self) -> None:
        result = resolve_spine_ages([], _T)
        assert result.spine_ages is None
        assert result.non_spine_ages.size == 0

    def test_multiple_terminal_ages_raise_or_marginalize(self) -> None:
        with pytest.raises(ValueError, match="More than one age"):
            resolve_spine_ages([_T, _T, 3.0], _T)
        result = resolve_spine_ages([_T, _T, 3.0], _T, multiple_terminal="marginalize")
        assert result.spine_ages is None
        assert result.use_unknown_spine


class TestSnappingGivesTheAtom:
    def test_age_just_below_t_is_scored_with_the_atom(self) -> None:
        kwargs = {
            "birth_rates": jnp.array([0.9, 1.6, 0.7]),
            "death_rates": jnp.array([1.2, 0.5, 1.0]),
            "times": jnp.array([3.0, 6.0, _T]),
        }
        ages_exact = jnp.array([_T, 2.0, 5.0])
        ages_near = jnp.array([_T - 1e-7, 2.0, 5.0])
        exact = float(log_likelihood(ages=ages_exact, **kwargs))
        near = float(log_likelihood(ages=ages_near, **kwargs))
        assert near == pytest.approx(exact, abs=1e-12)


class TestProcessUsesLabelledSpine:
    def test_matches_explicit_spine_call(self) -> None:
        birth, death = [0.6, 0.9], [0.8, 0.5]
        process = BirthDeathProcess.from_age_samples(
            spine_ages=[7.0],
            non_spine_ages=[2.0, 4.0, 6.5],
            epoch_end_times=[5.0, _T],
            birth_rates=birth,
            death_rates=death,
        )
        explicit = log_likelihood(
            jnp.array(birth),
            jnp.array(death),
            ages=jnp.array([2.0, 4.0, 6.5]),
            times=jnp.array([5.0, _T]),
            spine_ages=jnp.array([7.0]),
            unbias=True,
        )
        assert float(process.log_likelihood()) == pytest.approx(float(explicit))
        unknown = log_likelihood(
            jnp.array(birth),
            jnp.array(death),
            ages=jnp.array([7.0, 2.0, 4.0, 6.5]),
            times=jnp.array([5.0, _T]),
            unbias=True,
        )
        assert float(process.log_likelihood()) != pytest.approx(float(unknown))
