"""Test AgeData construction and the rate-function likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.likelihood import AgeData, log_likelihood, log_likelihood_rates
from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate

_TIMES = jnp.array([1.0, 2.5])
_AGES = jnp.array([0.2, 0.5, 0.9, 1.4, 2.0, 2.3])


class TestAgeData:
    """Test host-side resolution of the spine and counts."""

    def test_unknown_spine(self) -> None:
        data = AgeData.from_groups([_AGES], 2.5)
        assert data.use_unknown_spine
        assert data.spine_ages is None
        assert data.n_observations == len(_AGES)
        assert int(data.ns[0]) == len(_AGES)

    def test_age_at_end_time_is_spine(self) -> None:
        data = AgeData.from_groups([jnp.append(_AGES, 2.5)], 2.5)
        assert not data.use_unknown_spine
        assert data.spine_ages is not None
        assert float(data.spine_ages[0]) == 2.5
        assert len(data.non_spine_ages) == len(_AGES)

    def test_ordered(self) -> None:
        data = AgeData.from_groups([_AGES], 2.5, is_ordered=True)
        assert data.spine_ages is not None
        assert float(data.spine_ages[0]) == float(_AGES[0])
        assert len(data.non_spine_ages) == len(_AGES) - 1
        assert not data.use_unknown_spine

    def test_rejects_out_of_range_and_empty(self) -> None:
        with pytest.raises(ValueError, match="Ages must be in the range"):
            AgeData.from_groups([jnp.array([0.1, 3.0])], 2.5)
        with pytest.raises(ValueError, match="No valid data"):
            AgeData.from_groups([jnp.array([])], 2.5)
        with pytest.raises(ValueError, match="population_sizes"):
            AgeData.from_groups([_AGES], 2.5, population_sizes=jnp.array([1, 2]))


class TestLogLikelihoodRates:
    """Test the rate-function likelihood against the epoch-array one."""

    def test_matches_array_api(self) -> None:
        birth, death = jnp.array([1.2, 0.8]), jnp.array([0.7, 1.1])
        data = AgeData.from_groups([_AGES], 2.5)
        want = log_likelihood(birth, death, _AGES, _TIMES + 1.5)
        got = log_likelihood_rates(
            data,
            PiecewiseConstantRate(birth, _TIMES + 1.5),
            PiecewiseConstantRate(death, _TIMES + 1.5),
        )
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_constant_valued_linear_rates_match_piecewise_constant(self) -> None:
        data = AgeData.from_groups([_AGES], 2.5)
        times = jnp.array([1.0, 2.5])
        birth, death = jnp.array([1.2, 0.8]), jnp.array([0.7, 1.1])
        exact = log_likelihood_rates(
            data,
            PiecewiseConstantRate(birth, times),
            PiecewiseConstantRate(death, times),
        )
        quad = log_likelihood_rates(
            data,
            PiecewiseLinearRate.from_constant(birth, times),
            PiecewiseLinearRate.from_constant(death, times),
        )
        np.testing.assert_allclose(quad, exact, rtol=1e-9)

    def test_gradient_through_linear_rates(self) -> None:
        data = AgeData.from_groups([_AGES], 2.5)
        times = jnp.array([1.0, 2.5])

        def nll(log_knots: jax.Array) -> jax.Array:
            k = jnp.exp(log_knots)
            return -log_likelihood_rates(
                data,
                PiecewiseLinearRate.continuous(k[:3], times),
                PiecewiseLinearRate.continuous(k[3:], times),
            )

        x = jnp.log(jnp.array([1.2, 1.0, 0.8, 0.7, 0.9, 1.1]))
        grad = jax.jit(jax.grad(nll))(x)
        assert bool(jnp.all(jnp.isfinite(grad)))
        step = 1e-6
        fd = jnp.array(
            [
                (nll(x.at[i].add(step)) - nll(x.at[i].add(-step))) / (2 * step)
                for i in range(6)
            ]
        )
        np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-7)


class TestPadding:
    """Test that masked padding leaves values and gradients unchanged."""

    @staticmethod
    def _nll(data: AgeData, log_values: jax.Array) -> jax.Array:
        v = jnp.exp(log_values)
        times = jnp.array([1.0, 2.5])
        return -log_likelihood_rates(
            data,
            PiecewiseLinearRate.continuous(v[:3], times),
            PiecewiseLinearRate.continuous(v[3:], times),
        )

    @pytest.mark.parametrize("with_spine", [False, True])
    def test_value_and_gradient_match(self, with_spine: bool) -> None:
        ages = jnp.append(_AGES, 2.5) if with_spine else _AGES
        plain = AgeData.from_groups([ages], 2.5)
        padded = plain.padded(16)
        assert len(padded.non_spine_ages) == 16
        assert padded.n_observations == plain.n_observations
        x = jnp.log(jnp.array([1.2, 1.0, 0.8, 0.7, 0.9, 1.1]))
        v0, g0 = jax.value_and_grad(lambda u: self._nll(plain, u))(x)
        v1, g1 = jax.value_and_grad(lambda u: self._nll(padded, u))(x)
        np.testing.assert_allclose(v1, v0, rtol=1e-12)
        np.testing.assert_allclose(g1, g0, rtol=1e-9, atol=1e-12)

    def test_cannot_pad_below_current_size(self) -> None:
        with pytest.raises(ValueError, match="smaller size"):
            AgeData.from_groups([_AGES], 2.5).padded(2)
