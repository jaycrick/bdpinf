"""Test interval-censored ages: range masses, the range likelihood and fits."""

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.bayesian import BayesianMCMCConfig, fit_bayesian
from bdpinf.inference.fitting import fit
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import BirthDeathModel
from bdpinf.models.math_age_masses import range_log_masses
from bdpinf.models.math_general import terms_for
from bdpinf.simulation.age_sampling import simulate_ages_from_rates

_T = 10.0
_EDGES = jnp.array([0.0, 1.5, 4.0, 7.0, 9.0, _T])
_NATURAL = jnp.array([1.2, 0.6, 0.4, 0.4])


def _rates(natural: jnp.ndarray = _NATURAL):
    model = BirthDeathModel.piecewise_constant([5.0, _T])
    return model, *model.rates(natural)


def _ranges(seed: int = 0, n: int = 40) -> AgeData:
    """Unit-width ranges around ages simulated from the constant-rate model."""
    rng = np.random.default_rng(seed)
    _, birth, death = _rates()
    ages = np.concatenate(
        [
            np.asarray(simulate_ages_from_rates(birth, death, rng).all_ages)
            for _ in range(4)
        ]
    )[:n]
    lower = np.minimum(np.floor(ages), _T - 1.0)
    return AgeData.from_ranges(lower, lower + 1.0, _T)


class TestAgeRanges:
    """Test the range likelihood."""

    def test_closed_form_matches_quadrature(self) -> None:
        _, birth, death = _rates()
        terms = terms_for(birth, death)
        lo, hi = _EDGES[:-1], _EDGES[1:]
        exact = range_log_masses(terms, lo, hi)
        numeric = range_log_masses(terms, lo, hi, quadrature=True)
        np.testing.assert_allclose(exact[0], numeric[0], atol=1e-6)
        np.testing.assert_allclose(exact[1], numeric[1], atol=1e-6)

    def test_cumulative_form_matches_quadrature_for_linear_rates(self) -> None:
        model = BirthDeathModel.piecewise_linear([3.0, 7.0, _T])
        natural = jnp.array([1.0, 0.8, 0.5, 0.3, 0.0, 0.2, 0.3, 0.4])
        terms = terms_for(*model.rates(natural))
        # Ranges that cut a knot, sit inside an epoch, and touch both ends.
        lo = jnp.array([0.0, 1.0, 2.5, 6.0, 8.0])
        hi = jnp.array([1.0, 2.0, 4.0, 9.0, _T])
        fast = range_log_masses(terms, lo, hi)
        slow = range_log_masses(terms, lo, hi, quadrature=True)
        np.testing.assert_allclose(fast[0], slow[0], atol=1e-8)
        np.testing.assert_allclose(fast[1], slow[1], atol=1e-8)

    def test_masses_over_a_partition_sum_to_one(self) -> None:
        _, birth, death = _rates()
        non_spine, spine = range_log_masses(
            terms_for(birth, death), _EDGES[:-1], _EDGES[1:]
        )
        assert float(jnp.exp(non_spine).sum()) == pytest.approx(1.0)
        assert float(jnp.exp(spine).sum()) == pytest.approx(1.0)

    def test_fine_ranges_approach_exact_ages(self) -> None:
        _, birth, death = _rates()
        rng = np.random.default_rng(2)
        ages = np.sort(simulate_ages_from_rates(birth, death, rng).all_ages)
        ages = np.asarray(ages)[:20]
        exact = AgeData.from_groups([ages], _T)
        width = 1e-4
        fine = AgeData.from_ranges(
            ages - width / 2, ages + width / 2, _T, population_size=int(exact.ns[0])
        )
        assert float(log_likelihood_rates(fine, birth, death)) == pytest.approx(
            float(log_likelihood_rates(exact, birth, death)), abs=2e-3
        )

    def test_padding_keeps_value(self) -> None:
        _, birth, death = _rates()
        data = _ranges()
        padded = data.padded(64)
        assert padded.n_observations == data.n_observations
        assert float(log_likelihood_rates(padded, birth, death)) == pytest.approx(
            float(log_likelihood_rates(data, birth, death))
        )

    def test_duplicate_ranges_are_merged(self) -> None:
        data = AgeData.from_ranges([1.0, 1.0, 3.0], [2.0, 2.0, 4.0], _T)
        assert len(data.non_spine_ages) == 2
        assert data.n_observations == 3
        assert data.has_ranges

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [([1.0], [1.0]), ([-1.0], [1.0]), ([1.0], [11.0]), ([1.0, 2.0], [3.0])],
    )
    def test_rejects_bad_ranges(self, lower: list[float], upper: list[float]) -> None:
        with pytest.raises(ValueError, match=r"ange|length"):
            AgeData.from_ranges(lower, upper, _T)

    def test_fit_runs_on_ranges(self) -> None:
        result = fit(_ranges(), BirthDeathModel.piecewise_constant([_T]))
        assert result.success
        assert np.isfinite(result.log_likelihood)

    def test_bayesian_runs_on_ranges(self) -> None:
        result = fit_bayesian(
            _ranges(n=15),
            BirthDeathModel.piecewise_constant([_T]),
            mcmc=BayesianMCMCConfig(
                num_samples=20, warmup_steps=20, num_chains=1, disable_progressbar=True
            ),
            rng_seed=0,
        )
        assert np.all(np.isfinite(result.samples))


class TestFromYearRanges:
    """Test AgeData.from_year_ranges."""

    def test_interior_range(self) -> None:
        data = AgeData.from_year_ranges(
            [300.0], [399.0], start_year=100, observation_year=2000
        )
        assert data.has_ranges
        assert data.end_time == 1900.0
        np.testing.assert_allclose(data.non_spine_ages, [2000 - 400])
        np.testing.assert_allclose(data.upper_ages, [2000 - 300])

    def test_clipped_to_support(self) -> None:
        data = AgeData.from_year_ranges(
            [50.0], [150.0], start_year=100, observation_year=2000
        )
        np.testing.assert_allclose(data.upper_ages, [2000 - 100])

    def test_outside_support_kept_one_year_wide(self) -> None:
        data = AgeData.from_year_ranges(
            [1990.0, 2010.0], [2005.0, 2020.0], start_year=100, observation_year=2000
        )
        lower = np.asarray(data.non_spine_ages)
        assert np.all(np.asarray(data.upper_ages) - lower >= 1.0)
        assert np.all(lower >= 0.0)

    def test_initial_population(self) -> None:
        data = AgeData.from_year_ranges(
            [300.0],
            [399.0],
            start_year=100,
            observation_year=2000,
            initial_population=5,
        )
        assert data.initial_population == 5

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            AgeData.from_year_ranges(
                [1.0, 2.0], [3.0], start_year=0, observation_year=10
            )

    def test_bad_window_raises(self) -> None:
        with pytest.raises(ValueError, match="later than"):
            AgeData.from_year_ranges([1.0], [2.0], start_year=10, observation_year=10)
