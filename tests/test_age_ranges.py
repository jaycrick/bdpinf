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
