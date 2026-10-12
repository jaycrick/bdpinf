"""Simple normalization tests for birth-death probability functions."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from bdpinf.inference import calculate_unknown_spine_mixture_age_density
from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.models.math_nested import (
    NestedTerms,
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.models.math_populations import calculate_populations_log_lik


def _pointwise_density_from_log_lik(
    log_density_fn: Callable[[NestedTerms, jax.Array], jax.Array],
    birth_rates: jax.Array,
    death_rates: jax.Array,
    times: jax.Array,
    grid: jax.Array,
) -> jax.Array:
    """Evaluate exp(log-density) on a 1D grid."""
    terms = nested_terms(birth_rates, death_rates, times)
    return jnp.exp(log_density_fn(terms, grid))


class TestProbabilityNormalization:
    """Test that probability densities and mass functions are properly normalized."""

    def test_population_probability_mass_sums_to_one(self) -> None:
        """Population probabilities p_n(T) should sum to 1 over n."""
        birth_rates = jnp.array([0.2], dtype=jnp.float64)
        death_rates = jnp.array([0.5], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)

        probs = []
        for n in range(60):
            log_p_n = calculate_populations_log_lik(
                birth_rates=birth_rates,
                death_rates=death_rates,
                ns=jnp.array([n], dtype=jnp.int32),
                times=times,
                initial_population=1,
            )
            probs.append(jnp.exp(log_p_n))

        total_prob = jnp.sum(jnp.stack(probs))
        assert bool(jnp.isclose(total_prob, jnp.float64(1.0), atol=1e-4))

    def test_non_spine_age_density_integrates_to_one(self) -> None:
        """Non-spine age density should integrate to 1 on [0, T]."""
        birth_rates = jnp.array([0.6], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        times = jnp.array([3.0], dtype=jnp.float64)

        grid = jnp.linspace(0.0, float(times[-1]), num=4000, dtype=jnp.float64)
        density_vals = _pointwise_density_from_log_lik(
            non_spine_log_density,
            birth_rates,
            death_rates,
            times,
            grid,
        )
        integral = float(jnp.trapezoid(density_vals, grid))

        assert abs(integral - 1.0) < 2e-3

    def test_spine_age_distribution_integrates_to_one_with_terminal_mass(
        self,
    ) -> None:
        """Spine distribution has continuous mass on [0, T) plus an atom at T."""
        birth_rates = jnp.array([0.6], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        times = jnp.array([3.0], dtype=jnp.float64)

        t_end = float(times[-1])
        eps = 1e-5
        grid = jnp.linspace(0.0, t_end - eps, num=4000, dtype=jnp.float64)

        cont_density_vals = _pointwise_density_from_log_lik(
            spine_log_density,
            birth_rates,
            death_rates,
            times,
            grid,
        )
        continuous_mass = float(jnp.trapezoid(cont_density_vals, grid))

        terminal_mass = float(
            jnp.exp(
                spine_log_density(
                    nested_terms(birth_rates, death_rates, times),
                    jnp.array([t_end], dtype=jnp.float64),
                ).sum(),
            )
        )

        total_mass = continuous_mass + terminal_mass
        assert abs(total_mass - 1.0) < 2e-3

    def test_extinction_probability_matches_population_zero_mass(self) -> None:
        """Pr(Z_T = 0) should equal p_0(T) from population probabilities."""
        birth_rates = jnp.array([0.4], dtype=jnp.float64)
        death_rates = jnp.array([0.25], dtype=jnp.float64)
        times = jnp.array([2.5], dtype=jnp.float64)

        extinction_pr = calculate_extinction_pr(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
        p0 = jnp.exp(
            calculate_populations_log_lik(
                birth_rates=birth_rates,
                death_rates=death_rates,
                ns=jnp.array([0], dtype=jnp.int32),
                times=times,
                initial_population=1,
            ),
        )

        assert bool(jnp.isclose(extinction_pr, p0, atol=1e-12))

    def test_unknown_spine_mixture_distribution_integrates_to_one(self) -> None:
        """Unknown-spine mixture has continuous mass plus spine atom / n at T."""
        birth_rates = jnp.array([0.6], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        times = jnp.array([3.0], dtype=jnp.float64)
        n = 5

        t_end = float(times[-1])
        eps = 1e-5
        grid = jnp.linspace(0.0, t_end - eps, num=4000, dtype=jnp.float64)

        mixture_cont = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=grid,
            times=times,
            num_individuals=n,
        )
        continuous_mass = float(jnp.trapezoid(mixture_cont, grid))

        spine_terminal_mass = float(
            jnp.exp(
                spine_log_density(
                    nested_terms(birth_rates, death_rates, times),
                    jnp.array([t_end], dtype=jnp.float64),
                ).sum(),
            )
        )
        total_mass = continuous_mass + spine_terminal_mass / n

        assert abs(total_mass - 1.0) < 2e-3

    def test_population_probability_mass_sums_to_one_multi_epoch(self) -> None:
        """Multi-epoch population probabilities p_n(T) should sum to 1."""
        birth_rates = jnp.array([0.15, 0.10, 0.08], dtype=jnp.float64)
        death_rates = jnp.array([0.25, 0.20, 0.18], dtype=jnp.float64)
        times = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

        probs = []
        for n in range(120):
            log_p_n = calculate_populations_log_lik(
                birth_rates=birth_rates,
                death_rates=death_rates,
                ns=jnp.array([n], dtype=jnp.int32),
                times=times,
                initial_population=1,
            )
            probs.append(jnp.exp(log_p_n))

        total_prob = jnp.sum(jnp.stack(probs))
        assert bool(jnp.isclose(total_prob, jnp.float64(1.0), atol=5e-4))

    def test_non_spine_age_density_integrates_to_one_multi_epoch(self) -> None:
        """Multi-epoch non-spine age density should integrate to 1 on [0, T]."""
        birth_rates = jnp.array([0.30, 0.18, 0.12], dtype=jnp.float64)
        death_rates = jnp.array([0.10, 0.14, 0.09], dtype=jnp.float64)
        times = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

        grid = jnp.linspace(0.0, float(times[-1]), num=4000, dtype=jnp.float64)
        density_vals = _pointwise_density_from_log_lik(
            non_spine_log_density,
            birth_rates,
            death_rates,
            times,
            grid,
        )
        integral = float(jnp.trapezoid(density_vals, grid))

        assert abs(integral - 1.0) < 2e-3

    def test_extinction_probability_matches_population_zero_mass_multi_epoch(
        self,
    ) -> None:
        """Multi-epoch Pr(Z_T = 0) should equal p_0(T)."""
        birth_rates = jnp.array([0.20, 0.12, 0.09], dtype=jnp.float64)
        death_rates = jnp.array([0.28, 0.22, 0.18], dtype=jnp.float64)
        times = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

        extinction_pr = calculate_extinction_pr(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
        )
        p0 = jnp.exp(
            calculate_populations_log_lik(
                birth_rates=birth_rates,
                death_rates=death_rates,
                ns=jnp.array([0], dtype=jnp.int32),
                times=times,
                initial_population=1,
            ),
        )

        assert bool(jnp.isclose(extinction_pr, p0, atol=1e-12))
