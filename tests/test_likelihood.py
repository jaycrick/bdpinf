"""Unit tests for the likelihood fns in the bdpinf.inference.likelihood module."""

import jax
import jax.numpy as jnp
import pytest

from bdpinf.inference import (
    calculate_unknown_spine_mixture_age_density,
    calculate_unknown_spine_mixture_birth_density,
    likelihood,
)
from bdpinf.models.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)


def _expected_unknown_spine_age_log_lik(
    birth_rates: jax.Array,
    death_rates: jax.Array,
    ages: jax.Array,
    times: jax.Array,
) -> jax.Array:
    """Compute the unknown-spine age log-likelihood via stable log-sum-exp."""
    sum_log_f = calculate_non_spine_ages_log_lik(
        birth_rates,
        death_rates,
        ages,
        times=times,
    )
    a_terms = []
    for age in ages:
        age_tensor = jnp.array([float(age)], dtype=jnp.float64)
        log_g = calculate_spine_ages_log_lik(
            birth_rates,
            death_rates,
            age_tensor,
            times=times,
        )
        log_f = calculate_non_spine_ages_log_lik(
            birth_rates,
            death_rates,
            age_tensor,
            times=times,
        )
        a_terms.append(log_g - log_f)

    return (
        sum_log_f
        - jnp.log(jnp.float64(len(ages)))
        + jax.scipy.special.logsumexp(jnp.stack(a_terms))
    )


class TestLikelihood:
    """Test likelihood functions."""

    def test_log_likelihood_simple(self) -> None:
        """Test log_likelihood with simple valid input and real function outputs."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        ages = jnp.array([1.0], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)

        result = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        expected = likelihood.calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ns=jnp.array([1], dtype=jnp.int32),
            initial_population=1,
        ) + _expected_unknown_spine_age_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        assert bool(jnp.isclose(result, expected))

    def test_log_likelihood_negative_branch(self) -> None:
        """Test log_likelihood returns -inf when population log-likelihood is non-finite."""
        result = likelihood.log_likelihood(
            birth_rates=jnp.array([1000.0], dtype=jnp.float64),
            death_rates=jnp.array([0.1], dtype=jnp.float64),
            ages=jnp.array([1.0], dtype=jnp.float64),
            times=jnp.array([2.0], dtype=jnp.float64),
        )
        assert bool(jnp.isneginf(result))

    def test_log_likelihood_invalid_params(self) -> None:
        """Test log_likelihood returns -inf for invalid parameters (e.g., neg rates)."""
        result = likelihood.log_likelihood(
            birth_rates=jnp.array([-0.5], dtype=jnp.float64),
            death_rates=jnp.array([0.2], dtype=jnp.float64),
            ages=jnp.array([1.0], dtype=jnp.float64),
            times=jnp.array([2.0], dtype=jnp.float64),
        )
        assert float(result) == float("-inf")

    def test_log_likelihood_log_scale(self) -> None:
        """Test log_likelihood with log_scale_params=True."""
        log_birth_rates = jnp.array([0.0], dtype=jnp.float64)
        log_death_rates = jnp.array([0.0], dtype=jnp.float64)
        ages = jnp.array([1.0], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)

        result = likelihood.log_likelihood(
            birth_rates=log_birth_rates,
            death_rates=log_death_rates,
            ages=ages,
            times=times,
            log_scale_params=True,
        )

        birth_rates = jnp.exp(log_birth_rates)
        death_rates = jnp.exp(log_death_rates)
        expected = likelihood.calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ns=jnp.array([1], dtype=jnp.int32),
            initial_population=1,
        ) + _expected_unknown_spine_age_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        assert bool(jnp.isclose(result, expected))

    def test_log_likelihood_spine_ages(self) -> None:
        """Test log_likelihood with spine_ages provided."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        ages = jnp.array([1.0], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)
        spine_ages = jnp.array([0.5], dtype=jnp.float64)

        result = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
            spine_ages=spine_ages,
        )

        expected = (
            likelihood.calculate_populations_log_lik(
                birth_rates=birth_rates,
                death_rates=death_rates,
                times=times,
                ns=jnp.array([2], dtype=jnp.int32),
                initial_population=1,
            )
            + calculate_spine_ages_log_lik(
                birth_rates,
                death_rates,
                ages=spine_ages,
                times=times,
            )
            + calculate_non_spine_ages_log_lik(
                birth_rates,
                death_rates,
                ages,
                times=times,
            )
        )

        assert bool(jnp.isclose(result, expected))

    def test_log_likelihood_is_ordered(self) -> None:
        """Test log_likelihood with is_ordered=True."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        ages = jnp.array([1.0, 0.5], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)

        result = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
            is_ordered=True,
        )

        expected = (
            likelihood.calculate_populations_log_lik(
                birth_rates=birth_rates,
                death_rates=death_rates,
                times=times,
                ns=jnp.array([2], dtype=jnp.int32),
                initial_population=1,
            )
            + calculate_spine_ages_log_lik(
                birth_rates,
                death_rates,
                ages=jnp.array([1.0], dtype=jnp.float64),
                times=times,
            )
            + calculate_non_spine_ages_log_lik(
                birth_rates,
                death_rates,
                jnp.array([0.5], dtype=jnp.float64),
                times=times,
            )
        )

        assert bool(jnp.isclose(result, expected))

    def test_log_likelihood_value_errors(self) -> None:
        """Test log_likelihood raises ValueError for invalid argument combinations."""
        # Both is_ordered and spine_ages
        with pytest.raises(ValueError):
            likelihood.log_likelihood(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                ages=jnp.array([1.0], dtype=jnp.float64),
                times=jnp.array([2.0], dtype=jnp.float64),
                is_ordered=True,
                spine_ages=jnp.array([0.5], dtype=jnp.float64),
            )
        with pytest.raises(ValueError):
            likelihood.log_likelihood(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                ages=jnp.array([1.0], dtype=jnp.float64),
                times=jnp.array([0.0], dtype=jnp.float64),
            )
        # ages out of range
        with pytest.raises(ValueError):
            likelihood.log_likelihood(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                ages=jnp.array([-1.0], dtype=jnp.float64),
                times=jnp.array([2.0], dtype=jnp.float64),
            )

    def test_negative_log_likelihood(self) -> None:
        """Test negative_log_likelihood returns negated log_likelihood value."""
        kwargs = {
            "birth_rates": jnp.array([0.5], dtype=jnp.float64),
            "death_rates": jnp.array([0.2], dtype=jnp.float64),
            "ages": jnp.array([1.0], dtype=jnp.float64),
            "times": jnp.array([2.0], dtype=jnp.float64),
        }
        result = likelihood.negative_log_likelihood(**kwargs)
        expected = -likelihood.log_likelihood(**kwargs)
        assert bool(jnp.isclose(result, expected))

    def test_unknown_spine_mixture_age_density_matches_manual_formula(self) -> None:
        """Test the exported mixture-density helper matches the manual formula."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        age_grid = jnp.array([0.0, 0.5, 1.5, 2.5], dtype=jnp.float64)
        times = jnp.array([3.0], dtype=jnp.float64)
        num_individuals = 4

        result = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_individuals=num_individuals,
        )

        expected_values = []
        for age in age_grid:
            age_tensor = jnp.array([float(age)], dtype=jnp.float64)
            non_spine_density = jnp.exp(
                calculate_non_spine_ages_log_lik(
                    birth_rates,
                    death_rates,
                    age_tensor,
                    times=times,
                ),
            )
            spine_density = jnp.exp(
                calculate_spine_ages_log_lik(
                    birth_rates,
                    death_rates,
                    age_tensor,
                    times=times,
                ),
            )
            expected_values.append(
                ((num_individuals - 1) * non_spine_density + spine_density)
                / num_individuals,
            )

        expected = jnp.stack(expected_values)

        assert bool(jnp.allclose(result, expected))

    def test_unknown_spine_mixture_age_density_rejects_nonpositive_n(self) -> None:
        """Test the helper rejects invalid individual counts."""
        with pytest.raises(ValueError):
            calculate_unknown_spine_mixture_age_density(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                age_grid=jnp.array([0.5], dtype=jnp.float64),
                times=jnp.array([3.0], dtype=jnp.float64),
                num_individuals=0,
            )

    def test_log_likelihood_unknown_spine_formula_equivalence(self) -> None:
        """Default branch should match unknown-spine log-sum-exp formula."""
        birth_rates = jnp.array([0.7], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        ages = jnp.array([0.2, 0.4, 0.9], dtype=jnp.float64)
        times = jnp.array([1.5], dtype=jnp.float64)

        observed = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        expected = likelihood.calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ns=jnp.array([3], dtype=jnp.int32),
            initial_population=1,
        ) + _expected_unknown_spine_age_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        assert bool(jnp.isclose(observed, expected))

    def test_log_likelihood_unknown_spine_single_age_reduces_to_spine(self) -> None:
        """With one age, unknown-spine branch should equal explicit spine likelihood."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        ages = jnp.array([0.5], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)

        unknown_spine_result = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        explicit_spine_result = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=jnp.array([], dtype=jnp.float64),
            times=times,
            spine_ages=ages,
        )

        assert bool(jnp.isclose(unknown_spine_result, explicit_spine_result))

    def test_log_likelihood_unknown_spine_differs_from_all_non_spine(self) -> None:
        """Unknown-spine branch should not collapse to all-non-spine in general."""
        birth_rates = jnp.array([0.6], dtype=jnp.float64)
        death_rates = jnp.array([0.1], dtype=jnp.float64)
        ages = jnp.array([0.1, 0.3, 0.8], dtype=jnp.float64)
        times = jnp.array([1.0], dtype=jnp.float64)

        observed = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        all_non_spine = likelihood.calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ns=jnp.array([3], dtype=jnp.int32),
            initial_population=1,
        ) + calculate_non_spine_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times=times,
        )

        assert not bool(jnp.isclose(observed, all_non_spine))

    def test_log_likelihood_terminal_age_auto_assigned_as_spine(self) -> None:
        """An age at final time should be treated as spine when spine_ages is omitted."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)
        ages = jnp.array([0.7, 2.0], dtype=jnp.float64)

        observed = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        expected = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=jnp.array([0.7], dtype=jnp.float64),
            spine_ages=jnp.array([2.0], dtype=jnp.float64),
            times=times,
        )

        assert bool(jnp.isclose(observed, expected))

    def test_log_likelihood_multiple_terminal_ages_raise_value_error(self) -> None:
        """Multiple ages at final time are invalid under unique-spine assumption."""
        with pytest.raises(ValueError, match="More than one age equals"):
            likelihood.log_likelihood(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                ages=jnp.array([2.0, 2.0, 0.5], dtype=jnp.float64),
                times=jnp.array([2.0], dtype=jnp.float64),
            )

    # =====================================================================
    # Tests for calculate_unknown_spine_mixture_birth_density
    # =====================================================================

    def test_birth_density_reverses_age_density(self) -> None:
        """Test that birth_density(b) equals age_density(T - b) with Jacobian = 1."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        times = jnp.array([3.0], dtype=jnp.float64)
        num_individuals = 5

        T = float(times[-1])
        birth_time_grid = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64)

        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        age_grid = T - birth_time_grid
        age_density = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_individuals=num_individuals,
        )

        # age_grid[i] = T - birth_time_grid[i], so age_density[i] == birth_density[i]
        assert bool(jnp.allclose(birth_density, age_density, rtol=1e-6))

    def test_birth_density_single_point_matches_age_density(self) -> None:
        """Test a single point: birth_density(b) == age_density(T - b)."""
        birth_rates = jnp.array([0.6], dtype=jnp.float64)
        death_rates = jnp.array([0.15], dtype=jnp.float64)
        times = jnp.array([2.5], dtype=jnp.float64)
        num_individuals = 3
        T = float(times[-1])

        b = 1.0
        birth_time_grid = jnp.array([b], dtype=jnp.float64)
        age = T - b
        age_grid = jnp.array([age], dtype=jnp.float64)

        birth_dens = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        age_dens = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_individuals=num_individuals,
        )

        assert bool(jnp.isclose(birth_dens[0], age_dens[0], rtol=1e-6))

    def test_birth_density_rejects_negative_birth_times(self) -> None:
        """Test that birth density rejects birth times < 0."""
        with pytest.raises(ValueError, match="Birth times must be in the range"):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                birth_time_grid=jnp.array([-0.5], dtype=jnp.float64),
                times=jnp.array([3.0], dtype=jnp.float64),
                num_individuals=5,
            )

    def test_birth_density_rejects_birth_times_beyond_T(self) -> None:
        """Test that birth density rejects birth times > T."""
        T = 2.0
        with pytest.raises(ValueError, match="Birth times must be in the range"):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                birth_time_grid=jnp.array([T + 0.5], dtype=jnp.float64),
                times=jnp.array([T], dtype=jnp.float64),
                num_individuals=5,
            )

    def test_birth_density_rejects_nonpositive_num_individuals(self) -> None:
        """Test that birth density rejects invalid individual counts."""
        with pytest.raises(
            ValueError, match="num_individuals must be a positive integer"
        ):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                birth_time_grid=jnp.array([1.0], dtype=jnp.float64),
                times=jnp.array([3.0], dtype=jnp.float64),
                num_individuals=0,
            )

        with pytest.raises(
            ValueError, match="num_individuals must be a positive integer"
        ):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=jnp.array([0.5], dtype=jnp.float64),
                death_rates=jnp.array([0.2], dtype=jnp.float64),
                birth_time_grid=jnp.array([1.0], dtype=jnp.float64),
                times=jnp.array([3.0], dtype=jnp.float64),
                num_individuals=-3,
            )

    def test_birth_density_non_negative(self) -> None:
        """Test that birth density is always non-negative."""
        birth_rates = jnp.array([0.4], dtype=jnp.float64)
        death_rates = jnp.array([0.25], dtype=jnp.float64)
        times = jnp.array([4.0], dtype=jnp.float64)
        num_individuals = 8

        birth_time_grid = jnp.linspace(0, float(times[-1]), 20, dtype=jnp.float64)

        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        assert bool(jnp.all(birth_density >= 0)), "Birth density should be non-negative"

    def test_birth_density_shape_matches_grid(self) -> None:
        """Test that output shape matches input grid shape."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        times = jnp.array([3.0], dtype=jnp.float64)
        num_individuals = 5

        for grid_size in [1, 5, 10, 50]:
            birth_time_grid = jnp.linspace(
                0,
                float(times[-1]),
                grid_size,
                dtype=jnp.float64,
            )

            birth_density = calculate_unknown_spine_mixture_birth_density(
                birth_rates=birth_rates,
                death_rates=death_rates,
                birth_time_grid=birth_time_grid,
                times=times,
                num_individuals=num_individuals,
            )

            assert birth_density.shape == birth_time_grid.shape, (
                f"Shape mismatch for grid_size={grid_size}"
            )

    def test_birth_density_deterministic(self) -> None:
        """Test that calling with the same inputs produces identical results."""
        birth_rates = jnp.array([0.55], dtype=jnp.float64)
        death_rates = jnp.array([0.18], dtype=jnp.float64)
        birth_time_grid = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)
        num_individuals = 4

        result1 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        result2 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        assert bool(jnp.allclose(result1, result2))

    def test_birth_density_with_single_individual(self) -> None:
        """Test birth density with num_individuals=1 (degenerate case)."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        birth_time_grid = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)

        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=1,
        )

        assert birth_density.shape == birth_time_grid.shape
        assert bool(jnp.all(birth_density >= 0))

    def test_birth_density_consistency_across_different_ranges(self) -> None:
        """Test birth density consistency when evaluated at same points over different time ranges."""
        birth_rates = jnp.array([0.6], dtype=jnp.float64)
        death_rates = jnp.array([0.3], dtype=jnp.float64)
        num_individuals = 5

        times1 = jnp.array([4.0], dtype=jnp.float64)
        birth_time_grid1 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

        times2 = jnp.array([2.0], dtype=jnp.float64)
        birth_time_grid2 = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)

        result1 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid1,
            times=times1,
            num_individuals=num_individuals,
        )

        result2 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid2,
            times=times2,
            num_individuals=num_individuals,
        )

        assert bool(jnp.all(result1 >= 0))
        assert bool(jnp.all(result2 >= 0))
        assert result1.shape == birth_time_grid1.shape
        assert result2.shape == birth_time_grid2.shape

    def test_birth_density_boundary_values(self) -> None:
        """Test birth density at boundary points b=0 and b=T."""
        birth_rates = jnp.array([0.5], dtype=jnp.float64)
        death_rates = jnp.array([0.2], dtype=jnp.float64)
        T = 3.0
        times = jnp.array([T], dtype=jnp.float64)
        num_individuals = 6

        birth_time_grid = jnp.array([0.0, T], dtype=jnp.float64)

        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        assert bool(jnp.all(jnp.isfinite(birth_density)))
        assert bool(jnp.all(birth_density >= 0))

    def test_birth_density_scalar_vs_tensor_rates(self) -> None:
        """Test that birth density works with both scalar and tensor rate inputs."""
        birth_time_grid = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        times = jnp.array([2.0], dtype=jnp.float64)
        num_individuals = 4

        result_scalar = calculate_unknown_spine_mixture_birth_density(
            birth_rates=0.5,
            death_rates=0.2,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        result_tensor = calculate_unknown_spine_mixture_birth_density(
            birth_rates=jnp.array([0.5], dtype=jnp.float64),
            death_rates=jnp.array([0.2], dtype=jnp.float64),
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=num_individuals,
        )

        assert bool(jnp.allclose(result_scalar, result_tensor))


class TestMixtureDensityNormalization:
    """Verify that mixture age densities integrate to approximately 1."""

    def test_unknown_spine_mixture_age_density_integrates_to_one(self) -> None:
        birth_rates = jnp.array([0.1], dtype=jnp.float64)
        death_rates = jnp.array([0.05], dtype=jnp.float64)
        T = 100.0
        times = jnp.array([T], dtype=jnp.float64)
        n_grid = 500
        age_grid = jnp.linspace(0.0, T, n_grid, dtype=jnp.float64)

        density = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_individuals=5,
        )

        assert density.shape == (n_grid,)
        assert bool(jnp.all(density >= 0.0))
        integral = float(jnp.trapezoid(density, age_grid))
        assert 0.7 < integral < 1.3

    def test_unknown_spine_mixture_birth_density_change_of_variables(self) -> None:
        birth_rates = jnp.array([0.1], dtype=jnp.float64)
        death_rates = jnp.array([0.05], dtype=jnp.float64)
        T = 100.0
        times = jnp.array([T], dtype=jnp.float64)
        n_grid = 200
        age_grid = jnp.linspace(1.0, T - 1.0, n_grid, dtype=jnp.float64)
        birth_time_grid = T - age_grid

        density_age = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_individuals=3,
        )
        density_birth = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_individuals=3,
        )
        assert bool(jnp.allclose(density_age, density_birth, atol=1e-8))
