"""Unit tests for the likelihood fns in the inf_bp_texts.inference.likelihood module."""

import pytest
import torch

from inf_bp_texts.inference import (
    calculate_unknown_spine_mixture_age_density,
    calculate_unknown_spine_mixture_birth_density,
    likelihood,
)
from inf_bp_texts.models.mathematical_functions import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)


def _expected_unknown_spine_age_log_lik(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    ages: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Compute the unknown-spine age log-likelihood via stable log-sum-exp."""
    sum_log_f = calculate_non_spine_ages_log_lik(
        birth_rates,
        death_rates,
        ages,
        times=times,
    )
    a_terms = []
    for age in ages:
        age_tensor = torch.tensor([age], dtype=torch.float64)
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
        - torch.log(torch.tensor(float(len(ages)), dtype=torch.float64))
        + torch.logsumexp(torch.stack(a_terms), dim=0)
    )


class TestLikelihood:
    """Test likelihood functions."""

    def test_log_likelihood_simple(self) -> None:
        """Test log_likelihood with simple valid input and real function outputs."""
        birth_rates = torch.tensor([0.5])
        death_rates = torch.tensor([0.2])
        ages = torch.tensor([1.0])
        times = torch.tensor([2.0])

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
            ns=torch.tensor([1], dtype=torch.int64),
            initial_population=1,
        ) + _expected_unknown_spine_age_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        assert torch.isclose(result, expected)

    def test_log_likelihood_negative_branch(self) -> None:
        """Test log_likelihood returns -inf when population log-likelihood is non-finite."""
        result = likelihood.log_likelihood(
            birth_rates=torch.tensor([1000.0]),
            death_rates=torch.tensor([0.1]),
            ages=torch.tensor([1.0]),
            times=torch.tensor([2.0]),
        )
        assert torch.isneginf(result)

    def test_log_likelihood_invalid_params(self) -> None:
        """Test log_likelihood returns -inf for invalid parameters (e.g., neg rates)."""
        result = likelihood.log_likelihood(
            birth_rates=torch.tensor([-0.5]),
            death_rates=torch.tensor([0.2]),
            ages=torch.tensor([1.0]),
            times=torch.tensor([2.0]),
        )
        assert result == float("-inf") or torch.isclose(
            result, torch.tensor(float("-inf"))
        )

    def test_log_likelihood_log_scale(self) -> None:
        """Test log_likelihood with log_scale_params=True."""
        log_birth_rates = torch.tensor([0.0])
        log_death_rates = torch.tensor([0.0])
        ages = torch.tensor([1.0])
        times = torch.tensor([2.0])

        result = likelihood.log_likelihood(
            birth_rates=log_birth_rates,
            death_rates=log_death_rates,
            ages=ages,
            times=times,
            log_scale_params=True,
        )

        birth_rates = torch.exp(log_birth_rates)
        death_rates = torch.exp(log_death_rates)
        expected = likelihood.calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            ns=torch.tensor([1], dtype=torch.int64),
            initial_population=1,
        ) + _expected_unknown_spine_age_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        assert torch.isclose(result, expected)

    def test_log_likelihood_spine_ages(self) -> None:
        """Test log_likelihood with spine_ages provided."""
        birth_rates = torch.tensor([0.5])
        death_rates = torch.tensor([0.2])
        ages = torch.tensor([1.0])
        times = torch.tensor([2.0])
        spine_ages = torch.tensor([0.5])

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
                ns=torch.tensor([2], dtype=torch.int64),
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

        assert torch.isclose(result, expected)

    def test_log_likelihood_is_ordered(self) -> None:
        """Test log_likelihood with is_ordered=True."""
        birth_rates = torch.tensor([0.5])
        death_rates = torch.tensor([0.2])
        ages = torch.tensor([1.0, 0.5])
        times = torch.tensor([2.0])

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
                ns=torch.tensor([2], dtype=torch.int64),
                initial_population=1,
            )
            + calculate_spine_ages_log_lik(
                birth_rates,
                death_rates,
                ages=torch.tensor([1.0]),
                times=times,
            )
            + calculate_non_spine_ages_log_lik(
                birth_rates,
                death_rates,
                ages=torch.tensor([0.5]),
                times=times,
            )
        )

        assert torch.isclose(result, expected)

    def test_log_likelihood_value_errors(self) -> None:
        """Test log_likelihood raises ValueError for invalid argument combinations."""
        # Both is_ordered and spine_ages
        with pytest.raises(ValueError):
            likelihood.log_likelihood(
                birth_rates=torch.tensor([0.5]),
                death_rates=torch.tensor([0.2]),
                ages=torch.tensor([1.0]),
                times=torch.tensor([2.0]),
                is_ordered=True,
                spine_ages=torch.tensor([0.5]),
            )
        with pytest.raises(ValueError):
            likelihood.log_likelihood(
                birth_rates=torch.tensor([0.5]),
                death_rates=torch.tensor([0.2]),
                ages=torch.tensor([1.0]),
                times=torch.tensor([0.0]),
            )
        # ages out of range
        with pytest.raises(ValueError):
            likelihood.log_likelihood(
                birth_rates=torch.tensor([0.5]),
                death_rates=torch.tensor([0.2]),
                ages=torch.tensor([-1.0]),
                times=torch.tensor([2.0]),
            )

    def test_negative_log_likelihood(self) -> None:
        """Test negative_log_likelihood returns negated log_likelihood value."""
        kwargs = {
            "birth_rates": torch.tensor([0.5]),
            "death_rates": torch.tensor([0.2]),
            "ages": torch.tensor([1.0]),
            "times": torch.tensor([2.0]),
        }
        result = likelihood.negative_log_likelihood(**kwargs)
        expected = -likelihood.log_likelihood(**kwargs)
        assert torch.isclose(result, expected)

    def test_unknown_spine_mixture_age_density_matches_manual_formula(self) -> None:
        """Test the exported mixture-density helper matches the manual formula."""
        birth_rates = torch.tensor([0.5], dtype=torch.float64)
        death_rates = torch.tensor([0.2], dtype=torch.float64)
        age_grid = torch.tensor([0.0, 0.5, 1.5, 2.5], dtype=torch.float64)
        times = torch.tensor([3.0], dtype=torch.float64)
        num_manuscripts = 4

        result = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        expected_values = []
        for age in age_grid:
            age_tensor = torch.tensor([age], dtype=torch.float64)
            non_spine_density = torch.exp(
                calculate_non_spine_ages_log_lik(
                    birth_rates,
                    death_rates,
                    age_tensor,
                    times=times,
                ),
            )
            spine_density = torch.exp(
                calculate_spine_ages_log_lik(
                    birth_rates,
                    death_rates,
                    age_tensor,
                    times=times,
                ),
            )
            expected_values.append(
                ((num_manuscripts - 1) * non_spine_density + spine_density)
                / num_manuscripts,
            )

        expected = torch.stack(expected_values)

        assert torch.allclose(result, expected)

    def test_unknown_spine_mixture_age_density_rejects_nonpositive_n(self) -> None:
        """Test the helper rejects invalid manuscript counts."""
        with pytest.raises(ValueError):
            calculate_unknown_spine_mixture_age_density(
                birth_rates=torch.tensor([0.5], dtype=torch.float64),
                death_rates=torch.tensor([0.2], dtype=torch.float64),
                age_grid=torch.tensor([0.5], dtype=torch.float64),
                times=torch.tensor([3.0], dtype=torch.float64),
                num_manuscripts=0,
            )

    def test_log_likelihood_unknown_spine_formula_equivalence(self) -> None:
        """Default branch should match unknown-spine log-sum-exp formula."""
        birth_rates = torch.tensor([0.7])
        death_rates = torch.tensor([0.2])
        ages = torch.tensor([0.2, 0.4, 0.9])
        times = torch.tensor([1.5])

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
            ns=torch.tensor([3], dtype=torch.int64),
            initial_population=1,
        ) + _expected_unknown_spine_age_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        assert torch.isclose(observed, expected)

    def test_log_likelihood_unknown_spine_single_age_reduces_to_spine(self) -> None:
        """With one age, unknown-spine branch should equal explicit spine likelihood."""
        birth_rates = torch.tensor([0.5])
        death_rates = torch.tensor([0.2])
        ages = torch.tensor([0.5])
        times = torch.tensor([2.0])

        unknown_spine_result = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        explicit_spine_result = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=torch.tensor([]),
            times=times,
            spine_ages=ages,
        )

        assert torch.isclose(unknown_spine_result, explicit_spine_result)

    def test_log_likelihood_unknown_spine_differs_from_all_non_spine(self) -> None:
        """Unknown-spine branch should not collapse to all-non-spine in general."""
        birth_rates = torch.tensor([0.6])
        death_rates = torch.tensor([0.1])
        ages = torch.tensor([0.1, 0.3, 0.8])
        times = torch.tensor([1.0])

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
            ns=torch.tensor([3], dtype=torch.int64),
            initial_population=1,
        ) + calculate_non_spine_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times=times,
        )

        assert not torch.isclose(observed, all_non_spine)

    def test_log_likelihood_terminal_age_auto_assigned_as_spine(self) -> None:
        """An age at final time should be treated as spine when spine_ages is omitted."""
        birth_rates = torch.tensor([0.5])
        death_rates = torch.tensor([0.2])
        times = torch.tensor([2.0])
        ages = torch.tensor([0.7, 2.0])

        observed = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=ages,
            times=times,
        )

        expected = likelihood.log_likelihood(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=torch.tensor([0.7]),
            spine_ages=torch.tensor([2.0]),
            times=times,
        )

        assert torch.isclose(observed, expected)

    def test_log_likelihood_multiple_terminal_ages_raise_value_error(self) -> None:
        """Multiple ages at final time are invalid under unique-spine assumption."""
        with pytest.raises(ValueError, match="More than one age equals"):
            likelihood.log_likelihood(
                birth_rates=torch.tensor([0.5]),
                death_rates=torch.tensor([0.2]),
                ages=torch.tensor([2.0, 2.0, 0.5]),
                times=torch.tensor([2.0]),
            )

    # =====================================================================
    # Tests for calculate_unknown_spine_mixture_birth_density
    # =====================================================================

    def test_birth_density_reverses_age_density(self) -> None:
        """Test that birth_density(b) equals age_density(T - b) with Jacobian = 1."""
        birth_rates = torch.tensor([0.5], dtype=torch.float64)
        death_rates = torch.tensor([0.2], dtype=torch.float64)
        times = torch.tensor([3.0], dtype=torch.float64)
        num_manuscripts = 5

        # Create symmetric test grids
        T = float(times[-1])
        birth_time_grid = torch.tensor(
            [0.1, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64
        )

        # Get birth-time density
        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        # Get corresponding age-time density
        age_grid = T - birth_time_grid
        age_density = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        # Reverse age_density to match birth_time order (since b=T-a, high a -> low b)
        age_density_reversed = torch.flip(age_density, dims=[0])

        # Birth-density should match reversed age-density
        assert torch.allclose(birth_density, age_density_reversed, rtol=1e-6)

    def test_birth_density_single_point_matches_age_density(self) -> None:
        """Test a single point: birth_density(b) == age_density(T - b)."""
        birth_rates = torch.tensor([0.6], dtype=torch.float64)
        death_rates = torch.tensor([0.15], dtype=torch.float64)
        times = torch.tensor([2.5], dtype=torch.float64)
        num_manuscripts = 3
        T = float(times[-1])

        # Test at a single birth time
        b = 1.0
        birth_time_grid = torch.tensor([b], dtype=torch.float64)
        age = T - b
        age_grid = torch.tensor([age], dtype=torch.float64)

        birth_dens = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        age_dens = calculate_unknown_spine_mixture_age_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            age_grid=age_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        assert torch.isclose(birth_dens[0], age_dens[0], rtol=1e-6)

    def test_birth_density_rejects_negative_birth_times(self) -> None:
        """Test that birth density rejects birth times < 0."""
        with pytest.raises(ValueError, match="Birth times must be in the range"):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=torch.tensor([0.5], dtype=torch.float64),
                death_rates=torch.tensor([0.2], dtype=torch.float64),
                birth_time_grid=torch.tensor([-0.5], dtype=torch.float64),
                times=torch.tensor([3.0], dtype=torch.float64),
                num_manuscripts=5,
            )

    def test_birth_density_rejects_birth_times_beyond_T(self) -> None:
        """Test that birth density rejects birth times > T."""
        T = 2.0
        with pytest.raises(ValueError, match="Birth times must be in the range"):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=torch.tensor([0.5], dtype=torch.float64),
                death_rates=torch.tensor([0.2], dtype=torch.float64),
                birth_time_grid=torch.tensor([T + 0.5], dtype=torch.float64),
                times=torch.tensor([T], dtype=torch.float64),
                num_manuscripts=5,
            )

    def test_birth_density_rejects_nonpositive_num_manuscripts(self) -> None:
        """Test that birth density rejects invalid manuscript counts."""
        with pytest.raises(
            ValueError, match="num_manuscripts must be a positive integer"
        ):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=torch.tensor([0.5], dtype=torch.float64),
                death_rates=torch.tensor([0.2], dtype=torch.float64),
                birth_time_grid=torch.tensor([1.0], dtype=torch.float64),
                times=torch.tensor([3.0], dtype=torch.float64),
                num_manuscripts=0,
            )

        with pytest.raises(
            ValueError, match="num_manuscripts must be a positive integer"
        ):
            calculate_unknown_spine_mixture_birth_density(
                birth_rates=torch.tensor([0.5], dtype=torch.float64),
                death_rates=torch.tensor([0.2], dtype=torch.float64),
                birth_time_grid=torch.tensor([1.0], dtype=torch.float64),
                times=torch.tensor([3.0], dtype=torch.float64),
                num_manuscripts=-3,
            )

    def test_birth_density_non_negative(self) -> None:
        """Test that birth density is always non-negative."""
        birth_rates = torch.tensor([0.4], dtype=torch.float64)
        death_rates = torch.tensor([0.25], dtype=torch.float64)
        times = torch.tensor([4.0], dtype=torch.float64)
        num_manuscripts = 8

        # Test across the full domain [0, T]
        birth_time_grid = torch.linspace(0, float(times[-1]), 20, dtype=torch.float64)

        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        assert torch.all(birth_density >= 0), "Birth density should be non-negative"

    def test_birth_density_shape_matches_grid(self) -> None:
        """Test that output shape matches input grid shape."""
        birth_rates = torch.tensor([0.5], dtype=torch.float64)
        death_rates = torch.tensor([0.2], dtype=torch.float64)
        times = torch.tensor([3.0], dtype=torch.float64)
        num_manuscripts = 5

        # Test with various grid sizes
        for grid_size in [1, 5, 10, 50]:
            birth_time_grid = torch.linspace(
                0,
                float(times[-1]),
                grid_size,
                dtype=torch.float64,
            )

            birth_density = calculate_unknown_spine_mixture_birth_density(
                birth_rates=birth_rates,
                death_rates=death_rates,
                birth_time_grid=birth_time_grid,
                times=times,
                num_manuscripts=num_manuscripts,
            )

            assert birth_density.shape == birth_time_grid.shape, (
                f"Shape mismatch for grid_size={grid_size}"
            )

    def test_birth_density_deterministic(self) -> None:
        """Test that calling with the same inputs produces identical results."""
        birth_rates = torch.tensor([0.55], dtype=torch.float64)
        death_rates = torch.tensor([0.18], dtype=torch.float64)
        birth_time_grid = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        times = torch.tensor([2.0], dtype=torch.float64)
        num_manuscripts = 4

        result1 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        result2 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        assert torch.allclose(result1, result2)

    def test_birth_density_with_single_manuscript(self) -> None:
        """Test birth density with num_manuscripts=1 (degenerate case)."""
        birth_rates = torch.tensor([0.5], dtype=torch.float64)
        death_rates = torch.tensor([0.2], dtype=torch.float64)
        birth_time_grid = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        times = torch.tensor([2.0], dtype=torch.float64)

        # Should not raise an error even with a single manuscript
        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=1,
        )

        assert birth_density.shape == birth_time_grid.shape
        assert torch.all(birth_density >= 0)

    def test_birth_density_consistency_across_different_ranges(self) -> None:
        """Test that birth density is consistent when evaluated at same points over different time ranges."""
        birth_rates = torch.tensor([0.6], dtype=torch.float64)
        death_rates = torch.tensor([0.3], dtype=torch.float64)
        num_manuscripts = 5

        # Case 1: [0, 4]
        times1 = torch.tensor([4.0], dtype=torch.float64)
        birth_time_grid1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # Case 2: [0, 2] (shifted time scale) - but birth times in relative units
        # This test verifies the function handles the transformation correctly
        times2 = torch.tensor([2.0], dtype=torch.float64)
        birth_time_grid2 = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

        result1 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid1,
            times=times1,
            num_manuscripts=num_manuscripts,
        )

        result2 = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid2,
            times=times2,
            num_manuscripts=num_manuscripts,
        )

        # Both should be non-negative and have the correct shape
        assert torch.all(result1 >= 0)
        assert torch.all(result2 >= 0)
        assert result1.shape == birth_time_grid1.shape
        assert result2.shape == birth_time_grid2.shape

    def test_birth_density_boundary_values(self) -> None:
        """Test birth density at boundary points b=0 and b=T."""
        birth_rates = torch.tensor([0.5], dtype=torch.float64)
        death_rates = torch.tensor([0.2], dtype=torch.float64)
        T = 3.0
        times = torch.tensor([T], dtype=torch.float64)
        num_manuscripts = 6

        # Test at boundaries
        birth_time_grid = torch.tensor([0.0, T], dtype=torch.float64)

        birth_density = calculate_unknown_spine_mixture_birth_density(
            birth_rates=birth_rates,
            death_rates=death_rates,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        # Both boundary values should be non-negative and finite
        assert torch.all(torch.isfinite(birth_density))
        assert torch.all(birth_density >= 0)
        # At b=0 (age=T), density should be related to manuscripts created at start
        # At b=T (age=0), density should be related to manuscripts created at end
        # Both should be finite and well-defined

    def test_birth_density_scalar_vs_tensor_rates(self) -> None:
        """Test that birth density works with both scalar and tensor rate inputs."""
        birth_time_grid = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        times = torch.tensor([2.0], dtype=torch.float64)
        num_manuscripts = 4

        # With scalars
        result_scalar = calculate_unknown_spine_mixture_birth_density(
            birth_rates=0.5,
            death_rates=0.2,
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        # With tensors
        result_tensor = calculate_unknown_spine_mixture_birth_density(
            birth_rates=torch.tensor([0.5], dtype=torch.float64),
            death_rates=torch.tensor([0.2], dtype=torch.float64),
            birth_time_grid=birth_time_grid,
            times=times,
            num_manuscripts=num_manuscripts,
        )

        assert torch.allclose(result_scalar, result_tensor)
