"""Simple normalization tests for birth-death probability functions."""

# ruff: noqa: S101

from collections.abc import Callable

import torch

from inf_bp_texts.inference import calculate_unknown_spine_mixture_age_density
from inf_bp_texts.models.mathematical_functions import (
    calculate_extinction_pr,
    calculate_non_spine_ages_log_lik,
    calculate_populations_log_lik,
    calculate_spine_ages_log_lik,
)


def _pointwise_density_from_loglik(
    loglik_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    times: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Evaluate exp(log-likelihood) at each point in a 1D grid."""
    vals = []
    for age in grid:
        log_val = loglik_fn(
            birth_rates,
            death_rates,
            torch.tensor([age], dtype=torch.float64),
            times,
        )
        vals.append(torch.exp(log_val))
    return torch.stack(vals)


def test_population_probability_mass_sums_to_one() -> None:
    """Population probabilities p_n(T) should sum to 1 over n."""
    birth_rates = torch.tensor([0.2], dtype=torch.float64)
    death_rates = torch.tensor([0.5], dtype=torch.float64)
    times = torch.tensor([2.0], dtype=torch.float64)

    probs = []
    for n in range(60):
        log_p_n = calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ns=torch.tensor([n], dtype=torch.int64),
            times=times,
            initial_population=1,
        )
        probs.append(torch.exp(log_p_n))

    total_prob = torch.sum(torch.stack(probs))
    assert torch.isclose(total_prob, torch.tensor(1.0, dtype=torch.float64), atol=1e-4)


def test_non_spine_age_density_integrates_to_one() -> None:
    """Non-spine age density should integrate to 1 on [0, T]."""
    birth_rates = torch.tensor([0.6], dtype=torch.float64)
    death_rates = torch.tensor([0.2], dtype=torch.float64)
    times = torch.tensor([3.0], dtype=torch.float64)

    grid = torch.linspace(0.0, float(times[-1].item()), steps=4000, dtype=torch.float64)
    density_vals = _pointwise_density_from_loglik(
        calculate_non_spine_ages_log_lik,
        birth_rates,
        death_rates,
        times,
        grid,
    )
    integral = torch.trapz(density_vals, grid)

    assert torch.isclose(integral, torch.tensor(1.0, dtype=torch.float64), atol=2e-3)


def test_spine_age_distribution_integrates_to_one_with_terminal_mass() -> None:
    """Spine distribution has continuous mass on [0, T) plus an atom at T."""
    birth_rates = torch.tensor([0.6], dtype=torch.float64)
    death_rates = torch.tensor([0.2], dtype=torch.float64)
    times = torch.tensor([3.0], dtype=torch.float64)

    t_end = float(times[-1].item())
    eps = 1e-5
    grid = torch.linspace(0.0, t_end - eps, steps=4000, dtype=torch.float64)

    cont_density_vals = _pointwise_density_from_loglik(
        calculate_spine_ages_log_lik,
        birth_rates,
        death_rates,
        times,
        grid,
    )
    continuous_mass = torch.trapz(cont_density_vals, grid)

    terminal_mass = torch.exp(
        calculate_spine_ages_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=torch.tensor([t_end], dtype=torch.float64),
            times=times,
        ),
    )

    total_mass = continuous_mass + terminal_mass
    assert torch.isclose(total_mass, torch.tensor(1.0, dtype=torch.float64), atol=2e-3)


def test_extinction_probability_matches_population_zero_mass() -> None:
    """Pr(Z_T = 0) should equal p_0(T) from population probabilities."""
    birth_rates = torch.tensor([0.4], dtype=torch.float64)
    death_rates = torch.tensor([0.25], dtype=torch.float64)
    times = torch.tensor([2.5], dtype=torch.float64)

    extinction_pr = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
    )
    p0 = torch.exp(
        calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ns=torch.tensor([0], dtype=torch.int64),
            times=times,
            initial_population=1,
        ),
    )

    assert torch.isclose(extinction_pr, p0, atol=1e-12)


def test_unknown_spine_mixture_distribution_integrates_to_one() -> None:
    """Unknown-spine mixture has continuous mass plus spine atom / n at T."""
    birth_rates = torch.tensor([0.6], dtype=torch.float64)
    death_rates = torch.tensor([0.2], dtype=torch.float64)
    times = torch.tensor([3.0], dtype=torch.float64)
    n = 5

    t_end = float(times[-1].item())
    eps = 1e-5
    grid = torch.linspace(0.0, t_end - eps, steps=4000, dtype=torch.float64)

    mixture_cont = calculate_unknown_spine_mixture_age_density(
        birth_rates=birth_rates,
        death_rates=death_rates,
        age_grid=grid,
        times=times,
        num_manuscripts=n,
    )
    continuous_mass = torch.trapz(mixture_cont, grid)

    spine_terminal_mass = torch.exp(
        calculate_spine_ages_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ages=torch.tensor([t_end], dtype=torch.float64),
            times=times,
        ),
    )
    total_mass = continuous_mass + spine_terminal_mass

    assert torch.isclose(total_mass, torch.tensor(1.0, dtype=torch.float64), atol=2e-3)


def test_population_probability_mass_sums_to_one_multi_epoch() -> None:
    """Multi-epoch population probabilities p_n(T) should sum to 1."""
    birth_rates = torch.tensor([0.15, 0.10, 0.08], dtype=torch.float64)
    death_rates = torch.tensor([0.25, 0.20, 0.18], dtype=torch.float64)
    times = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    probs = []
    for n in range(120):
        log_p_n = calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ns=torch.tensor([n], dtype=torch.int64),
            times=times,
            initial_population=1,
        )
        probs.append(torch.exp(log_p_n))

    total_prob = torch.sum(torch.stack(probs))
    assert torch.isclose(total_prob, torch.tensor(1.0, dtype=torch.float64), atol=5e-4)


def test_non_spine_age_density_integrates_to_one_multi_epoch() -> None:
    """Multi-epoch non-spine age density should integrate to 1 on [0, T]."""
    birth_rates = torch.tensor([0.30, 0.18, 0.12], dtype=torch.float64)
    death_rates = torch.tensor([0.10, 0.14, 0.09], dtype=torch.float64)
    times = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    grid = torch.linspace(0.0, float(times[-1].item()), steps=4000, dtype=torch.float64)
    density_vals = _pointwise_density_from_loglik(
        calculate_non_spine_ages_log_lik,
        birth_rates,
        death_rates,
        times,
        grid,
    )
    integral = torch.trapz(density_vals, grid)

    assert torch.isclose(integral, torch.tensor(1.0, dtype=torch.float64), atol=2e-3)


def test_extinction_probability_matches_population_zero_mass_multi_epoch() -> None:
    """Multi-epoch Pr(Z_T = 0) should equal p_0(T)."""
    birth_rates = torch.tensor([0.20, 0.12, 0.09], dtype=torch.float64)
    death_rates = torch.tensor([0.28, 0.22, 0.18], dtype=torch.float64)
    times = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    extinction_pr = calculate_extinction_pr(
        birth_rates=birth_rates,
        death_rates=death_rates,
        times=times,
    )
    p0 = torch.exp(
        calculate_populations_log_lik(
            birth_rates=birth_rates,
            death_rates=death_rates,
            ns=torch.tensor([0], dtype=torch.int64),
            times=times,
            initial_population=1,
        ),
    )

    assert torch.isclose(extinction_pr, p0, atol=1e-12)
