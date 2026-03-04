"""Unit tests for the likelihood fns in the inf_bp_texts.inference.likelihood module."""

# ruff: noqa: S101

import pytest
import torch

from inf_bp_texts.inference import likelihood


def test_log_likelihood_simple() -> None:
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
    ) + likelihood.calculate_ages_log_lik(
        birth_rates,
        death_rates,
        ages,
        times=times,
        are_spines=False,
    )

    assert torch.isclose(result, expected)


def test_log_likelihood_negative_branch() -> None:
    """Test log_likelihood returns -inf when population log-likelihood is non-finite."""
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([1000.0]),
        death_rates=torch.tensor([0.1]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
    )
    assert torch.isneginf(result)


def test_log_likelihood_invalid_params() -> None:
    """Test log_likelihood returns -inf for invalid parameters (e.g., neg rates)."""
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([-0.5]),
        death_rates=torch.tensor([0.2]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
    )
    assert result == float("-inf") or torch.isclose(result, torch.tensor(float("-inf")))


def test_log_likelihood_log_scale() -> None:
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
    ) + likelihood.calculate_ages_log_lik(
        birth_rates,
        death_rates,
        ages,
        times=times,
        are_spines=False,
    )

    assert torch.isclose(result, expected)


def test_log_likelihood_spine_ages() -> None:
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
        + likelihood.calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages=spine_ages,
            times=times,
            are_spines=True,
        )
        + likelihood.calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages,
            times=times,
            are_spines=False,
        )
    )

    assert torch.isclose(result, expected)


def test_log_likelihood_is_ordered() -> None:
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
        + likelihood.calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages=torch.tensor([1.0]),
            times=times,
            are_spines=True,
        )
        + likelihood.calculate_ages_log_lik(
            birth_rates,
            death_rates,
            ages=torch.tensor([0.5]),
            times=times,
            are_spines=False,
        )
    )

    assert torch.isclose(result, expected)


def test_log_likelihood_value_errors() -> None:
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


def test_negative_log_likelihood() -> None:
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
