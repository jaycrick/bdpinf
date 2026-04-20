"""Unit tests for the likelihood fns in the inf_bp_texts.inference.likelihood module."""

# ruff: noqa: S101

import pytest
import torch

from inf_bp_texts.inference import likelihood


def _expected_unknown_spine_age_log_lik(
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    ages: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Compute the unknown-spine age log-likelihood via stable log-sum-exp."""
    sum_log_f = likelihood.calculate_ages_log_lik(
        birth_rates,
        death_rates,
        ages,
        times=times,
        are_spines=False,
    )
    a_terms = []
    for age in ages:
        age_tensor = torch.tensor([age], dtype=torch.float64)
        log_g = likelihood.calculate_ages_log_lik(
            birth_rates,
            death_rates,
            age_tensor,
            times=times,
            are_spines=True,
        )
        log_f = likelihood.calculate_ages_log_lik(
            birth_rates,
            death_rates,
            age_tensor,
            times=times,
            are_spines=False,
        )
        a_terms.append(log_g - log_f)

    return (
        sum_log_f
        - torch.log(torch.tensor(float(len(ages)), dtype=torch.float64))
        + torch.logsumexp(torch.stack(a_terms), dim=0)
    )


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
    ) + _expected_unknown_spine_age_log_lik(
        birth_rates=birth_rates,
        death_rates=death_rates,
        ages=ages,
        times=times,
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
    ) + _expected_unknown_spine_age_log_lik(
        birth_rates=birth_rates,
        death_rates=death_rates,
        ages=ages,
        times=times,
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


def test_log_likelihood_unknown_spine_formula_equivalence() -> None:
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


def test_log_likelihood_unknown_spine_single_age_reduces_to_spine() -> None:
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


def test_log_likelihood_unknown_spine_differs_from_all_non_spine() -> None:
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
    ) + likelihood.calculate_ages_log_lik(
        birth_rates,
        death_rates,
        ages,
        times=times,
        are_spines=False,
    )

    assert not torch.isclose(observed, all_non_spine)


def test_log_likelihood_terminal_age_auto_assigned_as_spine() -> None:
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


def test_log_likelihood_multiple_terminal_ages_raise_value_error() -> None:
    """Multiple ages at final time are invalid under unique-spine assumption."""
    with pytest.raises(ValueError, match="More than one age equals"):
        likelihood.log_likelihood(
            birth_rates=torch.tensor([0.5]),
            death_rates=torch.tensor([0.2]),
            ages=torch.tensor([2.0, 2.0, 0.5]),
            times=torch.tensor([2.0]),
        )
