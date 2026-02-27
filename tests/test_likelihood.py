"""Unit tests for the likelihood fns in the inf_bp_texts.inference.likelihood module."""

# ruff: noqa: S101

import pytest
import torch

from inf_bp_texts.inference import likelihood


def test_log_likelihood_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_likelihood with simple valid input and fixed outputs."""
    # Patch the mathematical functions to return fixed values
    monkeypatch.setattr(
        likelihood,
        "calculate_populations_log_lik",
        lambda **kwargs: torch.tensor(1.0),
    )
    monkeypatch.setattr(
        likelihood,
        "calculate_ages_log_lik",
        lambda *a, **k: torch.tensor(2.0),
    )
    # Simple valid input
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([0.5]),
        death_rates=torch.tensor([0.2]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
    )
    assert torch.isclose(result, torch.tensor(3.0))


def test_log_likelihood_negative_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_likelihood returns -inf if population log-lik is -inf."""
    # Patch to return -inf for population
    monkeypatch.setattr(
        likelihood,
        "calculate_populations_log_lik",
        lambda **kwargs: torch.tensor(float("-inf")),
    )
    monkeypatch.setattr(
        likelihood,
        "calculate_ages_log_lik",
        lambda *a, **k: torch.tensor(2.0),
    )
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([0.5]),
        death_rates=torch.tensor([0.2]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
    )
    assert result == float("-inf") or torch.isclose(result, torch.tensor(float("-inf")))


def test_log_likelihood_invalid_params() -> None:
    """Test log_likelihood returns -inf for invalid parameters (e.g., neg rates)."""
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([-0.5]),
        death_rates=torch.tensor([0.2]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
    )
    assert result == float("-inf") or torch.isclose(result, torch.tensor(float("-inf")))


def test_log_likelihood_log_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_likelihood with log_scale_params=True."""
    monkeypatch.setattr(
        likelihood,
        "calculate_populations_log_lik",
        lambda **kwargs: torch.tensor(1.0),
    )
    monkeypatch.setattr(
        likelihood,
        "calculate_ages_log_lik",
        lambda *a, **k: torch.tensor(2.0),
    )
    # log_scale_params True, so rates are exp'd
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([0.0]),
        death_rates=torch.tensor([0.0]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
        log_scale_params=True,
    )
    assert torch.isclose(result, torch.tensor(3.0))


def test_log_likelihood_spine_ages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_likelihood with spine_ages provided."""
    monkeypatch.setattr(
        likelihood,
        "calculate_populations_log_lik",
        lambda **kwargs: torch.tensor(1.0),
    )
    monkeypatch.setattr(
        likelihood,
        "calculate_ages_log_lik",
        lambda *a, **k: torch.tensor(2.0),
    )
    # Provide spine_ages
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([0.5]),
        death_rates=torch.tensor([0.2]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
        spine_ages=torch.tensor([0.5]),
    )
    assert torch.isclose(result, torch.tensor(3.0))


def test_log_likelihood_is_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test log_likelihood with is_ordered=True."""
    monkeypatch.setattr(
        likelihood,
        "calculate_populations_log_lik",
        lambda **kwargs: torch.tensor(1.0),
    )
    monkeypatch.setattr(
        likelihood,
        "calculate_ages_log_lik",
        lambda *a, **k: torch.tensor(2.0),
    )
    # is_ordered True
    result = likelihood.log_likelihood(
        birth_rates=torch.tensor([0.5]),
        death_rates=torch.tensor([0.2]),
        ages=torch.tensor([1.0, 0.5]),
        times=torch.tensor([2.0]),
        is_ordered=True,
    )
    assert torch.isclose(result, torch.tensor(3.0))


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


def test_negative_log_likelihood(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test negative_log_likelihood returns negated log_likelihood value."""
    monkeypatch.setattr(
        likelihood,
        "log_likelihood",
        lambda **kwargs: torch.tensor(5.0),
    )
    result = likelihood.negative_log_likelihood(
        birth_rates=torch.tensor([0.5]),
        death_rates=torch.tensor([0.2]),
        ages=torch.tensor([1.0]),
        times=torch.tensor([2.0]),
    )
    assert torch.isclose(result, torch.tensor(-5.0))
