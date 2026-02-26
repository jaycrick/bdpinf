"""Input validation for birth-death process calculations."""

from typing import NamedTuple

import torch

from inf_bp_texts.utils import (
    ensure_1d_tensors,
    validate_matching_lengths,
    validate_matching_shapes,
)


def validate_times_positive(times: torch.Tensor, *, error_message: str) -> None:
    """Validate that times are strictly positive."""
    if torch.any(times <= 0):
        raise ValueError(error_message)


def validate_times_strictly_increasing(
    times: torch.Tensor,
    *,
    all_positive: bool = False,
) -> None:
    """Validate that times are strictly increasing."""
    error_message = f"times must be strictly increasing. Got times: {times}"
    if all_positive:
        diffs = torch.diff(torch.cat([torch.tensor([0.0], dtype=torch.float64), times]))
    else:
        diffs = torch.diff(times)
    if torch.any(diffs <= 0):
        raise ValueError(error_message)


def validate_ages_in_range(
    ages: torch.Tensor,
    end_time: torch.Tensor,
    *,
    error_message: str,
) -> None:
    """Validate that ages are within [0, end_time]."""
    if torch.any(ages < 0) or torch.any(ages > end_time):
        raise ValueError(error_message)


def validate_rates_non_negative(
    birth_rates: torch.Tensor | None,
    death_rates: torch.Tensor | None,
) -> None:
    """Validate that birth and death rates are non-negative."""
    if birth_rates is not None and torch.any(birth_rates < 0):
        msg = f"birth_rates must be non-negative. Got birth_rates: {birth_rates}"
        raise ValueError(msg)
    if death_rates is not None and torch.any(death_rates < 0):
        msg = f"death_rates must be non-negative. Got death_rates: {death_rates}"
        raise ValueError(msg)


class ValidationConfig(NamedTuple):
    """Configuration for input validation."""

    validate_shapes: bool = True
    validate_lengths: bool = True
    require_times_positive: bool = True
    require_times_increasing: bool = True
    require_ages_in_range: bool = True
    require_rates_non_negative: bool = True


def validate_rate_time_consistency(
    tensor_map: dict[str, torch.Tensor],
    *,
    validate_lengths: bool = True,
    validate_shapes: bool = True,
) -> None:
    """Validate consistency between rate and time tensors."""
    rate_time_labels = [
        label
        for label in ("birth_rates", "death_rates", "times")
        if label in tensor_map
    ]
    if not validate_lengths or len(rate_time_labels) <= 1:
        return

    validate_matching_lengths(
        *(tensor_map[label] for label in rate_time_labels),
        names=tuple(rate_time_labels),
    )
    if validate_shapes:
        validate_matching_shapes(
            *(tensor_map[label] for label in rate_time_labels),
            names=tuple(rate_time_labels),
        )


def validate_times_properties(
    times: torch.Tensor,
    *,
    require_positive: bool = True,
    require_increasing: bool = True,
) -> None:
    """Validate time tensor properties."""
    if require_positive:
        validate_times_positive(
            times,
            error_message=f"times must be positive. Got times: {times}",
        )
    if require_increasing:
        validate_times_strictly_increasing(
            times,
            all_positive=require_positive,
        )


def normalize_and_validate_inputs(
    birth_rates: torch.Tensor | float | None = None,
    death_rates: torch.Tensor | float | None = None,
    times: torch.Tensor | float | None = None,
    ages: torch.Tensor | float | None = None,
    *,
    config: ValidationConfig | None = None,
) -> tuple[torch.Tensor, ...]:
    """Normalize and validate input tensors.

    Parameters
    ----------
    birth_rates, death_rates, times, ages : torch.Tensor | float | None
        Input values to normalize and validate.
    config : ValidationConfig, optional
        Validation configuration. If None, defaults are used.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Normalized tensors in the order they were passed.

    """
    # Build input list and labels
    inputs = []
    labels = []
    if birth_rates is not None:
        inputs.append(birth_rates)
        labels.append("birth_rates")
    if death_rates is not None:
        inputs.append(death_rates)
        labels.append("death_rates")
    if times is not None:
        inputs.append(times)
        labels.append("times")
    if ages is not None:
        inputs.append(ages)
        labels.append("ages")

    # Normalize to tensors
    normalized = ensure_1d_tensors(*inputs) if inputs else ()
    tensor_map = dict(zip(labels, normalized, strict=True))

    # Use default config if not provided
    if config is None:
        config = ValidationConfig()

    # Validate rates are non-negative
    if config.require_rates_non_negative:
        validate_rates_non_negative(
            tensor_map.get("birth_rates"),
            tensor_map.get("death_rates"),
        )

    # Validate rate/time consistency
    validate_rate_time_consistency(
        tensor_map,
        validate_lengths=config.validate_lengths,
        validate_shapes=config.validate_shapes,
    )

    # Validate times
    if times is not None:
        validate_times_properties(
            tensor_map["times"],
            require_positive=config.require_times_positive,
            require_increasing=config.require_times_increasing,
        )

    # Validate ages
    if config.require_ages_in_range and ages is not None:
        if times is None:
            msg = "times is required for age range validation."
            raise ValueError(msg)
        end_time = tensor_map["times"][-1]
        validate_ages_in_range(
            tensor_map["ages"],
            end_time,
            error_message=(
                f"ages must be in [0, {end_time}]. Got ages: {tensor_map['ages']}"
            ),
        )

    return tuple(tensor_map[label] for label in labels)
