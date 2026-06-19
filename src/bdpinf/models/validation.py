"""Input validation for birth-death process calculations."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from inf_bp_texts.utils import (
    ensure_1d_tensors,
    validate_matching_lengths,
    validate_matching_shapes,
)


def _validate_times_positive(times: jax.Array) -> None:
    """Validate that times are strictly positive."""
    if jnp.any(times <= 0):
        raise ValueError(f"times must be positive. Got times: {times}")


def _validate_times_strictly_increasing(
    times: jax.Array,
    *,
    all_positive: bool = False,
) -> None:
    """Validate that times are strictly increasing."""
    if all_positive:
        diffs = jnp.diff(jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), times]))
    else:
        diffs = jnp.diff(times)
    if jnp.any(diffs <= 0):
        raise ValueError(f"times must be strictly increasing. Got times: {times}")


def _validate_ages_in_range(
    ages: jax.Array,
    end_time: jax.Array,
) -> None:
    """Validate that ages are within [0, end_time]."""
    if jnp.any(ages < 0) or jnp.any(ages > end_time):
        raise ValueError(f"ages must be in [0, {end_time}]. Got ages: {ages}")


def _validate_rates_non_negative(
    birth_rates: jax.Array | None,
    death_rates: jax.Array | None,
) -> None:
    """Validate that birth and death rates are non-negative."""
    if birth_rates is not None and jnp.any(birth_rates < 0):
        msg = f"birth_rates must be non-negative. Got birth_rates: {birth_rates}"
        raise ValueError(msg)
    if death_rates is not None and jnp.any(death_rates < 0):
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


def _validate_rate_time_consistency(
    tensor_map: dict[str, jax.Array],
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


def _validate_times_properties(
    times: jax.Array,
    *,
    require_positive: bool = True,
    require_increasing: bool = True,
) -> None:
    """Validate time tensor properties."""
    if require_positive:
        _validate_times_positive(times)
    if require_increasing:
        _validate_times_strictly_increasing(
            times,
            all_positive=require_positive,
        )


def normalize_and_validate_inputs(
    birth_rates: jax.Array | float | None = None,
    death_rates: jax.Array | float | None = None,
    times: jax.Array | float | None = None,
    ages: jax.Array | float | None = None,
    *,
    config: ValidationConfig | None = None,
) -> tuple[jax.Array, ...]:
    """Normalize and validate input tensors.

    Parameters
    ----------
    birth_rates, death_rates, times, ages : jax.Array | float | None
        Input values to normalize and validate.
    config : ValidationConfig, optional
        Validation configuration. If None, defaults are used.

    Returns
    -------
    tuple[jax.Array, ...]
        Normalized arrays in the order they were passed.

    """
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

    normalized = ensure_1d_tensors(*inputs) if inputs else ()
    tensor_map = dict(zip(labels, normalized, strict=True))

    if config is None:
        config = ValidationConfig()

    if config.require_rates_non_negative:
        _validate_rates_non_negative(
            tensor_map.get("birth_rates"),
            tensor_map.get("death_rates"),
        )

    _validate_rate_time_consistency(
        tensor_map,
        validate_lengths=config.validate_lengths,
        validate_shapes=config.validate_shapes,
    )

    if times is not None:
        _validate_times_properties(
            tensor_map["times"],
            require_positive=config.require_times_positive,
            require_increasing=config.require_times_increasing,
        )

    if config.require_ages_in_range and ages is not None:
        if times is None:
            msg = "times is required for age range validation."
            raise ValueError(msg)
        end_time = tensor_map["times"][-1]
        _validate_ages_in_range(tensor_map["ages"], end_time)

    return tuple(tensor_map[label] for label in labels)
