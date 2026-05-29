"""Simulation functions for birth-death process models.

This module contains functions for simulating manuscript data from birth-death
processes and making predictions using fitted models.

Examples:
    Simulating manuscript creation and survival with constant rates:

    >>> import numpy as np
    >>> from inf_bp_texts.simulation.simulator import simulate_birth_death_process
    >>>
    >>> # Set simulation parameters
    >>> birth_rate = 0.5
    >>> death_rate = 0.3
    >>> observation_time = 500
    >>>
    >>> # Run simulation with constant rates
    >>> manuscripts = simulate_birth_death_process(
    ...     birth_rate, death_rate, observation_time, random_seed=42
    ... )
    >>> print(f"Simulated {len(manuscripts)} surviving manuscripts")

    Simulating with time-varying rates:

    >>> # Rates change at t=200 and t=400
    >>> birth_rates = np.array([0.5, 0.3, 0.6])
    >>> death_rates = np.array([0.3, 0.4, 0.2])
    >>> times = np.array([200, 400, 600])
    >>>
    >>> manuscripts = simulate_birth_death_process(
    ...     birth_rates, death_rates, times, random_seed=42
    ... )
    >>> print(f"Simulated {len(manuscripts)} manuscripts")

    Predicting future manuscript counts:

    >>> from inf_bp_texts.simulation.simulator import predict_manuscript_counts
    >>>
    >>> # Current manuscript ages
    >>> current_ages = np.array([100, 200, 300, 400])
    >>>
    >>> # Predict counts in 100 years
    >>> predicted_count = predict_manuscript_counts(
    ...     current_ages, birth_rate, death_rate, future_time=100
    ... )
    >>> print(f"Predicted future manuscripts: {predicted_count}")

"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from inf_bp_texts.models.birth_death_process import BirthDeathProcess
from inf_bp_texts.models.mathematical_functions import (
    _population_geometric_parameter,
    calculate_extinction_pr,
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)

MAX_REJECTION_ATTEMPTS = 200_000


class _NonSpineSamplingContext(NamedTuple):
    birth_rate_tensor: torch.Tensor
    death_rate_tensor: torch.Tensor
    max_time_tensor: torch.Tensor
    birth_rate: float
    denom_lambda: float
    envelope_constant: float


def _as_scalar_float(value: torch.Tensor | float, *, name: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            msg = f"{name} must be a scalar tensor"
            raise ValueError(msg)
        scalar_value = float(value.item())
    else:
        scalar_value = float(value)
    return scalar_value


def _draw_from_density_by_rejection(
    density_fn: Callable[[float], float],
    max_time: float,
    rng: np.random.Generator,
    *,
    grid_size: int = 2048,
    max_attempts: int = 200_000,
) -> float:
    # A dense grid gives a robust envelope estimate for smooth age densities.
    grid = np.linspace(0.0, max_time, num=grid_size, endpoint=False)
    density_vals = np.array([density_fn(float(age)) for age in grid], dtype=np.float64)

    if not np.all(np.isfinite(density_vals)):
        msg = "Density envelope contains non-finite values"
        raise RuntimeError(msg)

    envelope = 1.05 * float(np.max(density_vals))
    if envelope <= 0.0:
        msg = "Density envelope must be strictly positive"
        raise RuntimeError(msg)

    for _ in range(max_attempts):
        candidate_age = float(rng.uniform(0.0, max_time))
        candidate_density = float(density_fn(candidate_age))
        if not np.isfinite(candidate_density) or candidate_density < 0.0:
            continue
        if rng.uniform() * envelope <= candidate_density:
            return candidate_age

    msg = "Rejection sampler exhausted max_attempts without acceptance"
    raise RuntimeError(msg)


def _sample_spine_ages(
    birth_rate_tensor: torch.Tensor,
    death_rate_tensor: torch.Tensor,
    max_time_tensor: torch.Tensor,
    rng: np.random.Generator,
    *,
    n_samples: int,
) -> list[float]:
    if n_samples < 0:
        msg = "n_samples must be non-negative"
        raise ValueError(msg)
    if n_samples == 0:
        return []

    birth_rate = float(birth_rate_tensor.item())
    death_rate = float(death_rate_tensor.item())
    max_time = float(max_time_tensor.item())

    # P(spine_age = T): discrete boundary mass at observation time.
    is_critical = np.isclose(birth_rate, death_rate, rtol=1e-12)
    if is_critical:
        avg_rate = 0.5 * (birth_rate + death_rate)
        boundary_mass = (1.0 + avg_rate * max_time) * np.exp(-avg_rate * max_time)
    else:
        denominator = birth_rate - death_rate
        numerator = (birth_rate * np.exp(-death_rate * max_time)) - (
            death_rate * np.exp(-birth_rate * max_time)
        )
        boundary_mass = numerator / denominator
    boundary_mass = float(np.clip(boundary_mass, 0.0, 1.0))

    continuous_mass = max(1.0 - boundary_mass, np.finfo(float).tiny)

    def normalized_continuous_density(age: float) -> float:
        log_lik = calculate_spine_ages_log_lik(
            birth_rates=birth_rate_tensor,
            death_rates=death_rate_tensor,
            ages=torch.tensor(age, dtype=torch.float64),
            times=max_time_tensor,
        )
        if not torch.isfinite(log_lik):
            return 0.0
        density = float(torch.exp(log_lik).item())
        if density < 0.0 or not np.isfinite(density):
            return 0.0
        return float(density / continuous_mass)

    sampled_ages: list[float] = []
    for _ in range(n_samples):
        if rng.uniform() < boundary_mass:
            sampled_ages.append(max_time)
            continue
        sampled_ages.append(
            _draw_from_density_by_rejection(
                density_fn=normalized_continuous_density,
                max_time=max_time,
                rng=rng,
            ),
        )

    return sampled_ages


def _build_non_spine_sampling_context(
    birth_rate_tensor: torch.Tensor,
    death_rate_tensor: torch.Tensor,
    max_time_tensor: torch.Tensor,
) -> _NonSpineSamplingContext:
    birth_rate = float(birth_rate_tensor.item())
    death_rate = float(death_rate_tensor.item())
    max_time = float(max_time_tensor.item())

    exp_lambda_t = np.exp(-birth_rate * max_time)
    denom_lambda = 1.0 - exp_lambda_t
    if denom_lambda <= 0.0:
        msg = "Invalid truncated-exponential proposal normalization"
        raise RuntimeError(msg)

    growth_rate = birth_rate - death_rate
    is_critical = np.isclose(growth_rate, 0.0, rtol=1e-12)
    if is_critical:
        # Critical case envelope: ((1 + λT) / T) * exp(-λa)
        proposal_scale = (1.0 + birth_rate * max_time) / max_time
    else:
        exp_growth = np.exp(-growth_rate * max_time)
        denom_growth = 1.0 - exp_growth
        if denom_growth == 0.0:
            msg = "Invalid non-spine envelope normalization"
            raise RuntimeError(msg)
        proposal_scale = (birth_rate - death_rate * exp_growth) / denom_growth

    envelope_constant = proposal_scale * (denom_lambda / birth_rate)
    if envelope_constant <= 0.0 or not np.isfinite(envelope_constant):
        msg = "Invalid non-spine envelope constant"
        raise RuntimeError(msg)

    return _NonSpineSamplingContext(
        birth_rate_tensor=birth_rate_tensor,
        death_rate_tensor=death_rate_tensor,
        max_time_tensor=max_time_tensor,
        birth_rate=birth_rate,
        denom_lambda=denom_lambda,
        envelope_constant=envelope_constant,
    )


def _sample_single_non_spine_age(
    context: _NonSpineSamplingContext,
    rng: np.random.Generator,
    *,
    max_attempts: int = MAX_REJECTION_ATTEMPTS,
) -> float:
    for _ in range(max_attempts):
        u = rng.uniform()
        candidate_age = -np.log(1.0 - u * context.denom_lambda) / context.birth_rate

        proposal_density = context.birth_rate * np.exp(
            -context.birth_rate * candidate_age,
        )
        proposal_density /= context.denom_lambda
        log_lik = calculate_non_spine_ages_log_lik(
            birth_rates=context.birth_rate_tensor,
            death_rates=context.death_rate_tensor,
            ages=torch.tensor(float(candidate_age), dtype=torch.float64),
            times=context.max_time_tensor,
        )
        if not torch.isfinite(log_lik):
            continue
        target_density = float(torch.exp(log_lik).item())
        if target_density <= 0.0 or proposal_density <= 0.0:
            continue

        accept_ratio = target_density / (context.envelope_constant * proposal_density)
        if not np.isfinite(accept_ratio) or accept_ratio <= 0.0:
            continue
        if rng.uniform() <= min(1.0, accept_ratio):
            return float(candidate_age)

    msg = "Non-spine rejection sampler exhausted max_attempts without acceptance"
    raise RuntimeError(msg)


def _sample_non_spine_ages(
    birth_rate_tensor: torch.Tensor,
    death_rate_tensor: torch.Tensor,
    max_time_tensor: torch.Tensor,
    rng: np.random.Generator,
    *,
    n_samples: int,
) -> list[float]:
    if n_samples < 0:
        msg = "n_samples must be non-negative"
        raise ValueError(msg)
    if n_samples == 0:
        return []

    context = _build_non_spine_sampling_context(
        birth_rate_tensor=birth_rate_tensor,
        death_rate_tensor=death_rate_tensor,
        max_time_tensor=max_time_tensor,
    )

    return [
        _sample_single_non_spine_age(context=context, rng=rng) for _ in range(n_samples)
    ]


def simulate_birth_death_process(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> BirthDeathProcess:
    """Simulate manuscript data from the birth-death process with time-varying rates.

    This creates a proper birth-death process simulation where manuscripts
    are born at rate λ and die at rate μ. The rates can change at specified
    time points. The simulation runs until the final time is reached and
    records both creation times and survival status.

    Args:
        birth_rates: Birth rate(s) (λ) for manuscript creation. Either a single
            float or an array of rates corresponding to time periods.
        death_rates: Death rate(s) (μ) for manuscript loss. Either a single
            float or an array of rates corresponding to time periods.
        times: End time(s) for the simulation. Either a single float or an array
            of time points at which rates change. Must be the same length as
            birth_rates and death_rates if arrays are provided.
        random_seed: Random seed for reproducibility

    Returns:
        BirthDeathProcess instance containing the simulated manuscript records

    """
    # Convert inputs to arrays for uniform handling
    birth_rates_arr = np.atleast_1d(birth_rates)
    death_rates_arr = np.atleast_1d(death_rates)
    times_arr = np.atleast_1d(times)

    # Validate inputs
    if not (len(birth_rates_arr) == len(death_rates_arr) == len(times_arr)):
        msg = "birth_rates, death_rates, and times must have the same length"
        raise ValueError(
            msg,
        )

    if np.any(birth_rates_arr <= 0) or np.any(death_rates_arr <= 0):
        msg = "Birth rates and death rates must be positive"
        raise ValueError(msg)

    if np.any(times_arr <= 0):
        msg = "Times must be positive"
        raise ValueError(msg)

    if len(times_arr) > 1 and np.any(np.diff(times_arr) <= 0):
        msg = "Times must be strictly increasing"
        raise ValueError(msg)

    # Base RNG used to derive per-attempt seeds for reproducibility
    base_seed_rng = np.random.default_rng(random_seed)

    if ensure_non_extinct and (not isinstance(max_retries, int) or max_retries < 0):
        raise ValueError("max_retries must be a non-negative integer")

    attempts_allowed = max_retries if ensure_non_extinct else 0
    # We allow (attempts_allowed + 1) attempts: initial + retries
    for attempt in range(attempts_allowed + 1):
        # Create a per-attempt RNG so retries produce different draws but are
        # reproducible when `random_seed` is provided.
        if random_seed is None:
            rng = np.random.default_rng()
        else:
            child_seed = int(base_seed_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(child_seed)

        # Initialize tracking variables for this attempt
        current_time = 0.0
        manuscripts_alive: list[tuple[float, int]] = [
            (0.0, 0),
        ]  # (creation_time, manuscript_id)
        all_manuscripts: list[dict[str, Any]] = [
            {
                "manuscript_id": 0,
                "parent_id": None,
                "creation_time": 0.0,
                "death_time": None,
                "is_spine": False,
            },
        ]  # Record of all manuscripts
        new_manuscript_id = 1

        # Initialize rate tracking
        period_idx = 0
        birth_rate = float(birth_rates_arr[period_idx])
        death_rate = float(death_rates_arr[period_idx])
        max_time = float(times_arr[-1])

        birth_death_rate = birth_rate + death_rate
        prop_birth = birth_rate / birth_death_rate
        n_alive = len(manuscripts_alive)
        total_rate = n_alive * birth_death_rate

        while current_time < max_time:
            if total_rate <= 0:
                break

            # Time to next event
            dt = rng.exponential(1.0 / total_rate)
            current_time += dt

            # Check if we've crossed into a new period and update rates
            if period_idx < len(times_arr) - 1 and current_time > times_arr[period_idx]:
                # Set current time to the period boundary
                current_time = times_arr[period_idx]
                # Update to next period's parameters
                period_idx += 1
                birth_rate = float(birth_rates_arr[period_idx])
                death_rate = float(death_rates_arr[period_idx])
                birth_death_rate = birth_rate + death_rate
                prop_birth = birth_rate / birth_death_rate
                total_rate = len(manuscripts_alive) * birth_death_rate
                # Continue to start of loop to generate dt with new rates
                continue

            if current_time >= max_time:
                break

            # Determine if event is birth or death
            if rng.random() < prop_birth:
                # Birth event
                new_entry = (current_time, new_manuscript_id)
                # Insert at a random position other than the first.
                if len(manuscripts_alive) == 1:
                    manuscripts_alive.append(new_entry)
                    parent_id = manuscripts_alive[0][1]
                else:
                    insert_idx = int(rng.integers(1, len(manuscripts_alive) + 1))
                    manuscripts_alive.insert(insert_idx, new_entry)
                    parent_id = manuscripts_alive[insert_idx - 1][1]

                all_manuscripts.append(
                    {
                        "manuscript_id": new_manuscript_id,
                        "parent_id": parent_id,
                        "creation_time": current_time,
                        "death_time": None,
                        "is_spine": False,
                    },
                )
                new_manuscript_id += 1
                total_rate += birth_death_rate
            else:
                # Death event
                death_idx = rng.integers(0, len(manuscripts_alive))
                creation_time, dead_id = manuscripts_alive.pop(death_idx)

                # Update the record
                for manuscript in all_manuscripts:
                    if manuscript["manuscript_id"] == dead_id:
                        manuscript["death_time"] = current_time
                        break

                total_rate -= birth_death_rate

        # Process surviving manuscripts
        spine_id = manuscripts_alive[0][1] if manuscripts_alive else None
        if spine_id is not None:
            for manuscript in all_manuscripts:
                if manuscript["manuscript_id"] == spine_id:
                    manuscript["is_spine"] = True
                    break

        # Create DataFrame and add derived columns
        df = pd.DataFrame(all_manuscripts)

        # Add survived column (True if death_time is None)
        df["survived"] = df["death_time"].isna()

        # Add age column for surviving manuscripts (vectorized)
        df["age"] = np.where(df["survived"], max_time - df["creation_time"], np.nan)

        # Add lifespan column for dead manuscripts (vectorized)
        df["lifespan"] = np.where(
            ~df["survived"],
            df["death_time"] - df["creation_time"],
            np.nan,
        )

        # If we don't need to ensure non-extinction, return first draw.
        if not ensure_non_extinct:
            return BirthDeathProcess.from_manuscript_records(
                df,
                epoch_end_times=times_arr,
                birth_rates=birth_rates_arr,
                death_rates=death_rates_arr,
            )

        # If we require non-extinction, check if at least one survived.
        if df["survived"].any():
            return BirthDeathProcess.from_manuscript_records(
                df,
                epoch_end_times=times_arr,
                birth_rates=birth_rates_arr,
                death_rates=death_rates_arr,
            )

        # Otherwise, loop to retry until attempts exhausted.

    # If we reach here, retries were exhausted without producing a non-extinct draw.
    msg = (
        f"simulate_birth_death_process: exhausted {attempts_allowed} retries "
        "without producing a non-extinct population"
    )
    raise RuntimeError(msg)


def simulate_bdp_ages_from_distributions(
    birth_rate: torch.Tensor,
    death_rate: torch.Tensor,
    max_time: torch.Tensor,
    rng: np.random.Generator,
    ensure_non_extinct: bool = False,
) -> BirthDeathProcess:
    """Sample population size and manuscript ages from BDP-derived distributions.

    The population size draw follows the existing extinction/geometric logic.
    Conditional on a non-zero population, this function samples exactly one spine
    age and ``n-1`` non-spine ages on ``[0, T]``.

    Args:
        birth_rate: Scalar birth-rate tensor.
        death_rate: Scalar death-rate tensor.
        max_time: Scalar observation-time tensor.
        rng: NumPy random number generator.
        ensure_non_extinct: If True, force a non-zero population draw by sampling
            from the conditional geometric component only.

    Returns:
        BirthDeathProcess instance containing the sampled age groups.

    Raises:
        ValueError: If rates or max_time are not positive, or tensors are not scalar.
        RuntimeError: If rejection sampling cannot find an accepted draw.

    """
    birth_rate_val = _as_scalar_float(birth_rate, name="birth_rate")
    death_rate_val = _as_scalar_float(death_rate, name="death_rate")
    max_time_val = _as_scalar_float(max_time, name="max_time")

    if birth_rate_val <= 0.0 or death_rate_val <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if max_time_val <= 0.0:
        msg = "max_time must be positive"
        raise ValueError(msg)

    birth_rate_tensor = torch.tensor(birth_rate_val, dtype=torch.float64)
    death_rate_tensor = torch.tensor(death_rate_val, dtype=torch.float64)
    max_time_tensor = torch.tensor(max_time_val, dtype=torch.float64)

    pop_size: int
    if ensure_non_extinct:
        pop_size = rng.geometric(
            _population_geometric_parameter(birth_rate, death_rate, max_time).item(),
        )
    elif rng.uniform() < calculate_extinction_pr(birth_rate, death_rate, max_time):
        pop_size = 0
    else:
        pop_size = rng.geometric(
            _population_geometric_parameter(
                birth_rate,
                death_rate,
                max_time,
            ).item(),
        )

    ages: dict[str, list[float]] = {"spine": [], "non_spine": []}
    if pop_size > 0:
        ages["spine"] = _sample_spine_ages(
            birth_rate_tensor=birth_rate_tensor,
            death_rate_tensor=death_rate_tensor,
            max_time_tensor=max_time_tensor,
            rng=rng,
            n_samples=1,
        )
        ages["non_spine"] = _sample_non_spine_ages(
            birth_rate_tensor=birth_rate_tensor,
            death_rate_tensor=death_rate_tensor,
            max_time_tensor=max_time_tensor,
            rng=rng,
            n_samples=pop_size - 1,
        )

    return BirthDeathProcess.from_age_samples(
        spine_ages=ages["spine"],
        non_spine_ages=ages["non_spine"],
        epoch_end_times=max_time,
        birth_rates=birth_rate,
        death_rates=death_rate,
    )


if __name__ == "__main__":
    for _ in range(10):
        sampled_ages = simulate_bdp_ages_from_distributions(
            birth_rate=torch.tensor(0.5),
            death_rate=torch.tensor(0.3),
            max_time=torch.tensor(5.0),
            rng=np.random.default_rng(),
        )
        print(sampled_ages)
