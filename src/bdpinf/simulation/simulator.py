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

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def simulate_birth_death_process(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
) -> pd.DataFrame:
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
        DataFrame with simulated manuscript data including creation times and ages

    """
    # Convert inputs to arrays for uniform handling
    birth_rates_arr = np.atleast_1d(birth_rates)
    death_rates_arr = np.atleast_1d(death_rates)
    times_arr = np.atleast_1d(times)

    # Validate inputs
    if not (len(birth_rates_arr) == len(death_rates_arr) == len(times_arr)):
        raise ValueError(
            "birth_rates, death_rates, and times must have the same length",
        )

    if np.any(birth_rates_arr <= 0) or np.any(death_rates_arr <= 0):
        raise ValueError("Birth rates and death rates must be positive")

    if np.any(times_arr <= 0):
        raise ValueError("Times must be positive")

    if len(times_arr) > 1 and np.any(np.diff(times_arr) <= 0):
        raise ValueError("Times must be strictly increasing")

    # Create random number generator
    rng = np.random.default_rng(random_seed)

    # Initialize tracking variables
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

    return df


def simulate_multiple_scenarios(
    parameter_sets: list[tuple[float, float]],
    times: float | NDArray[np.float64] = 2000.0,
    n_simulations: int = 10,
    random_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Simulate multiple birth-death scenarios for comparison.

    Args:
        parameter_sets: List of (birth_rate, death_rate) tuples
        times: End time(s) for simulation. Either a single float or array
            of time points for time-varying rates.
        n_simulations: Number of simulations per parameter set
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping scenario names to simulation results

    """
    # Create random number generator
    rng = np.random.default_rng(random_seed)

    results: dict[str, pd.DataFrame] = {}

    for i, (birth_rate, death_rate) in enumerate(parameter_sets):
        scenario_name = f"scenario_{i + 1}_b{birth_rate:.3f}_d{death_rate:.3f}"
        scenario_data: list[pd.DataFrame] = []

        for sim_idx in range(n_simulations):
            sim_seed = int(rng.integers(low=0, high=2**31))

            sim_data = simulate_birth_death_process(
                birth_rates=birth_rate,
                death_rates=death_rate,
                times=times,
                random_seed=sim_seed,
            )

            # Add simulation metadata
            sim_data["simulation_id"] = sim_idx
            sim_data["birth_rate"] = birth_rate
            sim_data["death_rate"] = death_rate

            scenario_data.append(sim_data)

        # Combine all simulations for this scenario
        results[scenario_name] = pd.concat(scenario_data, ignore_index=True)

    return results


def calculate_simulation_statistics(
    simulation_df: pd.DataFrame,
    age_bins: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Calculate summary statistics from simulation results.

    Args:
        simulation_df: DataFrame from simulate_birth_death_process
        age_bins: Optional age bins for histogram analysis

    Returns:
        Dictionary with simulation statistics

    """
    surviving_manuscripts = simulation_df[simulation_df["survived"]]

    if len(surviving_manuscripts) == 0:
        return {"error": "No surviving manuscripts in simulation"}

    stats: dict[str, Any] = {
        "total_manuscripts_created": len(simulation_df),
        "manuscripts_survived": len(surviving_manuscripts),
        "survival_rate": len(surviving_manuscripts) / len(simulation_df),
        "mean_age": surviving_manuscripts["age"].mean(),
        "median_age": surviving_manuscripts["age"].median(),
        "age_std": surviving_manuscripts["age"].std(),
        "min_age": surviving_manuscripts["age"].min(),
        "max_age": surviving_manuscripts["age"].max(),
    }

    # Calculate age distribution if bins provided
    if age_bins is not None:
        counts, bin_edges = np.histogram(surviving_manuscripts["age"], bins=age_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        stats["age_distribution"] = {
            "bin_centers": bin_centers,
            "counts": counts,
            "bin_edges": bin_edges,
        }

    return stats
