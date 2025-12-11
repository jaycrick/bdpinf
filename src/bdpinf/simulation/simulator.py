"""
Simulation functions for birth-death process models.

This module contains functions for simulating manuscript data from birth-death
processes and making predictions using fitted models.

Examples:
    Simulating manuscript creation and survival:

    >>> import numpy as np
    >>> from inf_bp_texts.simulation.simulator import simulate_birth_death_process
    >>>
    >>> # Set simulation parameters
    >>> birth_rate = 0.5
    >>> death_rate = 0.3
    >>> observation_time = 500
    >>>
    >>> # Run simulation
    >>> manuscripts = simulate_birth_death_process(
    ...     birth_rate, death_rate, observation_time, random_seed=42
    ... )
    >>> print(f"Simulated {len(manuscripts)} surviving manuscripts")

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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def simulate_birth_death_process(
    birth_rate: float,
    death_rate: float,
    max_time: float = 2000,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate manuscript data from the birth-death process.

    This creates a proper birth-death process simulation where manuscripts
    are born at rate λ and die at rate μ. The simulation runs until max_time
    and records both creation times and survival status. If the

    Args:
        birth_rate: Birth rate (λ) for manuscript creation
        death_rate: Death rate (μ) for manuscript loss
        max_time: Maximum time to simulate the process
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with simulated manuscript data including creation times and ages
    """
    if birth_rate <= 0 or death_rate <= 0:
        raise ValueError("Birth rate and death rate must be positive")

    # Create random number generator
    rng = np.random.default_rng(random_seed)

    # Initialize tracking variables
    current_time = 0.0
    manuscripts_alive: List[Tuple[float, int]] = [
        (0.0, 0)
    ]  # (creation_time, manuscript_id)
    all_manuscripts: List[Dict[str, Any]] = [
        {
            "manuscript_id": 0,
            "parent_id": None,
            "creation_time": 0.0,
            "death_time": None,
            "is_spine": False,
        }
    ]  # Record of all manuscripts
    new_manuscript_id = 1
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
                }
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

    return pd.DataFrame(all_manuscripts)


def simulate_multiple_scenarios(
    parameter_sets: List[Tuple[float, float]],
    max_time: float = 2000,
    n_simulations: int = 10,
    random_seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Simulate multiple birth-death scenarios for comparison.

    Args:
        parameter_sets: List of (birth_rate, death_rate) tuples
        max_time: Maximum simulation time
        n_simulations: Number of simulations per parameter set
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping scenario names to simulation results
    """
    # Create random number generator
    rng = np.random.default_rng(random_seed)

    results: Dict[str, pd.DataFrame] = {}

    for i, (birth_rate, death_rate) in enumerate(parameter_sets):
        scenario_name = f"scenario_{i + 1}_b{birth_rate:.3f}_d{death_rate:.3f}"
        scenario_data: List[pd.DataFrame] = []

        for sim_idx in range(n_simulations):
            sim_seed = int(rng.integers(low=0, high=2**31))

            sim_data = simulate_birth_death_process(
                birth_rate=birth_rate,
                death_rate=death_rate,
                max_time=max_time,
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
    age_bins: Optional[NDArray[np.float64]] = None,
) -> Dict[str, Any]:
    """
    Calculate summary statistics from simulation results.

    Args:
        simulation_df: DataFrame from simulate_birth_death_process
        age_bins: Optional age bins for histogram analysis

    Returns:
        Dictionary with simulation statistics
    """
    surviving_manuscripts = simulation_df[simulation_df["survived"]]

    if len(surviving_manuscripts) == 0:
        return {"error": "No surviving manuscripts in simulation"}

    stats: Dict[str, Any] = {
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
