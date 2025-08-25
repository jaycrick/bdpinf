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

from ..models.mathematical_functions import predict_manuscript_survival


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
    and records both creation times and survival status.

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

    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize tracking variables
    creation_times: List[float] = []
    current_time = 0.0
    manuscripts_alive: List[Tuple[float, int]] = []  # (creation_time, manuscript_id)
    all_manuscripts: List[Dict[str, Any]] = []  # Record of all manuscripts
    manuscript_id = 0

    while current_time < max_time:
        # Current population size
        n_alive = len(manuscripts_alive)

        # Total rate = birth_rate + death_rate * n_alive
        total_rate = birth_rate + death_rate * n_alive

        if total_rate <= 0:
            break

        # Time to next event
        dt = np.random.exponential(1.0 / total_rate)
        current_time += dt

        if current_time >= max_time:
            break

        # Determine if event is birth or death
        if np.random.random() < birth_rate / total_rate:
            # Birth event
            manuscripts_alive.append((current_time, manuscript_id))
            all_manuscripts.append(
                {
                    "manuscript_id": manuscript_id,
                    "creation_time": current_time,
                    "death_time": None,
                    "survived": True,
                }
            )
            creation_times.append(current_time)
            manuscript_id += 1
        else:
            # Death event (if any manuscripts alive)
            if manuscripts_alive:
                # Choose a random manuscript to die
                death_idx = np.random.randint(len(manuscripts_alive))
                creation_time, dead_id = manuscripts_alive.pop(death_idx)

                # Remove the creation time from creation_times
                # since this manuscript no longer survives
                if creation_time in creation_times:
                    creation_times.remove(creation_time)

                # Update the record
                for manuscript in all_manuscripts:
                    if manuscript["manuscript_id"] == dead_id:
                        manuscript["death_time"] = current_time
                        manuscript["survived"] = False
                        break

    # Process surviving manuscripts
    surviving_data: List[Dict[str, Any]] = []
    for manuscript in all_manuscripts:
        if manuscript["survived"]:
            creation_time = manuscript["creation_time"]
            age = max_time - creation_time
            surviving_data.append(
                {
                    "manuscript_id": manuscript["manuscript_id"],
                    "creation_time": creation_time,
                    "age": age,
                    "survived": True,
                }
            )

    # Process dead manuscripts
    dead_data: List[Dict[str, Any]] = []
    for manuscript in all_manuscripts:
        if not manuscript["survived"]:
            death_time = manuscript["death_time"]
            creation_time = manuscript["creation_time"]
            if death_time is not None and creation_time is not None:
                lifespan = death_time - creation_time
                dead_data.append(
                    {
                        "manuscript_id": manuscript["manuscript_id"],
                        "creation_time": creation_time,
                        "death_time": death_time,
                        "lifespan": lifespan,
                        "survived": False,
                    }
                )

    # Combine data
    all_data = surviving_data + dead_data

    if not all_data:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(
            columns=["manuscript_id", "creation_time", "age", "survived"]
        )

    return pd.DataFrame(all_data)


def predict_manuscript_counts(
    birth_rate: float,
    death_rate: float,
    ages: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Predict expected manuscript counts for given ages using birth-death model.

    Args:
        birth_rate: Birth rate parameter (λ)
        death_rate: Death rate parameter (μ)
        ages: Array of ages to predict for

    Returns:
        Array of predicted manuscript counts
    """
    if birth_rate <= 0 or death_rate <= 0:
        raise ValueError("Birth rate and death rate must be positive")

    result = predict_manuscript_survival(birth_rate, death_rate, ages)

    # Ensure we return an NDArray
    if not isinstance(result, np.ndarray):
        return np.array([result], dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def generate_synthetic_dataset(
    true_birth_rate: float,
    true_death_rate: float,
    observation_years: Tuple[float, float] = (500, 1500),
    n_manuscripts: int = 30,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic manuscript dataset for testing and demonstration.

    Args:
        true_birth_rate: True birth rate parameter
        true_death_rate: True death rate parameter
        observation_years: Tuple of (start_year, end_year) for manuscript creation
        n_manuscripts: Number of manuscripts to generate
        noise_level: Amount of Poisson noise to add to counts
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic manuscript data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    start_year, end_year = observation_years

    # Create manuscript years evenly spaced
    manuscript_years = np.linspace(
        start_year, end_year, n_manuscripts, dtype=np.float64
    )

    # Calculate ages when observed at end_year + some buffer
    observation_time = end_year + 100
    ages_at_observation = observation_time - manuscript_years

    # Calculate true counts using the survival model
    true_counts = predict_manuscript_survival(
        true_birth_rate, true_death_rate, ages_at_observation
    )

    # Add Poisson noise to simulate realistic observation uncertainty
    if noise_level > 0:
        noisy_counts = np.random.poisson(true_counts * (1 + noise_level))
        # Ensure we have at least some manuscripts
        noisy_counts = np.maximum(noisy_counts, 1)
    else:
        noisy_counts = np.round(true_counts).astype(int)

    return pd.DataFrame(
        {
            "age": ages_at_observation,
            "count": noisy_counts,
            "manuscript_year": manuscript_years,
            "true_count": true_counts,
        }
    )


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
    if random_seed is not None:
        np.random.seed(random_seed)

    results = {}

    for i, (birth_rate, death_rate) in enumerate(parameter_sets):
        scenario_name = f"scenario_{i + 1}_b{birth_rate:.3f}_d{death_rate:.3f}"
        scenario_data: List[pd.DataFrame] = []

        for sim_idx in range(n_simulations):
            # Use different seed for each simulation if provided
            sim_seed = random_seed + sim_idx if random_seed is not None else None

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

    stats = {
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
