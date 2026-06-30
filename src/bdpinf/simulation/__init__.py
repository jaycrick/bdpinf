"""Simulation package for birth-death process data generation.

This package contains simulation and prediction utilities for birth-death
process models.
"""

from inf_bp_texts.visualization.splitting_plots import (
    plot_conditioned_trajectories,
    plot_empirical_population_mean,
    plot_level_survival_probabilities,
)

from .simulator import (
    LifespanSimulationArrays,
    simulate_bdp_ages_from_distributions,
    simulate_birth_death_lifespans,
    simulate_birth_death_lifespans_arrays,
    simulate_birth_death_process,
    simulate_birth_death_process_from_contour_process,
)
from .splitting import (
    SplittingConfig,
    select_level_times_for_band,
    simulate_subcritical_survival_splitting,
)
from .survival_conditioning import (
    ConditionedSurvivalResult,
    EraConditioningDiagnostic,
    MatchedSurvivalSample,
    build_target_ranges,
    sample_conditioned_survival_per_posterior,
    sample_conditioned_survival_process,
)

__all__ = [
    "ConditionedSurvivalResult",
    "EraConditioningDiagnostic",
    "LifespanSimulationArrays",
    "MatchedSurvivalSample",
    "SplittingConfig",
    "build_target_ranges",
    "plot_conditioned_trajectories",
    "plot_empirical_population_mean",
    "plot_level_survival_probabilities",
    "sample_conditioned_survival_per_posterior",
    "sample_conditioned_survival_process",
    "select_level_times_for_band",
    "simulate_bdp_ages_from_distributions",
    "simulate_birth_death_lifespans",
    "simulate_birth_death_lifespans_arrays",
    "simulate_birth_death_process",
    "simulate_birth_death_process_from_contour_process",
    "simulate_subcritical_survival_splitting",
]
