"""Simulation package for birth-death process data generation.

This package contains simulation and prediction utilities for birth-death
process models.
"""

from .conditioned import (
    ConditionedTrajectory,
    simulate_birth_death_process_from_contour_process,
    simulate_subcritical_survival_splitting,
)
from .simulator import (
    LifespanSimulationArrays,
    simulate_bdp_ages_from_distributions,
    simulate_birth_death_lifespans,
    simulate_birth_death_lifespans_arrays,
    simulate_birth_death_process,
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
    "ConditionedTrajectory",
    "EraConditioningDiagnostic",
    "LifespanSimulationArrays",
    "MatchedSurvivalSample",
    "build_target_ranges",
    "sample_conditioned_survival_per_posterior",
    "sample_conditioned_survival_process",
    "simulate_bdp_ages_from_distributions",
    "simulate_birth_death_lifespans",
    "simulate_birth_death_lifespans_arrays",
    "simulate_birth_death_process",
    "simulate_birth_death_process_from_contour_process",
    "simulate_subcritical_survival_splitting",
]
