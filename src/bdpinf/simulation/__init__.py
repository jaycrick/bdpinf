"""Simulation package for birth-death process data generation.

This package contains simulation and prediction utilities for birth-death
process models.
The forward and era-conditioned simulators are implemented in `bdsurv`;
the functions here either wrap their results in a ``BirthDeathProcess``
or re-export them unchanged.
"""

from bdsurv import (
    ConditionedSurvivalResult,
    EraConditioningDiagnostic,
    LifespanSimulationArrays,
    MatchedSurvivalSample,
    build_target_ranges,
    sample_conditioned_survival_per_posterior,
    sample_conditioned_survival_process,
    simulate_birth_death_lifespans_arrays,
)

from .age_sampling import SampledAges, sample_ages_in_ranges, simulate_ages_from_rates
from .conditioned import (
    ConditionedTrajectory,
    simulate_contour_given_survival,
    simulate_survival_splitting,
)
from .forward import simulate_birth_death, simulate_birth_death_lifespans
from .simulator import (
    simulate_bdp_ages_from_distributions,
    simulate_bdp_ages_given_size,
)

__all__ = [
    "ConditionedSurvivalResult",
    "ConditionedTrajectory",
    "EraConditioningDiagnostic",
    "LifespanSimulationArrays",
    "MatchedSurvivalSample",
    "SampledAges",
    "build_target_ranges",
    "sample_ages_in_ranges",
    "sample_conditioned_survival_per_posterior",
    "sample_conditioned_survival_process",
    "simulate_ages_from_rates",
    "simulate_bdp_ages_from_distributions",
    "simulate_bdp_ages_given_size",
    "simulate_birth_death",
    "simulate_birth_death_lifespans",
    "simulate_birth_death_lifespans_arrays",
    "simulate_contour_given_survival",
    "simulate_survival_splitting",
]
