"""Simulation package for birth-death process data generation.

This package contains simulation and prediction utilities for birth-death
process models.
"""

from .simulator import (
    simulate_bdp_ages_from_distributions,
    simulate_birth_death_process,
)

__all__ = [
    "BirthDeathLineage",
    "build_birth_death_lineage",
    "simulate_bdp_ages_from_distributions",
    "simulate_birth_death_process",
]
