"""Simulation package for birth-death process data generation.

This package contains simulation and prediction utilities for birth-death
process models.
"""

from .simulator import (
    calculate_simulation_statistics,
    simulate_birth_death_process,
    simulate_multiple_scenarios,
)

__all__ = [
    "calculate_simulation_statistics",
    "simulate_birth_death_process",
    "simulate_multiple_scenarios",
]
