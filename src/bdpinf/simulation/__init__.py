"""
Simulation package for birth-death process data generation.

This package contains simulation and prediction utilities for birth-death
process models.
"""

from .simulator import (
    calculate_simulation_statistics,
    generate_synthetic_dataset,
    predict_manuscript_counts,
    simulate_birth_death_process,
    simulate_multiple_scenarios,
)

__all__ = [
    "simulate_birth_death_process",
    "predict_manuscript_counts",
    "generate_synthetic_dataset",
    "simulate_multiple_scenarios",
    "calculate_simulation_statistics",
]
