"""
Models package for birth-death process analysis.

This package contains the core model definitions and mathematical functions
for birth-death process modeling of manuscript data.
"""

from .birth_death_process import BirthDeathProcess
from .mathematical_functions import (
    calculate_age_density,
    calculate_log_population_probability,
    predict_manuscript_survival,
)

__all__ = [
    "BirthDeathProcess",
    "calculate_log_population_probability",
    "calculate_age_density",
    "predict_manuscript_survival",
]
