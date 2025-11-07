"""
Models package for birth-death process analysis.

This package contains the core model definitions and mathematical functions
for birth-death process modeling of manuscript data.
"""

from .birth_death_process import BirthDeathProcess
from .mathematical_functions import (
    calculate_age_log_likelihood,
    calculate_population_log_likelihood,
    predict_manuscript_survival,
)

__all__ = [
    "BirthDeathProcess",
    "calculate_population_log_likelihood",
    "calculate_age_log_likelihood",
    "predict_manuscript_survival",
]
