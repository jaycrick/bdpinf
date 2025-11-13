"""
Models package for birth-death process analysis.

This package contains the core model definitions and mathematical functions
for birth-death process modeling of manuscript data.
"""

from .birth_death_process import BirthDeathProcess
from .mathematical_functions import (
    calculate_ages_log_likelihood,
    calculate_non_spine_ages_log_likelihood,
    calculate_populations_log_likelihood,
    predict_manuscript_survival,
)

__all__ = [
    "BirthDeathProcess",
    "calculate_populations_log_likelihood",
    "calculate_non_spine_ages_log_likelihood",
    "calculate_ages_log_likelihood",
    "predict_manuscript_survival",
]
