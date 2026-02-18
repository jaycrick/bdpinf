"""inf-bp-texts: A Python platform for performing inference and analysis on various datasets downloaded from APIs.

This package provides birth-death process modeling tools for manuscript data analysis,
including parameter estimation, simulation, and visualization capabilities.

The package is organized into focused modules:
- models: Core classes and mathematical functions
- inference: Parameter fitting and likelihood calculations
- simulation: Birth-death process simulation
- visualization: Plotting and data visualization

Import the components you need directly from their respective modules.
"""

__version__ = "0.1.0"
__author__ = "JC"
__email__ = "jcacademicresearch@gmail.com"

# Import only the core model class at package level
from .models.birth_death_process import BirthDeathProcess

__all__ = ["BirthDeathProcess"]
