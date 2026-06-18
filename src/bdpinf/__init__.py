"""inf-bp-texts: A Python platform for inference and analysis on manuscript datasets.

This package provides birth-death process modeling tools for manuscript data analysis,
including parameter estimation, simulation, and visualization capabilities.

The package is organized into focused modules:
- models: Core classes and mathematical functions
- inference: Parameter fitting and likelihood calculations
- simulation: Birth-death process simulation
- visualization: Plotting and data visualization

Import the components you need directly from their respective modules.
"""

import logging

import jax

jax.config.update("jax_enable_x64", True)

from ._paths import get_project_root
from .models.birth_death_process import BirthDeathProcess

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0"
__author__ = "JC"
__email__ = "jcacademicresearch@gmail.com"

__all__ = ["BirthDeathProcess", "get_project_root"]
