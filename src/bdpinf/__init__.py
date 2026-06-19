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
import os

# When jax[cuda12] is installed but the GPU is unavailable (e.g. vGPU with VMM
# restrictions, or subprocess workers spawned by joblib), JAX raises a fatal
# RuntimeError during CUDA plugin discovery.  Defaulting to cpu here prevents
# that failure in fresh subprocesses while still honouring an explicit
# JAX_PLATFORMS=cuda set by the caller before import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from ._paths import get_project_root
from .models.birth_death_process import BirthDeathProcess

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0"
__author__ = "JC"
__email__ = "jcacademicresearch@gmail.com"

__all__ = ["BirthDeathProcess", "get_project_root"]
