"""Visualization package for birth-death process analysis.

This package contains plotting and visualization utilities for birth-death
process models and their results.
"""

from .plotting import (
    plot_bootstrap_confidence_intervals,
    plot_parameter_comparison,
    plot_simulation_results,
    save_figure,
)

__all__ = [
    "plot_bootstrap_confidence_intervals",
    "plot_parameter_comparison",
    "plot_simulation_results",
    "save_figure",
]
