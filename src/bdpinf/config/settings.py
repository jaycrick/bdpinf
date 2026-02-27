"""Configuration management for inf-bp-texts package.

This module provides centralized configuration classes for different components
of the birth-death process analysis toolkit.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""

    method: str = "L-BFGS-B"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    bounds_buffer: float = 1e-10
    initial_birth_rate: float = 0.1
    initial_death_rate: float = 0.01

    def get_scipy_options(self) -> dict[str, Any]:
        """Get options dictionary for scipy.optimize.minimize."""
        return {
            "maxiter": self.max_iterations,
            "ftol": self.tolerance,
        }


@dataclass
class PlottingConfig:
    """Configuration for plotting and visualization."""

    figure_width: int = 800
    figure_height: int = 600
    color_scheme: str = "viridis"
    show_confidence_intervals: bool = True
    line_width: float = 2.0
    marker_size: int = 6
    title_font_size: int = 16
    axis_font_size: int = 12

    def get_plotly_layout(self) -> dict[str, Any]:
        """Get layout dictionary for plotly figures."""
        return {
            "width": self.figure_width,
            "height": self.figure_height,
            "title_font_size": self.title_font_size,
            "font_size": self.axis_font_size,
        }


@dataclass
class ModelConfig:
    """Configuration for birth-death process models."""

    default_observation_time: float = 2025.0
    min_manuscripts_for_fitting: int = 10
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    numerical_stability_threshold: float = 1e-15
    max_parameter_value: float = 100.0

    def get_parameter_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get parameter bounds for optimization."""
        return (
            (1e-10, self.max_parameter_value),  # birth_rate bounds
            (1e-10, self.max_parameter_value),  # death_rate bounds
        )


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""

    default_max_time: float = 2000.0
    default_random_seed: int | None = None
    max_events: int = 100000
    time_step: float = 0.1

    def is_valid_time_range(self, start_time: float, end_time: float) -> bool:
        """Check if time range is valid for simulation."""
        return 0 <= start_time < end_time <= self.default_max_time * 2


# Default configurations
DEFAULT_OPTIMIZATION = OptimizationConfig()
DEFAULT_PLOTTING = PlottingConfig()
DEFAULT_MODEL = ModelConfig()
DEFAULT_SIMULATION = SimulationConfig()


def get_config(config_type: str) -> object:
    """Get default configuration for a specific component.

    Args:
        config_type: Type of configuration
        ("optimization", "plotting", "model", "simulation")

    Returns:
        Configuration object

    Raises:
        ValueError: If config_type is not recognized

    """
    configs = {
        "optimization": DEFAULT_OPTIMIZATION,
        "plotting": DEFAULT_PLOTTING,
        "model": DEFAULT_MODEL,
        "simulation": DEFAULT_SIMULATION,
    }

    if config_type not in configs:
        msg = f"Unknown config type: {config_type}. Available: {list(configs.keys())}"
        raise ValueError(
            msg,
        )

    return configs[config_type]
