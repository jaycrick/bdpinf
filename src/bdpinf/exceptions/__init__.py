"""Exception package initialization."""

from .validation import (
    ConfigurationError,
    DataProcessingError,
    InfBpTextsError,
    ModelFittingError,
    PlottingError,
    SimulationError,
    ValidationError,
    validate_creation_times,
    validate_dataframe_columns,
    validate_dataframe_not_empty,
    validate_non_negative_number,
    validate_parameter_bounds,
    validate_positive_number,
)

__all__ = [
    # Base exceptions
    "InfBpTextsError",
    "ValidationError",
    "ModelFittingError",
    "DataProcessingError",
    "SimulationError",
    "PlottingError",
    "ConfigurationError",
    # Validation functions
    "validate_positive_number",
    "validate_non_negative_number",
    "validate_dataframe_columns",
    "validate_dataframe_not_empty",
    "validate_parameter_bounds",
    "validate_creation_times",
]
