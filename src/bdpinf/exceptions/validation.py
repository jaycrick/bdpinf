"""
Custom exceptions for inf-bp-texts package.

This module defines a hierarchy of exceptions that provide clear, actionable
error messages for different types of failures in the birth-death process
analysis toolkit.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd


class InfBpTextsError(Exception):
    """Base exception for inf-bp-texts package."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ValidationError(InfBpTextsError):
    """Raised when input validation fails."""

    def __init__(
        self, message: str, field_name: Optional[str] = None, received_value: Any = None
    ):
        super().__init__(message)
        self.field_name = field_name
        self.received_value = received_value


class ModelFittingError(InfBpTextsError):
    """Raised when model fitting fails."""

    def __init__(
        self,
        message: str,
        optimization_result: Optional[Dict[str, Any]] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.optimization_result = optimization_result
        self.convergence_info = convergence_info


class DataProcessingError(InfBpTextsError):
    """Raised when data processing fails."""

    def __init__(self, message: str, data_info: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.data_info = data_info


class SimulationError(InfBpTextsError):
    """Raised when simulation fails."""

    def __init__(
        self, message: str, simulation_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.simulation_params = simulation_params


class PlottingError(InfBpTextsError):
    """Raised when plotting fails."""

    def __init__(self, message: str, backend_info: Optional[str] = None):
        super().__init__(message)
        self.backend_info = backend_info


class ConfigurationError(InfBpTextsError):
    """Raised when configuration is invalid."""

    pass


# Validation utilities
def validate_positive_number(value: Union[int, float], name: str) -> Union[int, float]:
    """
    Validate that a number is positive.

    Args:
        value: Number to validate
        name: Name of the parameter (for error messages)

    Returns:
        The validated value

    Raises:
        ValidationError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{name} must be a number, got {type(value).__name__}",
            field_name=name,
            received_value=value,
        )

    if value <= 0:
        raise ValidationError(
            f"{name} must be positive, got {value}",
            field_name=name,
            received_value=value,
        )

    return value


def validate_non_negative_number(
    value: Union[int, float], name: str
) -> Union[int, float]:
    """
    Validate that a number is non-negative.

    Args:
        value: Number to validate
        name: Name of the parameter (for error messages)

    Returns:
        The validated value

    Raises:
        ValidationError: If value is negative
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{name} must be a number, got {type(value).__name__}",
            field_name=name,
            received_value=value,
        )

    if value < 0:
        raise ValidationError(
            f"{name} must be non-negative, got {value}",
            field_name=name,
            received_value=value,
        )

    return value


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValidationError: If required columns are missing
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"Expected pandas DataFrame, got {type(df).__name__}",
            received_value=type(df).__name__,
        )

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {sorted(df.columns)}",
            field_name="columns",
            received_value=list(df.columns),
        )


def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Validate that DataFrame is not empty.

    Args:
        df: DataFrame to validate
        name: Name of the DataFrame (for error messages)

    Raises:
        ValidationError: If DataFrame is empty
    """
    if df.empty:
        raise ValidationError(
            f"{name} cannot be empty (shape: {df.shape})",
            field_name=name,
            received_value=f"shape_{df.shape}",
        )


def validate_parameter_bounds(
    birth_rate: float, death_rate: float, max_value: float = 100.0
) -> None:
    """
    Validate that birth and death rate parameters are within reasonable bounds.

    Args:
        birth_rate: Birth rate parameter
        death_rate: Death rate parameter
        max_value: Maximum allowed parameter value

    Raises:
        ValidationError: If parameters are out of bounds
    """
    validate_positive_number(birth_rate, "birth_rate")
    validate_positive_number(death_rate, "death_rate")

    if birth_rate > max_value:
        raise ValidationError(
            f"birth_rate too large: {birth_rate} > {max_value}. "
            "This may indicate convergence issues.",
            field_name="birth_rate",
            received_value=birth_rate,
        )

    if death_rate > max_value:
        raise ValidationError(
            f"death_rate too large: {death_rate} > {max_value}. "
            "This may indicate convergence issues.",
            field_name="death_rate",
            received_value=death_rate,
        )


def validate_creation_times(creation_times: List[float], final_time: float) -> None:
    """
    Validate creation times list.

    Args:
        creation_times: List of creation times
        final_time: Final observation time

    Raises:
        ValidationError: If creation times are invalid
    """
    if not isinstance(creation_times, (list, tuple)):
        raise ValidationError(
            f"creation_times must be a list or tuple, got {type(creation_times).__name__}",
            field_name="creation_times",
            received_value=type(creation_times).__name__,
        )

    for i, time in enumerate(creation_times):
        if not isinstance(time, (int, float)):
            raise ValidationError(
                f"creation_times[{i}] must be a number, got {type(time).__name__}",
                field_name=f"creation_times[{i}]",
                received_value=time,
            )

        validate_non_negative_number(time, f"creation_times[{i}]")

        if time > final_time:
            raise ValidationError(
                f"creation_times[{i}] ({time}) cannot be after final_time ({final_time})",
                field_name=f"creation_times[{i}]",
                received_value=time,
            )
