"""Configuration for pytest tests with comprehensive fixtures and utilities."""

# ruff: noqa: S101

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from inf_bp_texts.models.birth_death_process import BirthDeathProcess

# Add the src directory to the path so we can import inf_bp_texts
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_creation_times() -> list[float]:
    """Sample creation times for testing."""
    return [100.0, 200.0, 300.0, 400.0, 500.0]


@pytest.fixture
def sample_model(sample_creation_times: list[float]) -> BirthDeathProcess:
    """Create a sample birth-death process model."""
    return BirthDeathProcess(
        creation_times=sample_creation_times,
        final_time=600.0,
        birth_rate=0.1,
        death_rate=0.05,
    )


@pytest.fixture
def unfitted_model(sample_creation_times: list[float]) -> BirthDeathProcess:
    """Create an unfitted birth-death process model."""
    return BirthDeathProcess(creation_times=sample_creation_times, final_time=600.0)


@pytest.fixture
def sample_manuscript_data() -> pd.DataFrame:
    """Create sample manuscript DataFrame."""
    return pd.DataFrame(
        {
            "age": [100.0, 200.0, 150.0, 300.0, 250.0],
            "count": [1, 1, 1, 1, 1],
            "docID": ["MS001", "MS002", "MS003", "MS004", "MS005"],
            "creation_time": [500.0, 400.0, 450.0, 300.0, 350.0],
        },
    )


@pytest.fixture
def sample_fit_results() -> dict[str, Any]:
    """Sample fitting results for testing."""
    return {
        "success": True,
        "birth_rate": 0.15,
        "death_rate": 0.03,
        "log_likelihood": -150.5,
        "n_data_points": 50,
        "message": "Optimization completed successfully",
    }


# Utility functions for testing
MAX_RATE_VALUE = 100


def assert_valid_birth_death_rates(birth_rate: float, death_rate: float) -> None:
    """Assert that birth and death rates are valid."""
    assert birth_rate > 0, f"Birth rate must be positive: {birth_rate}"
    assert death_rate > 0, f"Death rate must be positive: {death_rate}"
    assert birth_rate < MAX_RATE_VALUE, f"Birth rate seems too large: {birth_rate}"
    assert death_rate < MAX_RATE_VALUE, f"Death rate seems too large: {death_rate}"


def assert_optimization_successful(fit_results: dict[str, Any]) -> None:
    """Assert that optimization was successful."""
    assert fit_results["success"], (
        f"Optimization failed: {fit_results.get('message', 'Unknown error')}"
    )
    assert fit_results["birth_rate"] is not None, "Birth rate is None"
    assert fit_results["death_rate"] is not None, "Death rate is None"
    assert fit_results["log_likelihood"] is not None, "Log likelihood is None"
    assert fit_results["log_likelihood"] is not None, "Log likelihood is None"
