"""Shared dataclasses for the fitting module.

Isolated here so that _optimizer_runners can import them without
creating a circular dependency with fitting.py.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax

DEFAULT_GRADIENT_THRESHOLD = 5e-4
DEFAULT_ABSOLUTE_TOLERANCE = 1e-7
DEFAULT_PATIENCE = 5
DEFAULT_ITERATIONS = 10000
MIN_EQUALITY_GROUP_SIZE = 2


@dataclass(frozen=True)
class OptimizerStrategy:
    """Configuration for a single optimizer strategy."""

    name: str
    solver_factory: Callable[..., Any]
    max_iters: int
    solver_kwargs: dict[str, Any]


@dataclass
class OptimizerRunResult:
    """Result payload from running optimizer strategies."""

    log_params: jax.Array
    converged: bool
    selected_attempt_index: int | None
    attempts: list["OptimizerAttemptResult"]


@dataclass(frozen=True)
class OptimizerAttemptResult:
    """Result payload for a single optimizer attempt."""

    optimizer: str
    attempt_index: int
    converged: bool
    iterations: int
    nll_history: list[float]
    exception_type: str | None = None
    exception_message: str | None = None
    grad_norms: list[float] | None = None
    param_history: list[list[float]] | None = None
    timestamps: list[float] | None = None


@dataclass(frozen=True)
class ConvergenceConfig:
    """Configuration for convergence criteria evaluation."""

    gradient_threshold: float
    absolute_tolerance: float
    patience: int

    @classmethod
    def default(cls) -> "ConvergenceConfig":
        """Create configuration with default values."""
        return cls(
            gradient_threshold=DEFAULT_GRADIENT_THRESHOLD,
            absolute_tolerance=DEFAULT_ABSOLUTE_TOLERANCE,
            patience=DEFAULT_PATIENCE,
        )
