"""Configuration for pytest tests with comprehensive fixtures and utilities."""

import os

# Force JAX to use CPU backend before any package import triggers JAX init.
# The A40-12Q vGPU does not support CUDA VMM required by XLA.
# Must be set BEFORE any import that transitively imports jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference import BirthDeathModel, Constraints, FitResult
from bdpinf.inference.fitting import StartSummary
from bdpinf.process import BirthDeathProcess


@pytest.fixture
def make_fit_result() -> Callable[..., FitResult]:
    """Return a factory for ready-made piecewise-constant fit results.

    The result is built directly (no optimisation), so tests can feed fitting
    consumers a realistic ``FitResult`` cheaply.
    """

    def factory(
        birth: Sequence[float],
        death: Sequence[float],
        *,
        times: Sequence[float] | None = None,
        converged: bool = True,
        nll_history: Sequence[float] = (10.0, 9.0),
        method: str = "trust-exact",
        log_likelihood: float = -9.0,
    ) -> FitResult:
        epochs = list(times) if times is not None else [float(len(birth))]
        if len(epochs) != len(birth):
            epochs = [float(k + 1) for k in range(len(birth))]
        model = BirthDeathModel.piecewise_constant(epochs)
        compiled = Constraints().compile(model.params)
        natural = np.concatenate([np.asarray(birth), np.asarray(death)]).astype(float)
        birth_fn, death_fn = model.rates(jnp.asarray(natural))
        return FitResult(
            model=model,
            constraints=Constraints(),
            params=dict(zip(model.names, natural.tolist(), strict=True)),
            birth=birth_fn,
            death=death_fn,
            log_likelihood=log_likelihood,
            converged=converged,
            score_norm=0.0,
            min_hessian_eig=1.0,
            at_interior_optimum=converged,
            covariance_free=None,
            covariance=None,
            covariance_unconstrained=None,
            n_iter=len(nll_history),
            n_function_evals=len(nll_history),
            message="ok",
            method=method,
            nll_history=list(nll_history),
            starts=[StartSummary(log_likelihood, converged, len(nll_history))],
            free_values=np.log(np.clip(natural, 1e-300, None)),
            compiled=compiled,
        )

    return factory


@pytest.fixture
def sample_creation_times() -> list[float]:
    """Sample creation times for testing."""
    return [100.0, 200.0, 300.0, 400.0, 500.0]


@pytest.fixture
def sample_model(sample_creation_times: list[float]) -> BirthDeathProcess:
    """Create a sample birth-death process model."""
    return BirthDeathProcess(
        creation_times=sample_creation_times,
        epoch_end_times=600.0,
        birth_rates=0.1,
        death_rates=0.05,
    )


@pytest.fixture
def unfitted_model(sample_creation_times: list[float]) -> BirthDeathProcess:
    """Create an unfitted birth-death process model."""
    return BirthDeathProcess(
        creation_times=sample_creation_times,
        epoch_end_times=600.0,
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


@pytest.fixture
def canonical_single_epoch_params() -> dict[str, Any]:
    """Canonical single-epoch parameters reused across regression and recovery tests."""
    return {
        "birth_rates": jnp.array([0.05], dtype=jnp.float64),
        "death_rates": jnp.array([0.03], dtype=jnp.float64),
        "times": jnp.array([1500.0], dtype=jnp.float64),
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
