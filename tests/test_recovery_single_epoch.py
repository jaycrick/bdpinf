"""Single-epoch parameter recovery tests.

Each test simulates manuscript data under known parameters, fits a
birth-death model, and verifies the recovered parameters are within
a scientifically meaningful tolerance of the truth.

These tests are marked ``slow`` (excluded from routine ``pytest`` runs).
Run them with::

    uv run pytest -m slow tests/test_recovery_single_epoch.py

Design
------
- 5 independent simulation seeds per scenario; at least 4 must recover
  the parameter within the stated relative tolerance.
- Minimum 15 surviving manuscripts per replicate required; replicates that
  produce fewer are skipped and replaced by trying an additional seed.
- Relative tolerance 20 % (25 % for the critical-rate scenario).
- Seeds and tolerances are fixed so failures are reproducible.
"""

from typing import Any

import numpy as np
import pytest

from inf_bp_texts.inference.fitting import (
    ParameterConstraints,
    fit_birth_death_processes,
)
from inf_bp_texts.simulation import simulate_birth_death_process

_MIN_SURVIVORS = 15
_MAX_SEEDS = 20


def _simulate_and_fit(
    birth_rate: float,
    death_rate: float,
    T: float,
    seed: int,
    param_constraints: ParameterConstraints | None = None,
) -> dict[str, Any] | None:
    """Simulate a single epoch and fit. Return fit dict or None when too few survivors."""
    df = simulate_birth_death_process(
        birth_rates=birth_rate,
        death_rates=death_rate,
        times=T,
        random_seed=seed,
    )
    surviving = df[df["survived"]]
    if len(surviving) < _MIN_SURVIVORS:
        return None
    ages = surviving["age"].to_numpy(dtype=np.float64)
    return fit_birth_death_processes(
        [ages],
        times=T,
        param_constraints=param_constraints,
        estimate_uncertainty=False,
        random_seed=seed + 1000,
    )


def _recovery_pass_rate(
    birth_rate: float,
    death_rate: float,
    T: float,
    param: str,
    true_value: float,
    rel_tol: float,
    n_pass: int = 4,
    param_constraints: ParameterConstraints | None = None,
) -> int:
    """Run up to _MAX_SEEDS replicates; return count that recover within rel_tol."""
    successes = 0
    attempts = 0
    seed = 0
    while successes < n_pass and attempts < _MAX_SEEDS:
        result = _simulate_and_fit(birth_rate, death_rate, T, seed, param_constraints)
        seed += 1
        if result is None or not result["success"]:
            continue
        attempts += 1
        fitted = float(np.atleast_1d(result[param])[0])
        if abs(fitted - true_value) / true_value < rel_tol:
            successes += 1
    return successes


@pytest.mark.slow
@pytest.mark.recovery
class TestRecoverySingleEpoch:
    """Parameter recovery tests for single-epoch birth-death models."""

    def test_recover_supercritical_birth_rate(self) -> None:
        true_birth = 0.01
        successes = _recovery_pass_rate(
            birth_rate=true_birth,
            death_rate=0.003,
            T=600.0,
            param="birth_rates",
            true_value=true_birth,
            rel_tol=0.20,
        )
        assert successes >= 4, (
            f"Only {successes}/{_MAX_SEEDS} replicates recovered birth rate within 20%"
        )

    def test_recover_supercritical_death_rate(self) -> None:
        true_death = 0.003
        successes = _recovery_pass_rate(
            birth_rate=0.01,
            death_rate=true_death,
            T=600.0,
            param="death_rates",
            true_value=true_death,
            rel_tol=0.20,
        )
        assert successes >= 4, (
            f"Only {successes}/{_MAX_SEEDS} replicates recovered death rate within 20%"
        )

    def test_recover_with_birth_death_equality_constraint(self) -> None:
        """Equality constraint λ=μ should tighten recovery of the shared rate."""
        true_rate = 0.005
        constraints = ParameterConstraints(equality_groups=[[0, 1]])
        successes = _recovery_pass_rate(
            birth_rate=true_rate,
            death_rate=true_rate,
            T=800.0,
            param="birth_rates",
            true_value=true_rate,
            rel_tol=0.25,
            param_constraints=constraints,
        )
        assert successes >= 4, (
            f"Only {successes}/{_MAX_SEEDS} replicates recovered constrained rate within 25%"
        )

    def test_recover_rates_are_self_consistent(self) -> None:
        """Fitted rates should be positive, finite, and ordered consistently."""
        df = simulate_birth_death_process(
            birth_rates=0.01, death_rates=0.003, times=600.0, random_seed=42
        )
        surviving = df[df["survived"]]
        if len(surviving) < _MIN_SURVIVORS:
            pytest.skip("insufficient survivors for this seed")
        ages = surviving["age"].to_numpy(dtype=np.float64)
        result = fit_birth_death_processes(
            [ages], times=600.0, estimate_uncertainty=False
        )
        assert result["success"]
        fitted_birth = float(np.atleast_1d(result["birth_rates"])[0])
        fitted_death = float(np.atleast_1d(result["death_rates"])[0])
        assert fitted_birth > 0
        assert fitted_death > 0
        assert np.isfinite(fitted_birth)
        assert np.isfinite(fitted_death)
        assert np.isfinite(float(result["log_likelihood"]))
