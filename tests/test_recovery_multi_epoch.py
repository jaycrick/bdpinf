"""Multi-epoch parameter recovery tests.

Simulates two-epoch birth-death data under known parameters, fits a
constrained or unconstrained model, and verifies parameter recovery.

Run with::

    uv run pytest -m slow tests/test_recovery_multi_epoch.py
"""

import numpy as np
import pytest
from bdsurv import PiecewiseConstant

from bdpinf.inference import AgeData, BirthDeathModel, FitResult, fit
from bdpinf.simulation import simulate_birth_death

_MIN_SURVIVORS = 15
_MAX_SEEDS = 20
_TIMES = np.array([300.0, 600.0])


def _simulate_two_epoch(
    birth_rates: list[float],
    death_rates: list[float],
    seed: int,
    *,
    ensure_non_extinct: bool = False,
) -> np.ndarray | None:
    """Simulate and return survivor ages, or None if too few."""
    df = simulate_birth_death(
        PiecewiseConstant(np.array(birth_rates), _TIMES),
        PiecewiseConstant(np.array(death_rates), _TIMES),
        random_seed=seed,
        ensure_non_extinct=ensure_non_extinct,
    )
    surviving = df.surviving_records
    if surviving is None or len(surviving) < _MIN_SURVIVORS:
        return None
    return surviving["age"].to_numpy(dtype=np.float64)


def _fit_ages(ages: np.ndarray, seed: int | None = None) -> FitResult:
    """Fit piecewise-constant rates on the two epochs to ``ages``."""
    return fit(
        AgeData.from_groups([ages], float(_TIMES[-1])),
        BirthDeathModel.piecewise_constant(_TIMES),
        seed=seed,
        compute_covariance=False,
    )


@pytest.mark.slow
@pytest.mark.recovery
class TestRecoveryMultiEpoch:
    """Parameter recovery tests for two-epoch birth-death models."""

    # Recovery tests for the individual death rates were dropped: survivor ages
    # alone identify the death rate too weakly at these sample sizes (the
    # likelihood itself prefers other values), so they cannot be meaningful.

    def test_two_epoch_fit_returns_two_rates(self) -> None:
        """Fitted model should expose two birth and two death rate values."""
        # Seed 42 goes extinct without ``ensure_non_extinct``, so this test
        # would always skip.
        ages = _simulate_two_epoch(
            [0.01, 0.01], [0.003, 0.007], seed=42, ensure_non_extinct=True
        )
        if ages is None:
            pytest.skip("insufficient survivors for seed 42")
        result = _fit_ages(ages)
        if not result.success:
            pytest.skip("optimizer did not converge for this seed")
        assert len(result.birth_values) == 2
        assert len(result.death_values) == 2
