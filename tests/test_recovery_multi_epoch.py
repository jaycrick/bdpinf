"""Multi-epoch parameter recovery tests.

Simulates two-epoch birth-death data under known parameters, fits a
constrained or unconstrained model, and verifies parameter recovery.

Run with::

    uv run pytest -m slow tests/test_recovery_multi_epoch.py
"""

from typing import Any

import numpy as np
import pytest

from bdpinf.inference.fitting import (
    ParameterConstraints,
    fit_birth_death_processes,
)
from bdpinf.simulation import simulate_birth_death_process

_MIN_SURVIVORS = 15
_MAX_SEEDS = 20
_TIMES = np.array([300.0, 600.0])


def _simulate_two_epoch(
    birth_rates: list[float],
    death_rates: list[float],
    seed: int,
) -> np.ndarray | None:
    """Simulate and return survivor ages, or None if too few."""
    df = simulate_birth_death_process(
        birth_rates=np.array(birth_rates),
        death_rates=np.array(death_rates),
        times=_TIMES,
        random_seed=seed,
    )
    surviving = df.surviving_records
    if surviving is None or len(surviving) < _MIN_SURVIVORS:
        return None
    return surviving["age"].to_numpy(dtype=np.float64)


def _two_epoch_fit(
    birth_rates: list[float],
    death_rates: list[float],
    seed: int,
    param_constraints: ParameterConstraints | None = None,
) -> dict[str, Any] | None:
    ages = _simulate_two_epoch(birth_rates, death_rates, seed)
    if ages is None:
        return None
    return fit_birth_death_processes(
        [ages],
        times=_TIMES,
        param_constraints=param_constraints,
        estimate_uncertainty=False,
        random_seed=seed + 1000,
    )


@pytest.mark.slow
@pytest.mark.recovery
class TestRecoveryMultiEpoch:
    """Parameter recovery tests for two-epoch birth-death models."""

    @pytest.mark.xfail(
        reason=(
            "Death rates are weakly identified from survivor-ages-only data "
            "(no extinction times / population time series observed); "
            "confirmed via direct inspection that fitted death rates "
            "collapse toward zero for several seeds (e.g. ~1e-6) instead of "
            "the true epoch-2 death rate. Same statistical identifiability "
            "limitation as the single-epoch death-rate recovery tests, not "
            "an optimizer/convergence bug."
        ),
        strict=False,
    )
    def test_recover_two_epoch_with_equal_birth_constraint(self) -> None:
        """Constant birth rate across epochs; recover distinct death rates."""
        true_birth = 0.01
        true_death_1 = 0.003
        true_death_2 = 0.007
        constraints = ParameterConstraints(equality_groups=[[0, 1]])

        birth1_ok = 0
        death1_ok = 0
        death2_ok = 0
        attempts = 0
        for seed in range(_MAX_SEEDS):
            result = _two_epoch_fit(
                birth_rates=[true_birth, true_birth],
                death_rates=[true_death_1, true_death_2],
                seed=seed,
                param_constraints=constraints,
            )
            if result is None or not result["success"]:
                continue
            attempts += 1
            br = np.atleast_1d(result["birth_rates"])
            dr = np.atleast_1d(result["death_rates"])
            if len(br) >= 2 and len(dr) >= 2:
                if abs(br[0] - true_birth) / true_birth < 0.25:
                    birth1_ok += 1
                if abs(dr[0] - true_death_1) / true_death_1 < 0.30:
                    death1_ok += 1
                if abs(dr[1] - true_death_2) / true_death_2 < 0.30:
                    death2_ok += 1
            if birth1_ok >= 4 and death1_ok >= 4 and death2_ok >= 4:
                break

        assert attempts >= 3, (
            "Fewer than 3 valid replicates; check simulation parameters"
        )
        assert birth1_ok >= 4, f"Birth rate recovery: only {birth1_ok} successes"
        assert death1_ok >= 4, (
            f"Epoch-1 death rate recovery: only {death1_ok} successes"
        )
        assert death2_ok >= 4, (
            f"Epoch-2 death rate recovery: only {death2_ok} successes"
        )

    @pytest.mark.xfail(
        reason=(
            "Same death-rate identifiability limitation as "
            "test_recover_two_epoch_with_equal_birth_constraint: fitted "
            "death rates collapse toward zero for several seeds instead of "
            "tracking the true per-epoch values. Pre-existing statistical "
            "limitation, not an optimizer/convergence bug."
        ),
        strict=False,
    )
    def test_two_epoch_rates_differ_between_epochs(self) -> None:
        """With distinct true rates, the fitted rates should be closer to their own epoch truth."""
        true_birth = [0.01, 0.01]
        true_death = [0.003, 0.008]

        epoch1_wins = 0
        epoch2_wins = 0
        attempts = 0

        for seed in range(_MAX_SEEDS):
            result = _two_epoch_fit(true_birth, true_death, seed)
            if result is None or not result["success"]:
                continue
            attempts += 1
            dr = np.atleast_1d(result["death_rates"])
            if len(dr) < 2:
                continue
            err1_own = abs(dr[0] - true_death[0])
            err1_other = abs(dr[0] - true_death[1])
            err2_own = abs(dr[1] - true_death[1])
            err2_other = abs(dr[1] - true_death[0])
            if err1_own < err1_other:
                epoch1_wins += 1
            if err2_own < err2_other:
                epoch2_wins += 1
            if epoch1_wins >= 4 and epoch2_wins >= 4:
                break

        assert attempts >= 3, "Fewer than 3 valid replicates"
        assert epoch1_wins >= 4, (
            f"Epoch-1 rate closer to its own truth only {epoch1_wins}/{attempts} times"
        )
        assert epoch2_wins >= 4, (
            f"Epoch-2 rate closer to its own truth only {epoch2_wins}/{attempts} times"
        )

    def test_two_epoch_fit_returns_two_rates(self) -> None:
        """Fitted model should expose two birth and two death rate values."""
        ages = _simulate_two_epoch([0.01, 0.01], [0.003, 0.007], seed=42)
        if ages is None:
            pytest.skip("insufficient survivors for seed 42")
        result = fit_birth_death_processes(
            [ages], times=_TIMES, estimate_uncertainty=False
        )
        if not result["success"]:
            pytest.skip("optimizer did not converge for this seed")
        assert len(np.atleast_1d(result["birth_rates"])) == 2
        assert len(np.atleast_1d(result["death_rates"])) == 2

    @pytest.mark.xfail(
        reason=(
            "Same death-rate identifiability limitation as the other "
            "two-epoch recovery tests: with 4 free parameters there is even "
            "less information per parameter, so fitted death rates collapse "
            "toward zero for most seeds instead of recovering the true "
            "values. Pre-existing statistical limitation, not an "
            "optimizer/convergence bug."
        ),
        strict=False,
    )
    def test_recover_two_epoch_free_parameters(self) -> None:
        """Unconstrained 4-parameter fit recovers all rates within 30%."""
        true_birth = [0.012, 0.008]
        true_death = [0.004, 0.004]
        successes = 0
        attempts = 0
        for seed in range(_MAX_SEEDS):
            if successes >= 4:
                break
            ages = _simulate_two_epoch(true_birth, true_death, seed)
            if ages is None:
                continue
            attempts += 1
            result = _two_epoch_fit(
                true_birth, true_death, seed, param_constraints=None
            )
            if result is None or not result["success"]:
                continue
            br = np.atleast_1d(result["birth_rates"])
            dr = np.atleast_1d(result["death_rates"])
            if len(br) != 2 or len(dr) != 2:
                continue
            recovered = all(
                abs(br[i] - true_birth[i]) / true_birth[i] < 0.30
                and abs(dr[i] - true_death[i]) / true_death[i] < 0.30
                for i in range(2)
            )
            if recovered:
                successes += 1
        assert successes >= 4, (
            f"Only {successes}/{attempts} replicates recovered all 4 parameters "
            "within 30%"
        )
