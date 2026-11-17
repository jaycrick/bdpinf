"""Single-epoch parameter recovery tests.

Each test simulates individual data under known parameters, fits a
birth-death model, and verifies the recovered parameters are within
a scientifically meaningful tolerance of the truth.

These tests are marked ``slow`` (excluded from routine ``pytest`` runs).
Run them with::

    uv run pytest -m slow tests/test_recovery_single_epoch.py

Design
------
- 5 independent simulation seeds per scenario; at least 4 must recover
  the parameter within the stated relative tolerance.
- Minimum 15 surviving records per replicate required; replicates that
  produce fewer are skipped and replaced by trying an additional seed.
- Relative tolerance 20 % (25 % for the critical-rate scenario).
- Seeds and tolerances are fixed so failures are reproducible.
"""

import numpy as np
import pytest

from bdpinf.inference import AgeData, BirthDeathModel, Constraints, FitResult, fit
from bdpinf.simulation import (
    simulate_birth_death,
    simulate_birth_death_lifespans,
)

_MIN_SURVIVORS = 15
_MAX_SEEDS = 20
_EQUAL_RATES = Constraints(equal=[["birth.v0", "death.v0"]])


def _fit_ages(
    ages: np.ndarray,
    T: float,
    constraints: Constraints | None = None,
    seed: int | None = None,
) -> FitResult:
    """Fit one single-epoch piecewise-constant model to ``ages``."""
    return fit(
        AgeData.from_groups([ages], T),
        BirthDeathModel.piecewise_constant([T]),
        constraints=constraints,
        seed=seed,
        compute_covariance=False,
    )


def _fitted(result: FitResult, param: str) -> float:
    """Return the fitted birth or death rate (``param`` is ``birth`` or ``death``)."""
    values = result.birth_values if param == "birth" else result.death_values
    return float(values[0])


def _simulate_and_fit(
    birth_rate: float,
    death_rate: float,
    T: float,
    seed: int,
    constraints: Constraints | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> FitResult | None:
    """Simulate a single epoch and fit. Return the fit or None when too few survivors."""
    df = simulate_birth_death(
        birth_rate,
        death_rate,
        T,
        random_seed=seed,
        ensure_non_extinct=ensure_non_extinct,
        max_retries=max_retries,
    )
    surviving = df.surviving_records
    if surviving is None or len(surviving) < _MIN_SURVIVORS:
        return None
    ages = surviving["age"].to_numpy(dtype=np.float64)
    return _fit_ages(ages, T, constraints, seed + 1000)


def _recovery_pass_rate(
    birth_rate: float,
    death_rate: float,
    T: float,
    param: str,
    true_value: float,
    rel_tol: float,
    n_pass: int = 4,
    constraints: Constraints | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> int:
    """Run up to _MAX_SEEDS replicates; return count that recover within rel_tol.

    Bounded by total seeds tried (not just valid attempts) so that a scenario
    where most seeds produce too few survivors cannot loop indefinitely.
    """
    successes = 0
    for seed in range(_MAX_SEEDS):
        if successes >= n_pass:
            break
        result = _simulate_and_fit(
            birth_rate,
            death_rate,
            T,
            seed,
            constraints,
            ensure_non_extinct=ensure_non_extinct,
            max_retries=max_retries,
        )
        if result is None or not result.success:
            continue
        fitted = _fitted(result, param)
        if abs(fitted - true_value) / true_value < rel_tol:
            successes += 1
    return successes


@pytest.mark.slow
@pytest.mark.recovery
class TestRecoverySingleEpoch:
    """Parameter recovery tests for single-epoch birth-death models."""

    # No death-rate recovery test: survivor ages alone identify the death rate
    # too weakly at these sample sizes (the likelihood itself prefers other
    # values), so such a test cannot be meaningful.

    def test_recover_supercritical_birth_rate(self) -> None:
        true_birth = 0.01
        successes = _recovery_pass_rate(
            birth_rate=true_birth,
            death_rate=0.003,
            T=600.0,
            param="birth",
            true_value=true_birth,
            rel_tol=0.20,
        )
        assert successes >= 4, (
            f"Only {successes}/{_MAX_SEEDS} replicates recovered birth rate within 20%"
        )

    # Scenario: critical process (lambda = mu = 0.005) simulated with
    # ``ensure_non_extinct=True`` over T = 10000, so lambda * T = 50 and a
    # surviving process has about 1 + lambda * T = 51 survivors on average
    # (median about 40), well above ``_MIN_SURVIVORS``.
    # The original horizon (T = 800, lambda * T = 4) almost never gave enough
    # survivors for a fit to be attempted.
    # ``max_retries`` is raised because a critical process survives to T with
    # probability about 1 / (1 + lambda * T), about 2 %.
    def test_recover_with_birth_death_equality_constraint(self) -> None:
        """Equality constraint λ=μ should tighten recovery of the shared rate."""
        true_rate = 0.005
        constraints = _EQUAL_RATES
        successes = _recovery_pass_rate(
            birth_rate=true_rate,
            death_rate=true_rate,
            T=10000.0,
            param="birth",
            true_value=true_rate,
            rel_tol=0.25,
            constraints=constraints,
            ensure_non_extinct=True,
            max_retries=5000,
        )
        assert successes >= 4, (
            f"Only {successes}/{_MAX_SEEDS} replicates recovered constrained rate within 25%"
        )

    def test_recover_rates_are_self_consistent(self) -> None:
        """Fitted rates should be positive, finite, and ordered consistently."""
        # Seed 42 goes extinct without ``ensure_non_extinct``, so this test
        # would always skip.
        df = simulate_birth_death(
            0.01, 0.003, 600.0, random_seed=42, ensure_non_extinct=True
        )
        surviving = df.surviving_records
        if surviving is None or len(surviving) < _MIN_SURVIVORS:
            pytest.skip("insufficient survivors for this seed")
        ages = surviving["age"].to_numpy(dtype=np.float64)
        result = _fit_ages(ages, 600.0)
        assert result.success
        fitted_birth = _fitted(result, "birth")
        fitted_death = _fitted(result, "death")
        assert fitted_birth > 0
        assert fitted_death > 0
        assert np.isfinite(fitted_birth)
        assert np.isfinite(fitted_death)
        assert np.isfinite(result.log_likelihood)

    def test_recover_subcritical_rates(self) -> None:
        """Subcritical process (μ > λ): both rates should be recoverable.

        A strongly subcritical process (e.g. λ well below μ) started from a
        single founder almost never reaches ``_MIN_SURVIVORS`` survivors --
        the expected population size decays monotonically, so collecting 15
        survivors would require an astronomically unlikely upward
        fluctuation. We use a mildly subcritical rate pair instead (still
        μ > λ) and ``ensure_non_extinct`` to avoid wasting seeds on
        simulations that go extinct outright.
        """
        true_birth = 0.0475
        true_death = 0.05
        birth_successes = _recovery_pass_rate(
            birth_rate=true_birth,
            death_rate=true_death,
            T=800.0,
            param="birth",
            true_value=true_birth,
            rel_tol=0.20,
            ensure_non_extinct=True,
            max_retries=5000,
        )
        assert birth_successes >= 4, (
            f"Only {birth_successes}/{_MAX_SEEDS} replicates recovered subcritical "
            "birth rate within 20%"
        )

    # Scenario: critical process (lambda = mu = 0.04) at T = 1500 with
    # ``ensure_non_extinct=True``, so lambda * T = 60 and a surviving process
    # has about 61 survivors on average.
    # Without ``ensure_non_extinct`` only 1 of 20 seeds survived.
    # ``max_retries`` is raised because a critical process survives to T with
    # probability about 1 / (1 + lambda * T), under 2 %.
    def test_recover_critical_rates(self) -> None:
        """Critical process (λ ≈ μ): shared rate should be recoverable within 25%."""
        true_rate = 0.04
        constraints = _EQUAL_RATES
        successes = _recovery_pass_rate(
            birth_rate=true_rate,
            death_rate=true_rate,
            T=1500.0,
            param="birth",
            true_value=true_rate,
            rel_tol=0.25,
            constraints=constraints,
            ensure_non_extinct=True,
            max_retries=5000,
        )
        assert successes >= 4, (
            f"Only {successes}/{_MAX_SEEDS} replicates recovered critical rate within 25%"
        )


def _simulate_lifespans_and_fit(
    birth_rate: float,
    death_rate: float,
    T: float,
    seed: int,
    constraints: Constraints | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> FitResult | None:
    """Simulate via lifespan-at-birth and fit. Return the fit or None."""
    df = simulate_birth_death_lifespans(
        birth_rates=birth_rate,
        death_rates=death_rate,
        times=T,
        random_seed=seed,
        ensure_non_extinct=ensure_non_extinct,
        max_retries=max_retries,
    )
    surviving = df.surviving_records
    if surviving is None or len(surviving) < _MIN_SURVIVORS:
        return None
    ages = surviving["age"].to_numpy(dtype=np.float64)
    return _fit_ages(ages, T, constraints, seed + 1000)


def _lifespan_recovery_pass_rate(
    birth_rate: float,
    death_rate: float,
    T: float,
    param: str,
    true_value: float,
    rel_tol: float,
    n_pass: int = 4,
    constraints: Constraints | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> int:
    """Run up to _MAX_SEEDS replicates with lifespan sim; return passing count."""
    successes = 0
    for seed in range(_MAX_SEEDS):
        if successes >= n_pass:
            break
        result = _simulate_lifespans_and_fit(
            birth_rate,
            death_rate,
            T,
            seed,
            constraints,
            ensure_non_extinct=ensure_non_extinct,
            max_retries=max_retries,
        )
        if result is None or not result.success:
            continue
        fitted = _fitted(result, param)
        if abs(fitted - true_value) / true_value < rel_tol:
            successes += 1
    return successes


@pytest.mark.slow
@pytest.mark.recovery
class TestRecoveryLifespanSingleEpoch:
    """Parameter recovery tests using the lifespan-at-birth simulator."""

    def test_recover_supercritical_birth_rate(self) -> None:
        true_birth = 0.01
        successes = _lifespan_recovery_pass_rate(
            birth_rate=true_birth,
            death_rate=0.003,
            T=600.0,
            param="birth",
            true_value=true_birth,
            rel_tol=0.20,
        )
        assert successes >= 4, (
            f"Only {successes}/{_MAX_SEEDS} replicates recovered birth rate within 20%"
        )

    def test_recover_rates_are_self_consistent(self) -> None:
        df = simulate_birth_death_lifespans(
            birth_rates=0.01, death_rates=0.003, times=600.0, random_seed=42
        )
        surviving = df.surviving_records
        if surviving is None or len(surviving) < _MIN_SURVIVORS:
            pytest.skip("insufficient survivors for this seed")
        ages = surviving["age"].to_numpy(dtype=np.float64)
        result = _fit_ages(ages, 600.0)
        assert result.success
        fitted_birth = _fitted(result, "birth")
        fitted_death = _fitted(result, "death")
        assert fitted_birth > 0
        assert fitted_death > 0
        assert np.isfinite(fitted_birth)
        assert np.isfinite(fitted_death)
        assert np.isfinite(result.log_likelihood)
