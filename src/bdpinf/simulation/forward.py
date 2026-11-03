"""Adapters wrapping the `bdsurv` forward simulators.

The Gillespie, lifespan-at-birth and piecewise-linear simulators are
implemented in the standalone `bdsurv` package, which does not depend on this
package's inference stack.
`bdsurv` returns its own lightweight ``SimulatedProcess`` (a records DataFrame
plus the epoch ends and rates), since `BirthDeathProcess` pulls in
`bdpinf.inference`.
This module wraps those results into `BirthDeathProcess` so the rest of
`bdpinf` sees the same canonical type as every other simulator here.

`simulate_birth_death_lifespans_arrays` and the era-conditioning functions
return no `BirthDeathProcess`, so `bdpinf.simulation` re-exports them from
`bdsurv` directly.
"""

from __future__ import annotations

import numpy as np
from bdsurv import ConstantRateLike, LinearRateLike, SimulatedProcess
from bdsurv import simulate_birth_death_lifespans as _lifespans
from bdsurv import simulate_birth_death_linear as _linear
from bdsurv import simulate_birth_death_process as _gillespie
from numpy.typing import NDArray

from bdpinf.process import BirthDeathProcess

__all__ = [
    "simulate_birth_death_lifespans",
    "simulate_birth_death_linear",
    "simulate_birth_death_process",
]


def _to_birth_death_process(process: SimulatedProcess) -> BirthDeathProcess:
    return BirthDeathProcess.from_records(
        process.records,
        epoch_end_times=process.epoch_end_times,
        start_time=process.start_time,
        initial_population=process.initial_population,
        birth_rates=process.birth_rates,
        death_rates=process.death_rates,
    )


def simulate_birth_death_process(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> BirthDeathProcess:
    """Simulate individual data from the birth-death process with time-varying rates.

    Thin wrapper around ``bdsurv.simulate_birth_death_process`` that converts
    its ``SimulatedProcess`` result into a canonical ``BirthDeathProcess``.
    Individuals are born at rate λ and die at rate μ, with rates that change at
    the given times; the simulation runs to the final time and records both
    creation times and survival status.

    Parameters
    ----------
    birth_rates
        Birth rate(s) (λ) for individual creation. Either a single
        float or an array of rates corresponding to time periods.
    death_rates
        Death rate(s) (μ) for individual loss. Either a single
        float or an array of rates corresponding to time periods.
    times
        End time(s) for the simulation. Either a single float or an array
        of time points at which rates change. Must be the same length as
        birth_rates and death_rates if arrays are provided.
    random_seed
        Random seed for reproducibility.
    ensure_non_extinct
        If ``True``, retry until at least one individual
        survives to the terminal time.
    max_retries
        Maximum number of retry attempts when
        ``ensure_non_extinct=True``.

    Returns
    -------
    BirthDeathProcess
        Process containing the simulated individual records.

    """
    return _to_birth_death_process(
        _gillespie(
            birth_rates,
            death_rates,
            times,
            random_seed,
            ensure_non_extinct=ensure_non_extinct,
            max_retries=max_retries,
        )
    )


def simulate_birth_death_lifespans(
    birth_rates: float | NDArray[np.float64],
    death_rates: float | NDArray[np.float64],
    times: float | NDArray[np.float64],
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
    max_population: int = 200_000,
) -> BirthDeathProcess:
    """Simulate a birth-death process using the lifespan-at-birth algorithm.

    Thin wrapper around ``bdsurv.simulate_birth_death_lifespans`` that
    converts its ``SimulatedProcess`` result into a canonical
    ``BirthDeathProcess``.
    A JIT-compiled alternative to :func:`simulate_birth_death_process` with
    the same public API: each individual's entire lifespan is decided at birth
    from a piecewise-exponential distribution.
    The ``records`` table has an additional ``birth_epoch`` column.

    Parameters
    ----------
    birth_rates
        Birth rate(s) (λ). Scalar or 1-D array (one per epoch).
    death_rates
        Death rate(s) (μ). Same length as ``birth_rates``.
    times
        Epoch end time(s). Scalar or strictly-increasing 1-D array.
    random_seed
        Integer seed for reproducibility, or ``None``.
    ensure_non_extinct
        If ``True``, retry until at least one individual
        survives to the terminal time.
    max_retries
        Maximum number of retry attempts when
        ``ensure_non_extinct=True``.
    max_population
        Maximum total individuals that may be born; passed
        to the JIT core as a static buffer size.

    Returns
    -------
    BirthDeathProcess
        Simulated process with full genealogy and spine assignment.

    Raises
    ------
    ValueError
        On invalid inputs (lengths, non-positive rates, non-increasing times).
    RuntimeError
        If ``ensure_non_extinct=True`` and all retries produce an extinct
        population, or if ``max_population`` is exceeded.

    """
    return _to_birth_death_process(
        _lifespans(
            birth_rates,
            death_rates,
            times,
            random_seed,
            ensure_non_extinct=ensure_non_extinct,
            max_retries=max_retries,
            max_population=max_population,
        )
    )


def simulate_birth_death_linear(
    birth_rate: ConstantRateLike | LinearRateLike,
    death_rate: ConstantRateLike | LinearRateLike,
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> BirthDeathProcess:
    r"""Simulate a birth-death process whose rates are linear in time.

    Thin wrapper around ``bdsurv.simulate_birth_death_linear`` that converts
    its ``SimulatedProcess`` result into a canonical ``BirthDeathProcess``.
    Each event time is drawn by inverting the cumulative hazard of the
    superposed birth and death processes,
    $s = 2E / (a + \sqrt{a^2 + 2bE})$ with $E \sim \mathrm{Exp}(1)$,
    where $a$ and $b$ are the total hazard and its slope, both scaled by the
    number alive.
    The event is a birth with probability $\lambda(t) / (\lambda(t) + \mu(t))$.
    Genealogy follows :func:`simulate_birth_death_process`.

    Parameters
    ----------
    birth_rate, death_rate
        Per-individual rates over ``[0, T]`` with a common end time $T$,
        such as ``PiecewiseConstantRate`` or ``PiecewiseLinearRate``.
        Their epoch boundaries need not coincide.
    random_seed
        Seed for reproducibility.
    ensure_non_extinct
        Retry until at least one individual survives to $T$.
    max_retries
        Maximum retries when ``ensure_non_extinct`` is true.

    Returns
    -------
    BirthDeathProcess
        Simulated process with records, genealogy and spine.
        Its stored epochs are the merged segment boundaries,
        and it stores no per-epoch rates.

    Raises
    ------
    ValueError
        If the end times differ or ``max_retries`` is negative.
    RuntimeError
        If every retry went extinct.

    """
    return _to_birth_death_process(
        _linear(
            birth_rate,
            death_rate,
            random_seed,
            ensure_non_extinct=ensure_non_extinct,
            max_retries=max_retries,
        )
    )
