"""Adapters wrapping the `bdsurv` forward simulators.

The exact event simulator and the lifespan-at-birth simulator are implemented
in the standalone `bdsurv` package, which does not depend on this package's
inference stack.
`bdsurv` returns its own lightweight ``SimulatedTree`` (a records DataFrame
plus the rates and the terminal time), since `BirthDeathProcess` pulls in
`bdpinf.inference`.
This module wraps those results into `BirthDeathProcess` so the rest of
`bdpinf` sees the same canonical type as every other simulator here.
`tree_to_birth_death_process` is the one internal converter (not exported), shared with
:mod:`bdpinf.simulation.conditioned`.

`simulate_birth_death_lifespans_arrays` and the era-conditioning functions
return no `BirthDeathProcess`, so `bdpinf.simulation` re-exports them from
`bdsurv` directly.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from bdsurv import (
    ConstantRateLike,
    LinearRateLike,
    LogisticHazard,
    PiecewiseHazard,
    RateInput,
    SimulatedTree,
    SumHazard,
)
from bdsurv import simulate_birth_death as _simulate_birth_death
from bdsurv import simulate_birth_death_lifespans as _lifespans
from numpy.typing import NDArray

from bdpinf.models.rate_functions import (
    LogisticRate,
    PiecewiseCubicRate,
    PiecewiseQuadraticRate,
)
from bdpinf.process import BirthDeathProcess

__all__ = [
    "simulate_birth_death",
    "simulate_birth_death_lifespans",
]


def to_bdsurv_rate(rate: object) -> RateInput:
    """Convert a spline rate to a ``bdsurv`` hazard; pass anything else through.

    :class:`~bdpinf.models.rate_functions.PiecewiseQuadraticRate` and
    :class:`~bdpinf.models.rate_functions.PiecewiseCubicRate` become a
    ``bdsurv.PiecewiseHazard`` built with ``PiecewiseHazard.from_polynomials``
    from their local-time coefficients (needs ``bdsurv`` 0.8 or later).
    ``bdsurv`` checks that the rate is non-negative on ``[0, T]``.
    :class:`~bdpinf.models.rate_functions.LogisticRate` becomes
    ``bdsurv.LogisticHazard(L, k, c)``, or
    ``bdsurv.SumHazard.with_floor(floor, ...)`` when it has a floor
    (needs ``bdsurv`` 0.9 or later; ``k`` must be non-zero and the parameters
    concrete).
    A hazard has no end time, so pass ``terminal_time`` to the simulators.
    """
    if isinstance(rate, LogisticRate):
        hazard = LogisticHazard(float(rate.L), float(rate.k), float(rate.c))  # pyright: ignore[reportArgumentType]
        return cast("RateInput", SumHazard.with_floor(float(rate.floor), hazard))  # pyright: ignore[reportArgumentType]
    if isinstance(rate, PiecewiseQuadraticRate | PiecewiseCubicRate):
        return PiecewiseHazard.from_polynomials(
            rate.local_coefficients(),
            np.asarray(rate.epoch_end_times, dtype=np.float64),
        )
    return cast("RateInput", rate)


def _epoch_ends(
    rate: ConstantRateLike | LinearRateLike, terminal_time: float
) -> NDArray[np.float64]:
    """Epoch end times; a smooth hazard has one epoch, ending at ``terminal_time``."""
    ends = getattr(rate, "epoch_end_times", None)
    if ends is None:
        return np.array([float(terminal_time)], dtype=np.float64)
    return np.atleast_1d(np.asarray(ends, dtype=np.float64))


def tree_to_birth_death_process(tree: SimulatedTree) -> BirthDeathProcess:
    """Convert a ``bdsurv.SimulatedTree`` into a ``BirthDeathProcess``.

    When both rates are piecewise constant on the same epochs, the process
    stores the per-epoch rates and the epoch end times.
    Otherwise (linear rates, or constant rates on different epochs) it stores
    no per-epoch rates, and its epochs are the sorted union of the two rates'
    epoch end times.
    The process starts at time 0 from one individual, as `bdsurv` always
    simulates.
    """
    birth, death = tree.birth_rate, tree.death_rate
    birth_ends = _epoch_ends(birth, tree.terminal_time)
    death_ends = _epoch_ends(death, tree.terminal_time)
    if (
        hasattr(birth, "values")
        and hasattr(death, "values")
        and np.array_equal(birth_ends, death_ends)
    ):
        return BirthDeathProcess.from_records(
            tree.records,
            epoch_end_times=birth_ends,
            start_time=0.0,
            initial_population=1,
            birth_rates=np.asarray(
                cast("ConstantRateLike", birth).values, dtype=np.float64
            ),
            death_rates=np.asarray(
                cast("ConstantRateLike", death).values, dtype=np.float64
            ),
        )
    return BirthDeathProcess.from_records(
        tree.records,
        epoch_end_times=np.union1d(birth_ends, death_ends),
        start_time=0.0,
        initial_population=1,
    )


def simulate_birth_death(
    birth_rate: RateInput,
    death_rate: RateInput,
    terminal_time: float | None = None,
    random_seed: int | None = None,
    *,
    ensure_non_extinct: bool = False,
    max_retries: int = 100,
) -> BirthDeathProcess:
    """Simulate a birth-death process by exact event simulation.

    Wraps ``bdsurv.simulate_birth_death``;
    see its docstring for the algorithm and the parameters.
    Rates are floats or rate objects (``bdsurv.PiecewiseConstant``,
    ``bdsurv.PiecewiseLinear``, or ``bdpinf``'s ``PiecewiseConstantRate`` and
    ``PiecewiseLinearRate``).

    Returns
    -------
    BirthDeathProcess
        Simulated process with records, genealogy and spine.
        For piecewise-constant rates on shared epochs it stores the per-epoch
        rates and epoch end times.
        Otherwise its stored epochs are the merged segment boundaries,
        and it stores no per-epoch rates.

    """
    return tree_to_birth_death_process(
        _simulate_birth_death(
            to_bdsurv_rate(birth_rate),
            to_bdsurv_rate(death_rate),
            terminal_time,
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
    """Simulate a birth-death process with the lifespan-at-birth algorithm.

    Wraps ``bdsurv.simulate_birth_death_lifespans``;
    see its docstring for the algorithm and the parameters.
    Rates are constant per epoch, given as scalars or arrays.

    Returns
    -------
    BirthDeathProcess
        Simulated process with genealogy and spine.
        Its ``records`` table has an additional ``birth_epoch`` column.

    """
    return tree_to_birth_death_process(
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
