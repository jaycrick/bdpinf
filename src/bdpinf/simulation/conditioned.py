"""Adapters wrapping the `bdsurv` survival-conditioned simulators.

`simulate_contour_given_survival` (exact contour-process reconstruction) and
`simulate_survival_splitting` (adaptive-grid RESTART multilevel splitting)
are implemented in the standalone `bdsurv` package, which does not depend on
this package's inference stack.
`bdsurv` returns its own lightweight `SimulatedTree` rather than
`BirthDeathProcess`, since `BirthDeathProcess` pulls in `bdpinf.inference`.
This module wraps `bdsurv`'s results into `BirthDeathProcess`, through the
converter in :mod:`bdpinf.simulation.forward`, so the rest of `bdpinf` sees
the same canonical type as every other simulator here.

`SplittingConfig` and `select_level_times_for_band` carry no
`BirthDeathProcess` dependency, so import them from `bdsurv` directly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from bdsurv import RateInput, SplittingConfig, SplittingSimulationResult
from bdsurv import simulate_contour_given_survival as _contour
from bdsurv import simulate_survival_splitting as _splitting

from bdpinf.process import BirthDeathProcess

from .forward import tree_to_birth_death_process

__all__ = [
    "ConditionedTrajectory",
    "simulate_contour_given_survival",
    "simulate_survival_splitting",
]


@dataclasses.dataclass(frozen=True)
class ConditionedTrajectory:
    """A sampled conditioned population path and associated process record.

    Parameters
    ----------
    particle_id : int
        Identifier for the particle that produced this conditioned trajectory.
    times : np.ndarray
        Monotone array of observation times for the population trace.
    populations : np.ndarray
        Population sizes corresponding to ``times``.
    process : BirthDeathProcess
        Individual-level birth-death process reconstructed from particle
        ancestry records.

    """

    particle_id: int
    times: np.ndarray
    populations: np.ndarray
    process: BirthDeathProcess


def simulate_contour_given_survival(
    birth_rate: RateInput,
    death_rate: RateInput,
    terminal_time: float | None = None,
    random_seed: int | None = None,
) -> BirthDeathProcess:
    """Simulate a birth-death process conditioned on survival to the end.

    Wraps ``bdsurv.simulate_contour_given_survival``, an exact contour-process
    reconstruction;
    see its docstring for the algorithm and the parameters.

    Returns
    -------
    BirthDeathProcess
        Process with exactly one surviving spine individual.

    """
    return tree_to_birth_death_process(
        _contour(birth_rate, death_rate, terminal_time, random_seed)
    )


def simulate_survival_splitting(config: SplittingConfig) -> SplittingSimulationResult:
    """Run adaptive-grid multilevel (RESTART) splitting for survival.

    Wraps ``bdsurv.simulate_survival_splitting``;
    see its docstring and ``bdsurv.SplittingConfig`` for the algorithm and the
    (keyword-only) configuration.

    Returns
    -------
    SplittingSimulationResult
        The ``bdsurv`` result, with each conditioned trajectory's tree
        converted to a ``BirthDeathProcess``
        (see :class:`ConditionedTrajectory`).

    """
    result = _splitting(config)
    trajectories = [
        ConditionedTrajectory(
            particle_id=trajectory.particle_id,
            times=trajectory.times,
            populations=trajectory.populations,
            process=tree_to_birth_death_process(trajectory.process),
        )
        for trajectory in result.conditioned_trajectories
    ]
    return dataclasses.replace(result, conditioned_trajectories=trajectories)
