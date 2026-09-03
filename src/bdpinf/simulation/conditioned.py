"""Adapters wrapping the `bdsurv` survival-conditioned simulators.

`simulate_birth_death_process_from_contour_process` (exact contour-process
reconstruction) and `simulate_subcritical_survival_splitting` (fixed-factor
RESTART multilevel splitting) were extracted into the standalone `bdsurv`
package so the simulation methodology could be published independently of
this package's inference stack. `bdsurv` returns its own lightweight
`SimulatedTree` (a records DataFrame plus rates and terminal time, with no
dependency on this package) rather than `BirthDeathProcess`, since
`BirthDeathProcess` pulls in `bdpinf.inference`. This module re-wraps
`bdsurv`'s results into `BirthDeathProcess` so the rest of `bdpinf` sees the
same canonical type as every other simulator here.

`SplittingConfig` and `select_level_times_for_band` need no adaptation --
they carry no `BirthDeathProcess`/`SimulatedTree` dependency -- and are
re-exported unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from bdsurv import SplittingConfig, select_level_times_for_band
from bdsurv import simulate_birth_death_process_from_contour_process as _contour
from bdsurv import simulate_subcritical_survival_splitting as _splitting
from bdsurv.splitting import SplittingSimulationResult
from bdsurv.tree import SimulatedTree

from bdpinf.process import BirthDeathProcess

__all__ = [
    "ConditionedTrajectory",
    "SplittingConfig",
    "select_level_times_for_band",
    "simulate_birth_death_process_from_contour_process",
    "simulate_subcritical_survival_splitting",
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


def _tree_to_birth_death_process(tree: SimulatedTree) -> BirthDeathProcess:
    return BirthDeathProcess.from_records(
        records=tree.records,
        epoch_end_times=tree.terminal_time,
        birth_rates=tree.birth_rate,
        death_rates=tree.death_rate,
    )


def simulate_birth_death_process_from_contour_process(
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    random_seed: int | None = None,
) -> BirthDeathProcess:
    """Simulate a birth-death process using contour-process reconstruction.

    Thin wrapper around ``bdsurv.simulate_birth_death_process_from_contour_process``
    that converts its ``SimulatedTree`` result into a canonical
    ``BirthDeathProcess``.
    """
    tree = _contour(
        birth_rate=birth_rate,
        death_rate=death_rate,
        terminal_time=terminal_time,
        random_seed=random_seed,
    )
    return _tree_to_birth_death_process(tree)


def simulate_subcritical_survival_splitting(
    config: SplittingConfig,
) -> SplittingSimulationResult:
    """Run fixed-factor enrichment (RESTART) splitting to estimate subcritical survival.

    Thin wrapper around ``bdsurv.simulate_subcritical_survival_splitting`` that
    converts each conditioned trajectory's ``SimulatedTree`` into a canonical
    ``BirthDeathProcess``. See ``bdsurv.splitting`` for the full algorithm
    description.
    """
    result = _splitting(config)
    trajectories = [
        ConditionedTrajectory(
            particle_id=trajectory.particle_id,
            times=trajectory.times,
            populations=trajectory.populations,
            process=_tree_to_birth_death_process(trajectory.process),
        )
        for trajectory in result.conditioned_trajectories
    ]
    return dataclasses.replace(result, conditioned_trajectories=trajectories)
