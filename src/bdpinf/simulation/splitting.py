"""Splitting-based rare-event simulation for subcritical birth-death processes.

This module provides continuous-time simulation utilities for sampling
subcritical birth-death branching-process trajectories conditioned on survival
up to a large terminal time.

Implemented methods:
- Multilevel splitting with exact Gillespie trajectory evolution
- Level-wise Fleming-Viot particle approximation for comparison

Both methods expose level diagnostics, conditioned trajectory samples, and
particle ancestry metadata suitable for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from inf_bp_texts.models import BirthDeathProcess
from inf_bp_texts.models.mathematical_functions import calculate_extinction_pr
from inf_bp_texts.visualization.plotting import (
    apply_professional_styling,
    apply_tufte_axes,
    ptolqual,
)

MIN_LEVEL_TIMES = 2


@dataclass(frozen=True)
class SplittingConfig:
    """Configuration for multilevel splitting in a subcritical regime.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate used in the birth-death dynamics.
    death_rate : float
        Per-individual death rate used in the birth-death dynamics.
    terminal_time : float
        Target time horizon T for conditioning on survival.
    n_particles : int
        Number of particles in the splitting particle ensemble.
    level_times : list[float] | None, optional
        Explicit level boundary times (including 0 and `terminal_time`). If
        ``None``, levels are constructed from ``n_levels``.
    n_levels : int, optional
        Number of level intervals to create when ``level_times`` is not
        provided. Defaults to 10.
    random_seed : int | None, optional
        Seed for random number generation to allow reproducible runs.
    n_conditioned_trajectories : int, optional
        Number of conditioned trajectories to return for downstream analysis.

    """

    birth_rate: float
    death_rate: float
    terminal_time: float
    n_particles: int
    level_times: list[float] | None = None
    n_levels: int = 10
    random_seed: int | None = None
    n_conditioned_trajectories: int | None = None


@dataclass(frozen=True)
class FlemingViotConfig:
    """Configuration for level-wise Fleming-Viot approximation.

    Parameters
    ----------
    birth_rate : float
        Per-individual birth rate used in the birth-death dynamics.
    death_rate : float
        Per-individual death rate used in the birth-death dynamics.
    terminal_time : float
        Target time horizon T for conditioning on survival.
    n_particles : int
        Number of particles in the Fleming–Viot ensemble.
    level_times : list[float] | None, optional
        Explicit level boundary times (including 0 and `terminal_time`). If
        ``None``, levels are constructed from ``n_levels``.
    n_levels : int, optional
        Number of level intervals to create when ``level_times`` is not
        provided. Defaults to 10.
    random_seed : int | None, optional
        Seed for random number generation to allow reproducible runs.
    n_conditioned_trajectories : int, optional
        Number of conditioned trajectories to return for downstream analysis.

    """

    birth_rate: float
    death_rate: float
    terminal_time: float
    n_particles: int
    level_times: list[float] | None = None
    n_levels: int = 10
    random_seed: int | None = None
    n_conditioned_trajectories: int = 12


@dataclass(frozen=True)
class LevelDiagnostic:
    """Per-level diagnostics for splitting/Fleming-Viot evolution.

    Parameters
    ----------
    level_index : int
        Index of the level interval (1-based with 0 reserved for t0).
    time : float
        Level boundary time corresponding to this diagnostic.
    survivor_count : int
        Number of particles surviving at the level boundary.
    conditional_survival_estimate : float
        Empirical conditional survival probability for this level interval.
    cumulative_survival_estimate : float
        Cumulative product of conditional survival estimates up to this
        level, estimating overall survival to the current time.
    mean_population : float
        Ensemble mean population size at the level boundary.

    """

    level_index: int
    time: float
    survivor_count: int
    conditional_survival_estimate: float
    cumulative_survival_estimate: float
    mean_population: float


@dataclass(frozen=True)
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
        Manuscript-level birth-death process reconstructed from particle
        ancestry records.

    """

    particle_id: int
    times: np.ndarray
    populations: np.ndarray
    process: BirthDeathProcess


@dataclass(frozen=True)
class SplittingSimulationResult:
    """Output of multilevel splitting simulation.

    Parameters
    ----------
    config : SplittingConfig
        The configuration used to run the simulation.
    level_times : np.ndarray
        Array of level boundary times including time 0 and terminal time.
    survival_probability_estimate : float
        Estimated probability of survival to the terminal time (product of
        conditional estimates across levels).
    level_diagnostics : list[LevelDiagnostic]
        Per-level diagnostic objects describing survival and population
        statistics.
    level_survival_probabilities : np.ndarray
        Empirical conditional survival probabilities for each level
        interval (length = len(level_times) - 1).
    level_mean_populations : np.ndarray
        Ensemble mean population at each level boundary.
    final_survivor_count : int
        Number of surviving manuscripts/lineages at the terminal time.
    conditioned_trajectories : list[ConditionedTrajectory]
        Sampled conditioned trajectories selected from surviving particles.
    particle_genealogy : pd.DataFrame
        Tabular parent–child lineage records produced during resampling.

    """

    config: SplittingConfig
    level_times: np.ndarray
    survival_probability_estimate: float
    level_diagnostics: list[LevelDiagnostic]
    level_survival_probabilities: np.ndarray
    level_mean_populations: np.ndarray
    final_survivor_count: int
    conditioned_trajectories: list[ConditionedTrajectory]
    particle_genealogy: pd.DataFrame


@dataclass(frozen=True)
class FlemingViotSimulationResult:
    """Output of level-wise Fleming-Viot simulation.

    Parameters
    ----------
    config : FlemingViotConfig
        The configuration used to run the Fleming–Viot approximation.
    level_times : np.ndarray
        Array of level boundary times including time 0 and terminal time.
    survival_proxy_estimate : float
        Proxy estimate of survival obtained from the Fleming–Viot procedure.
    level_diagnostics : list[LevelDiagnostic]
        Per-level diagnostic objects describing survival and population
        statistics.
    level_survival_probabilities : np.ndarray
        Empirical conditional survival probabilities for each level
        interval (length = len(level_times) - 1).
    level_mean_populations : np.ndarray
        Ensemble mean population at each level boundary.
    final_survivor_count : int
        Number of surviving manuscripts/lineages at the terminal time.
    conditioned_trajectories : list[ConditionedTrajectory]
        Sampled conditioned trajectories selected from surviving particles.
    particle_genealogy : pd.DataFrame
        Tabular parent–child lineage records produced during resampling.

    """

    config: FlemingViotConfig
    level_times: np.ndarray
    survival_proxy_estimate: float
    level_diagnostics: list[LevelDiagnostic]
    level_survival_probabilities: np.ndarray
    level_mean_populations: np.ndarray
    final_survivor_count: int
    conditioned_trajectories: list[ConditionedTrajectory]
    particle_genealogy: pd.DataFrame


@dataclass
class _ParticleState:
    """Mutable per-particle state for exact event simulation.

    Parameters
    ----------
    particle_id : int
        Unique identifier assigned to the particle instance.
    current_time : float
        Current simulated time for this particle.
    next_manuscript_id : int
        Next available manuscript identifier for descendants.
    alive_ids : list[int]
        List of manuscript ids currently alive within the particle.
    manuscript_records : dict[int, dict[str, Any]]
        Mapping from manuscript id to a record dict containing creation and
        death metadata.
    trace_times : list[float]
        Times at which the particle trace population was recorded.
    trace_populations : list[int]
        Population sizes corresponding to ``trace_times``.
    rng : np.random.Generator
        Random number generator instance used for stochastic events.

    """

    particle_id: int
    current_time: float
    next_manuscript_id: int
    alive_ids: list[int]
    manuscript_records: dict[int, dict[str, Any]]
    trace_times: list[float]
    trace_populations: list[int]
    rng: np.random.Generator


def _validate_common_inputs(
    *,
    birth_rate: float,
    death_rate: float,
    terminal_time: float,
    n_particles: int,
) -> None:
    if birth_rate <= 0.0 or death_rate <= 0.0:
        msg = "birth_rate and death_rate must be positive"
        raise ValueError(msg)
    if birth_rate >= death_rate:
        msg = "Subcritical regime required: birth_rate must be < death_rate"
        raise ValueError(msg)
    if terminal_time <= 0.0:
        msg = "terminal_time must be positive"
        raise ValueError(msg)
    if n_particles <= 0:
        msg = "n_particles must be positive"
        raise ValueError(msg)


def _build_level_times(
    *,
    terminal_time: float,
    level_times: list[float] | None,
    n_levels: int,
) -> np.ndarray:
    if level_times is None:
        if n_levels <= 0:
            msg = "n_levels must be positive when level_times is not provided"
            raise ValueError(msg)
        levels = np.linspace(0.0, terminal_time, num=n_levels + 1, dtype=np.float64)
    else:
        levels = np.asarray(level_times, dtype=np.float64)

    if levels.ndim != 1:
        msg = "level_times must be one-dimensional"
        raise ValueError(msg)
    if levels.size < MIN_LEVEL_TIMES:
        msg = "At least two level times are required"
        raise ValueError(msg)
    if not np.isclose(levels[0], 0.0):
        msg = "First level time must be 0"
        raise ValueError(msg)
    if np.any(np.diff(levels) <= 0.0):
        msg = "level_times must be strictly increasing"
        raise ValueError(msg)
    if not np.isclose(levels[-1], terminal_time):
        msg = "Last level time must equal terminal_time"
        raise ValueError(msg)
    return levels


def _make_child_rng(
    *,
    seed_rng: np.random.Generator,
) -> np.random.Generator:
    child_seed = int(seed_rng.integers(0, 2**31 - 1))
    return np.random.default_rng(child_seed)


def _calculate_extinction_probability(
    *,
    birth_rate: float,
    death_rate: float,
    interval_time: float,
) -> float:
    probability = float(calculate_extinction_pr(birth_rate, death_rate, interval_time))
    return float(np.clip(probability, 0.0, 1.0))


def _adaptive_particle_count(
    *,
    birth_rate: float,
    death_rate: float,
    interval_time: float,
) -> int:
    extinction_probability = _calculate_extinction_probability(
        birth_rate=birth_rate,
        death_rate=death_rate,
        interval_time=interval_time,
    )
    if extinction_probability <= 0.0:
        return 1
    return int(np.ceil(1.0 / extinction_probability))


def _initialize_particle(
    *,
    particle_id: int,
    seed_rng: np.random.Generator,
) -> _ParticleState:
    root_record = {
        "manuscript_id": 0,
        "parent_id": None,
        "creation_time": 0.0,
        "death_time": None,
        "is_spine": False,
    }
    return _ParticleState(
        particle_id=particle_id,
        current_time=0.0,
        next_manuscript_id=1,
        alive_ids=[0],
        manuscript_records={0: root_record},
        trace_times=[0.0],
        trace_populations=[1],
        rng=_make_child_rng(seed_rng=seed_rng),
    )


def _append_trace_point(
    *,
    state: _ParticleState,
    time_point: float,
) -> None:
    if time_point < state.trace_times[-1]:
        msg = "time_point must be monotone in particle trace"
        raise RuntimeError(msg)
    if np.isclose(time_point, state.trace_times[-1]):
        state.trace_populations[-1] = len(state.alive_ids)
        return
    state.trace_times.append(time_point)
    state.trace_populations.append(len(state.alive_ids))


def _simulate_to_level(
    *,
    state: _ParticleState,
    birth_rate: float,
    death_rate: float,
    target_time: float,
) -> None:
    if target_time < state.current_time:
        msg = "target_time cannot be less than current_time"
        raise ValueError(msg)

    birth_probability = birth_rate / (birth_rate + death_rate)

    while state.current_time < target_time:
        if len(state.alive_ids) == 0:
            state.current_time = target_time
            _append_trace_point(state=state, time_point=target_time)
            return

        total_rate = len(state.alive_ids) * (birth_rate + death_rate)
        dt = float(state.rng.exponential(1.0 / total_rate))
        event_time = state.current_time + dt

        if event_time >= target_time:
            state.current_time = target_time
            _append_trace_point(state=state, time_point=target_time)
            return

        state.current_time = event_time
        if state.rng.uniform() < birth_probability:
            parent_idx = int(state.rng.integers(0, len(state.alive_ids)))
            parent_id = state.alive_ids[parent_idx]
            child_id = state.next_manuscript_id
            state.next_manuscript_id += 1
            state.alive_ids.append(child_id)
            state.manuscript_records[child_id] = {
                "manuscript_id": child_id,
                "parent_id": parent_id,
                "creation_time": event_time,
                "death_time": None,
                "is_spine": False,
            }
        else:
            death_idx = int(state.rng.integers(0, len(state.alive_ids)))
            dead_id = state.alive_ids.pop(death_idx)
            state.manuscript_records[dead_id]["death_time"] = event_time

        _append_trace_point(state=state, time_point=event_time)


def _clone_particle(
    *,
    donor: _ParticleState,
    particle_id: int,
    seed_rng: np.random.Generator,
) -> _ParticleState:
    return _ParticleState(
        particle_id=particle_id,
        current_time=float(donor.current_time),
        next_manuscript_id=int(donor.next_manuscript_id),
        alive_ids=donor.alive_ids.copy(),
        manuscript_records={
            key: value.copy() for key, value in donor.manuscript_records.items()
        },
        trace_times=donor.trace_times.copy(),
        trace_populations=donor.trace_populations.copy(),
        rng=_make_child_rng(seed_rng=seed_rng),
    )


def _particle_to_birth_death_process(
    *,
    state: _ParticleState,
    terminal_time: float,
    birth_rate: float,
    death_rate: float,
) -> BirthDeathProcess:
    records_df = pd.DataFrame(state.manuscript_records.values())
    if records_df.empty:
        records_df = pd.DataFrame(
            columns=[
                "manuscript_id",
                "parent_id",
                "creation_time",
                "death_time",
                "is_spine",
            ],
        )

    records_df["is_spine"] = False
    if state.alive_ids:
        spine_id = state.alive_ids[0]
        records_df.loc[records_df["manuscript_id"] == spine_id, "is_spine"] = True

    records_df["survived"] = records_df["death_time"].isna()
    records_df["age"] = np.where(
        records_df["survived"],
        terminal_time - records_df["creation_time"].to_numpy(dtype=np.float64),
        np.nan,
    )
    records_df["lifespan"] = np.where(
        ~records_df["survived"],
        records_df["death_time"].to_numpy(dtype=np.float64)
        - records_df["creation_time"].to_numpy(dtype=np.float64),
        np.nan,
    )

    return BirthDeathProcess.from_manuscript_records(
        manuscripts=records_df,
        epoch_end_times=float(terminal_time),
        birth_rates=float(birth_rate),
        death_rates=float(death_rate),
    )


def _build_conditioned_trajectories(
    *,
    particles: list[_ParticleState],
    config: Any,
    seed_rng: np.random.Generator,
) -> list[ConditionedTrajectory]:
    survivors = [particle for particle in particles if len(particle.alive_ids) > 0]
    if not survivors:
        return []

    if config.n_conditioned_trajectories is None:
        n_take = len(survivors)
    else:
        if config.n_conditioned_trajectories <= 0:
            return []
        n_take = min(config.n_conditioned_trajectories, len(survivors))

    chosen_indices = seed_rng.choice(len(survivors), size=n_take, replace=False)
    trajectories: list[ConditionedTrajectory] = []
    for raw_idx in np.atleast_1d(chosen_indices):
        particle = survivors[int(raw_idx)]
        process = _particle_to_birth_death_process(
            state=particle,
            terminal_time=float(config.terminal_time),
            birth_rate=float(config.birth_rate),
            death_rate=float(config.death_rate),
        )
        trajectories.append(
            ConditionedTrajectory(
                particle_id=particle.particle_id,
                times=np.asarray(particle.trace_times, dtype=np.float64),
                populations=np.asarray(particle.trace_populations, dtype=np.int64),
                process=process,
            ),
        )
    return trajectories


def simulate_subcritical_survival_splitting(
    config: SplittingConfig,
) -> SplittingSimulationResult:
    """Run multilevel splitting for subcritical survival to terminal_time.

    The estimator is
        P(tau > T) ~= product_k p_hat_k,
    where p_hat_k is the empirical conditional survival fraction at level k.
    """
    _validate_common_inputs(
        birth_rate=float(config.birth_rate),
        death_rate=float(config.death_rate),
        terminal_time=float(config.terminal_time),
        n_particles=int(config.n_particles),
    )

    levels = _build_level_times(
        terminal_time=float(config.terminal_time),
        level_times=config.level_times,
        n_levels=int(config.n_levels),
    )

    seed_rng = np.random.default_rng(config.random_seed)
    particles = [
        _initialize_particle(particle_id=i, seed_rng=seed_rng)
        for i in range(config.n_particles)
    ]

    diagnostics: list[LevelDiagnostic] = []
    level_probs: list[float] = []
    mean_populations: list[float] = [1.0]
    lineage_rows: list[dict[str, Any]] = []
    particle_id_counter = config.n_particles
    cumulative = 1.0
    previous_time = 0.0

    for level_index in range(1, len(levels)):
        target_time = float(levels[level_index])
        interval_time = target_time - previous_time
        for particle in particles:
            _simulate_to_level(
                state=particle,
                birth_rate=float(config.birth_rate),
                death_rate=float(config.death_rate),
                target_time=target_time,
            )

        populations_at_level = np.asarray(
            [len(particle.alive_ids) for particle in particles],
            dtype=np.int64,
        )
        survivors = np.flatnonzero(populations_at_level > 0)
        survivor_count = int(survivors.size)
        conditional_prob = survivor_count / float(config.n_particles)
        cumulative *= conditional_prob

        diagnostics.append(
            LevelDiagnostic(
                level_index=level_index,
                time=target_time,
                survivor_count=survivor_count,
                conditional_survival_estimate=conditional_prob,
                cumulative_survival_estimate=cumulative,
                mean_population=float(np.mean(populations_at_level)),
            ),
        )
        level_probs.append(conditional_prob)
        mean_populations.append(float(np.mean(populations_at_level)))

        if survivor_count == 0:
            particles = []
            break

        target_particle_count = _adaptive_particle_count(
            birth_rate=float(config.birth_rate),
            death_rate=float(config.death_rate),
            interval_time=float(interval_time),
        )
        selected_slots = seed_rng.choice(
            survivors,
            size=target_particle_count,
            replace=True,
        )
        next_particles: list[_ParticleState] = []
        for child_slot, parent_slot_raw in enumerate(np.atleast_1d(selected_slots)):
            parent_slot = int(parent_slot_raw)
            parent_particle = particles[parent_slot]
            child_particle = _clone_particle(
                donor=parent_particle,
                particle_id=particle_id_counter,
                seed_rng=seed_rng,
            )
            particle_id_counter += 1
            next_particles.append(child_particle)
            lineage_rows.append(
                {
                    "level_index": level_index,
                    "time": target_time,
                    "child_slot": child_slot,
                    "parent_slot": parent_slot,
                    "child_particle_id": child_particle.particle_id,
                    "parent_particle_id": parent_particle.particle_id,
                },
            )
        particles = next_particles
        previous_time = target_time

    conditioned_trajectories = _build_conditioned_trajectories(
        particles=particles,
        config=config,
        seed_rng=seed_rng,
    )

    final_survivor_count = int(sum(len(p.alive_ids) > 0 for p in particles))
    particle_genealogy = pd.DataFrame(lineage_rows)
    if particle_genealogy.empty:
        particle_genealogy = pd.DataFrame(
            columns=[
                "level_index",
                "time",
                "child_slot",
                "parent_slot",
                "child_particle_id",
                "parent_particle_id",
            ],
        )

    return SplittingSimulationResult(
        config=config,
        level_times=levels,
        survival_probability_estimate=float(cumulative),
        level_diagnostics=diagnostics,
        level_survival_probabilities=np.asarray(level_probs, dtype=np.float64),
        level_mean_populations=np.asarray(mean_populations, dtype=np.float64),
        final_survivor_count=final_survivor_count,
        conditioned_trajectories=conditioned_trajectories,
        particle_genealogy=particle_genealogy,
    )


def simulate_subcritical_survival_fleming_viot(
    config: FlemingViotConfig,
) -> FlemingViotSimulationResult:
    """Run a level-wise Fleming-Viot approximation for comparison.

    This approximation evolves all particles with exact continuous-time dynamics
    on each level interval, then replaces extinct particles by cloning randomly
    selected survivors at the same level boundary.
    """
    _validate_common_inputs(
        birth_rate=float(config.birth_rate),
        death_rate=float(config.death_rate),
        terminal_time=float(config.terminal_time),
        n_particles=int(config.n_particles),
    )

    levels = _build_level_times(
        terminal_time=float(config.terminal_time),
        level_times=config.level_times,
        n_levels=int(config.n_levels),
    )

    seed_rng = np.random.default_rng(config.random_seed)
    particles = [
        _initialize_particle(particle_id=i, seed_rng=seed_rng)
        for i in range(config.n_particles)
    ]

    diagnostics: list[LevelDiagnostic] = []
    level_probs: list[float] = []
    mean_populations: list[float] = [1.0]
    lineage_rows: list[dict[str, Any]] = []
    particle_id_counter = config.n_particles
    cumulative_proxy = 1.0

    for level_index in range(1, len(levels)):
        target_time = float(levels[level_index])
        for particle in particles:
            _simulate_to_level(
                state=particle,
                birth_rate=float(config.birth_rate),
                death_rate=float(config.death_rate),
                target_time=target_time,
            )

        populations = np.asarray([len(p.alive_ids) for p in particles], dtype=np.int64)
        survivors = np.flatnonzero(populations > 0)
        dead_slots = np.flatnonzero(populations == 0)
        survivor_count = int(survivors.size)
        conditional = survivor_count / float(config.n_particles)
        cumulative_proxy *= conditional

        diagnostics.append(
            LevelDiagnostic(
                level_index=level_index,
                time=target_time,
                survivor_count=survivor_count,
                conditional_survival_estimate=conditional,
                cumulative_survival_estimate=cumulative_proxy,
                mean_population=float(np.mean(populations)),
            ),
        )
        level_probs.append(conditional)
        mean_populations.append(float(np.mean(populations)))

        if survivor_count == 0:
            particles = []
            break

        for dead_slot_raw in dead_slots:
            dead_slot = int(dead_slot_raw)
            donor_slot = int(seed_rng.choice(survivors))
            donor_particle = particles[donor_slot]
            replacement = _clone_particle(
                donor=donor_particle,
                particle_id=particle_id_counter,
                seed_rng=seed_rng,
            )
            particle_id_counter += 1
            lineage_rows.append(
                {
                    "level_index": level_index,
                    "time": target_time,
                    "child_slot": dead_slot,
                    "parent_slot": donor_slot,
                    "child_particle_id": replacement.particle_id,
                    "parent_particle_id": donor_particle.particle_id,
                },
            )
            particles[dead_slot] = replacement

    conditioned_trajectories = _build_conditioned_trajectories(
        particles=particles,
        config=config,
        seed_rng=seed_rng,
    )

    final_survivor_count = int(sum(len(p.alive_ids) > 0 for p in particles))
    genealogy = pd.DataFrame(lineage_rows)
    if genealogy.empty:
        genealogy = pd.DataFrame(
            columns=[
                "level_index",
                "time",
                "child_slot",
                "parent_slot",
                "child_particle_id",
                "parent_particle_id",
            ],
        )

    return FlemingViotSimulationResult(
        config=config,
        level_times=levels,
        survival_proxy_estimate=float(cumulative_proxy),
        level_diagnostics=diagnostics,
        level_survival_probabilities=np.asarray(level_probs, dtype=np.float64),
        level_mean_populations=np.asarray(mean_populations, dtype=np.float64),
        final_survivor_count=final_survivor_count,
        conditioned_trajectories=conditioned_trajectories,
        particle_genealogy=genealogy,
    )


def plot_conditioned_trajectories(
    trajectories: list[ConditionedTrajectory],
    *,
    max_trajectories: int = 12,
    title: str = "Conditioned Trajectories",
) -> go.Figure:
    """Plot sample conditioned population paths."""
    fig = go.Figure()
    if not trajectories:
        fig.add_annotation(text="No conditioned trajectories available")
        return cast(
            "go.Figure",
            apply_professional_styling(
                fig,
                title,
                "Time",
                "Population size",
            ),
        )

    n_show = min(max_trajectories, len(trajectories))
    for idx in range(n_show):
        traj = trajectories[idx]
        color = ptolqual[idx % len(ptolqual)]
        fig.add_trace(
            go.Scatter(
                x=traj.times,
                y=traj.populations,
                mode="lines",
                line={"color": color, "width": 1.6},
                name=f"Particle {traj.particle_id}",
            ),
        )

    styled = cast(
        "go.Figure",
        apply_professional_styling(fig, title, "Time", "Population size"),
    )
    return cast("go.Figure", apply_tufte_axes(styled))


def plot_empirical_population_mean(
    *,
    level_times: np.ndarray,
    mean_populations: np.ndarray,
    title: str = "Empirical Mean Population Across Levels",
) -> go.Figure:
    """Plot empirical ensemble mean population vs level times."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=level_times,
            y=mean_populations,
            mode="lines+markers",
            line={"color": ptolqual[0], "width": 2.0},
            marker={"size": 6},
            name="Empirical mean",
        ),
    )
    styled = cast(
        "go.Figure",
        apply_professional_styling(fig, title, "Level time", "Mean population"),
    )
    return cast("go.Figure", apply_tufte_axes(styled))


def plot_level_survival_probabilities(
    *,
    level_times: np.ndarray,
    level_survival_probabilities: np.ndarray,
    title: str = "Estimated Survival Probabilities Across Levels",
) -> go.Figure:
    """Plot level-wise conditional survival probabilities."""
    if len(level_times) != len(level_survival_probabilities) + 1:
        msg = (
            "level_times must include t0; probabilities are defined for level intervals"
        )
        raise ValueError(msg)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=level_times[1:],
            y=level_survival_probabilities,
            mode="lines+markers",
            line={"color": ptolqual[1], "width": 2.0},
            marker={"size": 6},
            name="Level conditional survival",
        ),
    )
    styled = cast(
        "go.Figure",
        apply_professional_styling(fig, title, "Level time", "Probability"),
    )
    styled = cast("go.Figure", apply_tufte_axes(styled))
    styled.update_yaxes(range=[0.0, 1.0])
    return styled
