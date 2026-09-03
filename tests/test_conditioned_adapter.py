"""Test the bdpinf.simulation.conditioned adapters over the bdsurv simulators."""

import numpy as np
import pytest
from bdsurv._extinction import pr_extinction

from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.process import BirthDeathProcess
from bdpinf.simulation import (
    SplittingConfig,
    simulate_birth_death_process_from_contour_process,
    simulate_subcritical_survival_splitting,
)


class TestContourAdapter:
    """Test the contour-process adapter returns a canonical BirthDeathProcess."""

    def test_returns_birth_death_process_with_one_spine(self) -> None:
        process = simulate_birth_death_process_from_contour_process(
            birth_rate=0.5,
            death_rate=0.2,
            terminal_time=5.0,
            random_seed=123,
        )

        assert isinstance(process, BirthDeathProcess)
        records = process.records
        assert records is not None
        spine_rows = records[records["is_spine"]]
        assert len(spine_rows) == 1
        assert bool(spine_rows.iloc[0]["survived"])

    def test_deterministic_for_fixed_seed(self) -> None:
        process_1 = simulate_birth_death_process_from_contour_process(
            birth_rate=0.5, death_rate=0.2, terminal_time=5.0, random_seed=123
        )
        process_2 = simulate_birth_death_process_from_contour_process(
            birth_rate=0.5, death_rate=0.2, terminal_time=5.0, random_seed=123
        )
        assert process_1 == process_2


class TestSplittingAdapter:
    """Test the splitting adapter wraps conditioned trajectories as BirthDeathProcess."""

    def test_conditioned_trajectories_are_birth_death_processes(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=40.0,
            n_particles=250,
            n_levels=8,
            random_seed=11,
        )

        result = simulate_subcritical_survival_splitting(config)

        assert len(result.conditioned_trajectories) == result.final_survivor_count
        for trajectory in result.conditioned_trajectories:
            assert isinstance(trajectory.process, BirthDeathProcess)
            records = trajectory.process.records
            assert records is not None
            spine_rows = records[records["is_spine"]]
            assert len(spine_rows) == 1
            assert bool(spine_rows.iloc[0]["survived"])


class TestExtinctionCrossCheck:
    """Pin bdsurv's closed-form extinction probability against bdpinf's JAX one.

    bdsurv reimplements the single-epoch special case of
    bdpinf.models.math_extinction.calculate_extinction_pr as a pure-numpy
    closed form (see bdsurv._extinction.pr_extinction), so the simulation
    package can avoid a JAX dependency. This test is the guard against the
    two implementations drifting apart.
    """

    @pytest.mark.parametrize(
        ("birth_rate", "death_rate", "terminal_time"),
        [
            (0.1, 1.0, 5.0),
            (0.9, 1.0, 50.0),
            (1.0, 2.0, 20.0),
            (0.5, 0.5, 4.0),  # critical case
            (0.5 - 1e-6, 0.5, 4.0),  # just below critical threshold
            (0.5 + 1e-6, 0.5, 4.0),  # just above critical threshold
            (2.0, 0.5, 3.0),  # supercritical
            (0.05, 1.0, 0.1),  # short horizon
        ],
    )
    def test_matches_bdsurv_closed_form(
        self, birth_rate: float, death_rate: float, terminal_time: float
    ) -> None:
        expected = float(calculate_extinction_pr(birth_rate, death_rate, terminal_time))
        actual = pr_extinction(birth_rate, death_rate, terminal_time)
        assert actual == pytest.approx(expected, abs=1e-8, rel=1e-6)

    def test_matches_across_random_grid(self) -> None:
        rng = np.random.default_rng(2026)
        birth_rates = rng.uniform(0.01, 5.0, size=50)
        death_rates = rng.uniform(0.01, 5.0, size=50)
        terminal_times = rng.uniform(0.1, 100.0, size=50)

        for b, d, t in zip(birth_rates, death_rates, terminal_times, strict=True):
            expected = float(calculate_extinction_pr(float(b), float(d), float(t)))
            actual = pr_extinction(float(b), float(d), float(t))
            assert actual == pytest.approx(expected, abs=1e-6, rel=1e-5)
