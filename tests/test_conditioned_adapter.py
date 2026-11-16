"""Test the bdpinf.simulation.conditioned adapters over the bdsurv simulators."""

from bdsurv import PiecewiseConstant, PiecewiseLinear, SplittingConfig

from bdpinf.process import BirthDeathProcess
from bdpinf.simulation import (
    simulate_contour_given_survival,
    simulate_survival_splitting,
)


class TestContourAdapter:
    """Test the contour-process adapter returns a canonical BirthDeathProcess."""

    def test_returns_birth_death_process_with_one_spine(self) -> None:
        process = simulate_contour_given_survival(
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
        process_1 = simulate_contour_given_survival(
            birth_rate=0.5, death_rate=0.2, terminal_time=5.0, random_seed=123
        )
        process_2 = simulate_contour_given_survival(
            birth_rate=0.5, death_rate=0.2, terminal_time=5.0, random_seed=123
        )
        assert process_1 == process_2

    def test_rate_objects_give_a_process_with_stored_epochs(self) -> None:
        constant = simulate_contour_given_survival(
            PiecewiseConstant([0.5, 0.4], [2.0, 5.0]),
            PiecewiseConstant([0.2, 0.3], [2.0, 5.0]),
            random_seed=123,
        )
        assert list(constant.epoch_end_times) == [2.0, 5.0]
        assert list(constant.birth_rates) == [0.5, 0.4]

        linear = simulate_contour_given_survival(
            PiecewiseLinear([0.5, 0.4], [0.4, 0.3], [2.0, 5.0]),
            PiecewiseConstant([0.2, 0.3], [2.0, 5.0]),
            random_seed=123,
        )
        assert list(linear.epoch_end_times) == [2.0, 5.0]
        assert linear.birth_rates is None
        assert linear.death_rates is None


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

        result = simulate_survival_splitting(config)

        assert len(result.conditioned_trajectories) == result.final_survivor_count
        for trajectory in result.conditioned_trajectories:
            assert isinstance(trajectory.process, BirthDeathProcess)
            records = trajectory.process.records
            assert records is not None
            spine_rows = records[records["is_spine"]]
            assert len(spine_rows) == 1
            assert bool(spine_rows.iloc[0]["survived"])
