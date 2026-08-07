"""Tests for spike detection helpers."""

import numpy as np
import pandas as pd

from bdpinf.analysis import spike_detection as sd
from bdpinf.analysis.spike_detection import (
    bootstrap_percentile_ci,
    compute_ci_width_table,
    select_extreme_replicates,
    select_max_ci_times,
)


class TestSpikeDetection:
    """Test spike detection and CI selection utilities."""

    def test_seed_from_parts_is_deterministic(self):
        a = sd.seed_from_parts(1, "a", 3.14)
        b = sd.seed_from_parts(1, "a", 3.14)
        assert isinstance(a, int)
        assert a == b

    def test_replicate_and_bootstrap_seed_different(self):
        rep = sd.replicate_seed(42, "Regime", "process", 0.5, 7)
        bs = sd.bootstrap_seed(rep)
        assert isinstance(rep, int)
        assert isinstance(bs, int)
        assert rep != bs

    def test_bootstrap_percentile_ci_is_deterministic(self) -> None:
        values = np.array([1.0, 2.0, 10.0, 11.0], dtype=np.float64)

        first = bootstrap_percentile_ci(
            values,
            bootstrap_samples=200,
            seed=123,
            statistic="mean",
        )
        second = bootstrap_percentile_ci(
            values,
            bootstrap_samples=200,
            seed=123,
            statistic="mean",
        )

        assert first == second
        assert first["ci_width"] > 0.0

    def test_compute_and_select_ci_targets(self) -> None:
        results_df = pd.DataFrame(
            [
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 1.0,
                    "success": True,
                    "simulation_index": 0,
                    "population_size": 3,
                    "birth_estimate": 1.0,
                    "death_estimate": 2.0,
                    "replicate_seed": 101,
                },
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 1.0,
                    "success": True,
                    "simulation_index": 1,
                    "population_size": 3,
                    "birth_estimate": 4.0,
                    "death_estimate": 5.0,
                    "replicate_seed": 102,
                },
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 2.0,
                    "success": True,
                    "simulation_index": 2,
                    "population_size": 3,
                    "birth_estimate": 2.0,
                    "death_estimate": 8.0,
                    "replicate_seed": 103,
                },
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 2.0,
                    "success": False,
                    "simulation_index": 3,
                    "population_size": 0,
                    "birth_estimate": np.nan,
                    "death_estimate": np.nan,
                    "replicate_seed": 104,
                },
            ],
        )
        regime_configs = [
            {"name": "R1", "birth_rate": 1.0, "death_rate": 2.0},
        ]

        ci_width_df = compute_ci_width_table(
            results_df,
            regime_configs,
            bootstrap_samples=50,
            seed=99,
        )

        assert set(ci_width_df["metric_name"]) == {"mean", "median", "variance"}
        assert set(ci_width_df["parameter_name"]) == {"birth", "death"}

        target_times_df = select_max_ci_times(ci_width_df)
        assert not target_times_df.empty
        assert set(target_times_df["T"]) <= {1.0, 2.0}

    def test_select_extreme_replicates_returns_high_and_low_rows(self) -> None:
        results_df = pd.DataFrame(
            [
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 1.0,
                    "success": True,
                    "simulation_index": 0,
                    "population_size": 3,
                    "birth_estimate": 1.0,
                    "death_estimate": 9.0,
                    "replicate_seed": 101,
                },
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 1.0,
                    "success": True,
                    "simulation_index": 1,
                    "population_size": 3,
                    "birth_estimate": 2.0,
                    "death_estimate": 8.0,
                    "replicate_seed": 102,
                },
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 1.0,
                    "success": True,
                    "simulation_index": 2,
                    "population_size": 3,
                    "birth_estimate": 3.0,
                    "death_estimate": 7.0,
                    "replicate_seed": 103,
                },
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 1.0,
                    "success": True,
                    "simulation_index": 3,
                    "population_size": 3,
                    "birth_estimate": 4.0,
                    "death_estimate": 6.0,
                    "replicate_seed": 104,
                },
            ],
        )
        target_times_df = pd.DataFrame(
            [
                {
                    "regime": "R1",
                    "sampling": "process",
                    "T": 1.0,
                    "parameter_name": "birth",
                    "metric_name": "mean",
                    "ci_width": 3.0,
                    "true_value": 1.0,
                    "n_successful": 4,
                    "n_replicates": 4,
                },
            ],
        )

        selected = select_extreme_replicates(
            results_df,
            target_times_df,
            num_simulations=4,
            include_lowest=True,
        )

        assert {"highest", "lowest"} <= set(selected["selection_side"])
        assert {"birth", "death"} <= set(selected["selection_parameter"])
        assert set(selected["simulation_index"]) == {0, 1, 2, 3}
