"""Test suite for BirthDeathProcess class initialization and basic functionality."""

# ruff: noqa: S101

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest
import torch
from numpy.testing import assert_array_equal

from inf_bp_texts.models import BirthDeathProcess


class TestBirthDeathProcessInit:
    """Test BirthDeathProcess initialization and basic methods."""

    def test_default_initialization(self) -> None:
        """Test default parameter initialization."""
        creation_times = []
        epoch_end_times = 1000.0
        process = BirthDeathProcess(
            creation_times=creation_times,
            epoch_end_times=epoch_end_times,
        )

        assert process.birth_rates is None  # Optional, not provided
        assert process.death_rates is None  # Optional, not provided
        assert process.creation_times == []
        assert process.epoch_end_times[-1].item() == 1000.0
        assert process.initial_population == 1
        assert (
            not process.fitted
        )  # Not fitted since birth_rates and death_rates not provided
        assert process.fit_results is None
        assert len(process.manuscript_ages) == 0

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        creation_times = [100.0, 200.0, 300.0]
        epoch_end_times = 500.0

        process = BirthDeathProcess(
            creation_times=creation_times,
            epoch_end_times=epoch_end_times,
        )

        assert process.birth_rates is None  # Optional, not provided
        assert process.death_rates is None  # Optional, not provided
        assert process.creation_times == creation_times
        assert process.start_time == 0.0
        assert process.initial_population == 1
        assert process.epoch_end_times[-1].item() == epoch_end_times

        # Ages should be calculated on-demand
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.manuscript_ages, expected_ages)

    def test_adjusted_epoch_end_times_property(self) -> None:
        """Test adjusted epoch end times are shifted by start_time."""
        scalar_process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=500.0,
            start_time=100.0,
        )
        assert_array_equal(scalar_process.adjusted_epoch_end_times, [400.0])

        vector_process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=[400.0, 800.0, 1200.0],
            start_time=100.0,
        )
        assert_array_equal(
            vector_process.adjusted_epoch_end_times,
            [300.0, 700.0, 1100.0],
        )

    def test_add_manuscript(self) -> None:
        """Test adding manuscripts."""
        process = BirthDeathProcess(creation_times=[], epoch_end_times=1000.0)

        # Add manuscripts
        creation_times = [100, 300, 700]
        for ct in creation_times:
            process.add_manuscript(ct)

        assert process.creation_times == creation_times
        expected_ages = np.array([900, 700, 300], dtype=np.float64)
        assert_array_equal(process.manuscript_ages, expected_ages)

    def test_set_epoch_end_times(self) -> None:
        """Test setting epoch end times."""
        process = BirthDeathProcess(creation_times=[], epoch_end_times=1000.0)
        process.creation_times = [100, 200, 300]

        # Initially epoch end time is 1000, so ages should be calculated
        expected_initial_ages = np.array([900, 800, 700], dtype=np.float64)
        assert_array_equal(process.manuscript_ages, expected_initial_ages)

        # Update epoch end time using setter
        process.set_epoch_end_times(500)
        assert process.epoch_end_times[-1].item() == 500

        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.manuscript_ages, expected_ages)

    def test_ages_calculated_on_demand(self) -> None:
        """Test that ages are calculated correctly on-demand when inputs change."""
        process = BirthDeathProcess(creation_times=[], epoch_end_times=1000.0)

        # Add manuscripts and verify ages are calculated correctly
        process.add_manuscript(100)
        assert_array_equal(process.manuscript_ages, [900])

        process.add_manuscript(300)
        assert_array_equal(process.manuscript_ages, [900, 700])

        # Change epoch end time and verify ages are recalculated correctly
        process.set_epoch_end_times(1200)
        assert_array_equal(process.manuscript_ages, [1100, 900])

    def test_manuscript_ages_values(self) -> None:
        """Test manuscript_ages property values."""
        process = BirthDeathProcess(creation_times=[100, 200, 300], epoch_end_times=500)

        ages = process.manuscript_ages
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(ages, expected_ages)

        # Verify multiple calls return consistent results
        ages2 = process.manuscript_ages
        assert_array_equal(ages, ages2)

    def test_manuscript_ages_property(self) -> None:
        """Test manuscript_ages property returns expected values."""
        process = BirthDeathProcess(creation_times=[100, 200, 300], epoch_end_times=500)

        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.manuscript_ages, expected_ages)
        assert_array_equal(process.manuscript_ages, expected_ages)

    def test_get_process_summary(self) -> None:
        """Test process summary generation."""
        process = BirthDeathProcess(
            creation_times=[100, 200, 300, 500],
            epoch_end_times=1000,
            initial_population=2,
            birth_rates=0.02,
            death_rates=0.005,
        )

        assert process.birth_rates.tolist() == [0.02]
        assert process.death_rates.tolist() == [0.005]
        assert process.n_manuscripts == 4
        assert process.epoch_end_times.tolist() == [1000.0]
        assert process.adjusted_epoch_end_times.tolist() == [1000.0]
        assert process.initial_population == 2
        assert min(process.manuscript_ages) == 500.0  # 1000 - 500
        assert max(process.manuscript_ages) == 900.0  # 1000 - 100
        assert (
            torch.mean(process.manuscript_ages).item() == 725.0
        )  # Mean of [900, 800, 700, 500] = 2900/4 = 725

    def test_initial_population_validation(self) -> None:
        """Test initial_population requires a positive integer."""
        with pytest.raises(TypeError, match="initial_population must be an integer"):
            BirthDeathProcess(
                creation_times=[],
                epoch_end_times=1000.0,
                initial_population=1.0,
            )

        with pytest.raises(
            ValueError,
            match="initial_population must be a positive integer",
        ):
            BirthDeathProcess(
                creation_times=[],
                epoch_end_times=1000.0,
                initial_population=0,
            )

    def test_empty_process_summary(self) -> None:
        """Test summary with no manuscripts."""
        process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=1000.0,
            birth_rates=0.01,
            death_rates=0.002,
        )

        assert process.birth_rates.tolist() == [0.01]
        assert process.death_rates.tolist() == [0.002]
        assert process.n_manuscripts == 0
        assert process.epoch_end_times.tolist() == [1000.0]

    def test_type_consistency(self) -> None:
        """Test that attributes maintain correct types."""
        process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=1000.0,
            birth_rates=0.02,
            death_rates=0.005,
        )

        assert isinstance(process.birth_rates, torch.Tensor)
        assert isinstance(process.death_rates, torch.Tensor)
        assert process.birth_rates.dtype == torch.float64
        assert process.death_rates.dtype == torch.float64
        assert process.birth_rates is not None
        assert process.death_rates is not None
        assert process.n_epochs == 1
        assert isinstance(process.creation_times, list)

        # Test that manuscript_ages returns correct type
        ages = process.manuscript_ages
        assert isinstance(ages, torch.Tensor)
        assert ages.dtype == torch.float64

    def test_manuscript_validation_on_epoch_end_time_update(self) -> None:
        """Test that invalid manuscripts trigger validation on epoch updates."""
        # Start with manuscripts that span before and after a future end time
        creation_times = [100.0, 200.0, 300.0, 600.0, 800.0]
        initial_epoch_end_times = 1000.0

        process = BirthDeathProcess(
            creation_times=creation_times,
            epoch_end_times=initial_epoch_end_times,
        )

        # Initially all manuscripts should be present
        assert len(process.creation_times) == 5
        assert process.creation_times == [100.0, 200.0, 300.0, 600.0, 800.0]

        # Update epoch end time to 500.0 should fail validation
        with pytest.raises(
            ValueError,
            match="creation_times must be less than or equal to latest epoch_end_time",
        ):
            process.set_epoch_end_times(500.0)

        # State should remain unchanged after failed validation
        assert len(process.creation_times) == 5
        assert process.creation_times == [100.0, 200.0, 300.0, 600.0, 800.0]
        assert process.epoch_end_times[-1].item() == 1000.0

    def test_creation_times_respect_start_time(self) -> None:
        """Test creation times are validated against start_time at initialization."""
        with pytest.raises(
            ValueError,
            match="creation_times must be greater than or equal to start_time",
        ):
            BirthDeathProcess(
                creation_times=[100.0, 200.0, 300.0],
                epoch_end_times=500.0,
                start_time=150.0,
            )

    def test_add_manuscript_respects_start_time(self) -> None:
        """Test add_manuscript validates manuscript time against start_time."""
        process = BirthDeathProcess(
            creation_times=[200.0, 250.0],
            epoch_end_times=500.0,
            start_time=150.0,
        )

        with pytest.raises(
            ValueError,
            match="creation_time must be greater than or equal to start_time",
        ):
            process.add_manuscript(100.0)

    def test_vector_epoch_times_and_rates(self) -> None:
        """Test vector epoch end times and vector rates with matching lengths."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0, 700.0],
            epoch_end_times=[400.0, 800.0, 1200.0],
            birth_rates=[0.02, 0.03, 0.04],
            death_rates=[0.005, 0.007, 0.009],
        )

        assert process.n_epochs == 3
        assert process.epoch_end_times[-1].item() == 1200.0
        assert_array_equal(process.epoch_end_times, np.array([400.0, 800.0, 1200.0]))
        assert_array_equal(process.birth_rates, np.array([0.02, 0.03, 0.04]))
        assert_array_equal(process.death_rates, np.array([0.005, 0.007, 0.009]))

        with pytest.raises(ValueError, match="birth_rates length"):
            process.set_epoch_end_times([400.0, 800.0])


class TestBirthDeathProcessGenealogy:
    """Test genealogy ordering and plotting helpers."""

    @staticmethod
    def _sample_genealogy_records() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "manuscript_id": 0,
                    "parent_id": None,
                    "creation_time": 0.0,
                    "death_time": 0.5,
                    "is_spine": False,
                    "survived": False,
                    "age": np.nan,
                    "lifespan": 0.5,
                },
                {
                    "manuscript_id": 1,
                    "parent_id": 0,
                    "creation_time": 1.0,
                    "death_time": 6.0,
                    "is_spine": False,
                    "survived": False,
                    "age": np.nan,
                    "lifespan": 5.0,
                },
                {
                    "manuscript_id": 2,
                    "parent_id": 0,
                    "creation_time": 2.0,
                    "death_time": np.nan,
                    "is_spine": False,
                    "survived": True,
                    "age": 8.0,
                    "lifespan": np.nan,
                },
                {
                    "manuscript_id": 3,
                    "parent_id": 1,
                    "creation_time": 3.0,
                    "death_time": np.nan,
                    "is_spine": False,
                    "survived": True,
                    "age": 7.0,
                    "lifespan": np.nan,
                },
                {
                    "manuscript_id": 4,
                    "parent_id": 1,
                    "creation_time": 4.0,
                    "death_time": 9.0,
                    "is_spine": False,
                    "survived": False,
                    "age": np.nan,
                    "lifespan": 5.0,
                },
            ],
        )

    def test_genealogy_dfs_order_marks_first_survivor_as_spine(self) -> None:
        process = BirthDeathProcess.from_manuscript_records(
            self._sample_genealogy_records(),
            epoch_end_times=10.0,
            start_time=0.0,
        )

        order = process.genealogy_dfs_order()

        assert order["manuscript_id"].tolist() == [0, 2, 1, 4, 3]
        assert order["dfs_index"].tolist() == [0, 1, 2, 3, 4]
        assert order.loc[order["is_spine"], "manuscript_id"].tolist() == [2]
        assert order.loc[order["manuscript_id"] == 2, "plot_death_time"].item() == 10.0
        assert order.loc[order["manuscript_id"] == 0, "plot_death_time"].item() == 0.5

    def test_plot_genealogy_tree_returns_expected_shapes_and_ticks(self) -> None:
        process = BirthDeathProcess.from_manuscript_records(
            self._sample_genealogy_records(),
            epoch_end_times=10.0,
            start_time=0.0,
        )

        fig = process.plot_genealogy_tree()

        assert isinstance(fig, go.Figure)
        assert fig.layout.yaxis.autorange == "reversed"
        assert tuple(fig.layout.yaxis.tickvals) == (0, 1, 2, 3, 4)
        assert tuple(fig.layout.yaxis.ticktext) == ("0", "2", "1", "4", "3")
        assert len(fig.layout.shapes) == 9

        shape_dicts = [shape.to_plotly_json() for shape in fig.layout.shapes]
        assert any(
            shape["line"]["color"] == "#000000" and shape["line"]["width"] == 2.0
            for shape in shape_dicts
        )
        assert any(
            shape["line"]["color"] == "#228833" and shape["line"]["width"] == 4.0
            for shape in shape_dicts
        )
        assert any(shape["line"].get("dash") == "dot" for shape in shape_dicts)

    def test_genealogy_helpers_require_records(self) -> None:
        process = BirthDeathProcess(creation_times=[1.0], epoch_end_times=10.0)

        with pytest.raises(ValueError, match="requires manuscript_records"):
            process.genealogy_dfs_order()

        with pytest.raises(ValueError, match="requires manuscript_records"):
            process.plot_genealogy_tree()


class TestBirthDeathProcessNewick:
    """Test strict Newick export for rooted genealogies."""

    @staticmethod
    def _sample_newick_records() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "manuscript_id": 1,
                    "parent_id": 0,
                    "creation_time": 2.0,
                    "death_time": 6.0,
                    "is_spine": False,
                    "survived": False,
                    "age": np.nan,
                    "lifespan": 4.0,
                },
                {
                    "manuscript_id": 0,
                    "parent_id": None,
                    "creation_time": 0.0,
                    "death_time": np.nan,
                    "is_spine": False,
                    "survived": True,
                    "age": 10.0,
                    "lifespan": np.nan,
                },
            ],
        )

    def test_to_newick_exports_deterministic_tree(self) -> None:
        process = BirthDeathProcess.from_manuscript_records(
            self._sample_newick_records(),
            epoch_end_times=10.0,
            start_time=0.0,
        )

        assert process.to_newick() == "((((1:4.0):0,(0:8.0):0):8.0):0):0.0;"

    def test_to_newick_requires_manuscript_records(self) -> None:
        process = BirthDeathProcess(creation_times=[1.0], epoch_end_times=10.0)

        with pytest.raises(ValueError, match="requires manuscript_records"):
            process.to_newick()
