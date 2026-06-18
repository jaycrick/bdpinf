"""Test suite for BirthDeathProcess class initialization and basic functionality."""

import altair as alt
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
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
            float(jnp.mean(process.manuscript_ages)) == 725.0
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

        assert isinstance(process.birth_rates, jax.Array)
        assert isinstance(process.death_rates, jax.Array)
        assert process.birth_rates.dtype == jnp.float64
        assert process.death_rates.dtype == jnp.float64
        assert process.birth_rates is not None
        assert process.death_rates is not None
        assert process.n_epochs == 1
        assert isinstance(process.creation_times, list)

        # Test that manuscript_ages returns correct type
        ages = process.manuscript_ages
        assert isinstance(ages, jax.Array)
        assert ages.dtype == jnp.float64

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

    def test_plot_genealogy_tree_builds_layered_rule_chart(self) -> None:
        process = BirthDeathProcess.from_manuscript_records(
            self._sample_genealogy_records(),
            epoch_end_times=10.0,
            start_time=0.0,
        )

        chart = process.plot_genealogy_tree()

        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        lifetimes, connectors = spec["layer"]
        assert lifetimes["mark"]["type"] == "rule"
        assert connectors["mark"]["type"] == "rule"

        # y axis labelled by manuscript_id in DFS order (root 0 on top), the same
        # ordering the previous Plotly ticktext produced.
        assert lifetimes["encoding"]["y"]["scale"]["domain"] == [
            "0",
            "2",
            "1",
            "4",
            "3",
        ]

        # Lifetimes are colour/size-coded by lineage category.
        color_scale = lifetimes["encoding"]["color"]["scale"]
        assert color_scale["domain"] == ["extinct", "survivor", "spine"]
        assert color_scale["range"][0] == "#000000"  # extinct
        assert color_scale["range"][2] == "#228833"  # spine

        # Parent-child connectors are dashed.
        assert connectors["mark"]["strokeDash"] == [2, 2]

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


class TestBirthDeathProcessFactories:
    """Test BirthDeathProcess class-method constructors."""

    def test_from_age_samples_stores_spine_and_non_spine(self) -> None:
        spine = [800.0, 900.0]
        non_spine = [200.0, 400.0, 600.0]
        T = 1000.0
        process = BirthDeathProcess.from_age_samples(
            spine_ages=spine,
            non_spine_ages=non_spine,
            epoch_end_times=T,
        )
        assert process.spine_ages is not None
        assert process.non_spine_ages is not None
        assert list(process.spine_ages) == pytest.approx(spine)
        assert list(process.non_spine_ages) == pytest.approx(non_spine)

    def test_from_age_samples_creation_times_consistent_with_ages(self) -> None:
        spine = [300.0]
        non_spine = [100.0, 700.0]
        T = 1000.0
        process = BirthDeathProcess.from_age_samples(
            spine_ages=spine,
            non_spine_ages=non_spine,
            epoch_end_times=T,
        )
        expected_creation_times = sorted(
            [T - age for age in [*spine, *non_spine]], reverse=False
        )
        assert sorted(process.creation_times) == pytest.approx(expected_creation_times)

    def test_from_age_samples_with_rates(self) -> None:
        process = BirthDeathProcess.from_age_samples(
            spine_ages=[500.0],
            non_spine_ages=[200.0, 400.0],
            epoch_end_times=1000.0,
            birth_rates=0.05,
            death_rates=0.02,
        )
        assert process.birth_rates is not None
        assert process.death_rates is not None
        assert process.birth_rates[0].item() == pytest.approx(0.05)

    def test_from_manuscript_records_creates_correct_epoch_structure(self) -> None:
        """from_manuscript_records filters to survivors and stores records."""
        T = 600.0
        records = pd.DataFrame(
            {
                "manuscript_id": [1, 2, 3, 4],
                "parent_id": [None, 1, 1, 2],
                "creation_time": [0.0, 100.0, 200.0, 300.0],
                "death_time": [T, T, 400.0, T],
                "is_spine": [True, False, False, False],
                "survived": [True, True, False, True],
                "age": [T, T - 100.0, None, T - 300.0],
                "lifespan": [T, T - 100.0, 200.0, T - 300.0],
            }
        )
        process = BirthDeathProcess.from_manuscript_records(
            records,
            epoch_end_times=T,
            birth_rates=0.01,
            death_rates=0.003,
        )
        # Only survivors (3 of 4 manuscripts) contribute creation_times
        assert len(process.creation_times) == 3
        assert sorted(process.creation_times) == pytest.approx([0.0, 100.0, 300.0])
        # manuscript_records stores all records (survived + dead)
        assert process.manuscript_records is not None
        assert len(process.manuscript_records) == 4
        # epoch structure
        assert len(process.epoch_end_times) == 1
        assert float(process.epoch_end_times[0]) == pytest.approx(T)


class TestBirthDeathProcessEquals:
    """Test BirthDeathProcess.equals() identity and difference detection."""

    def test_equals_returns_true_for_identical_construction(self) -> None:
        p1 = BirthDeathProcess(
            creation_times=[100.0, 200.0],
            epoch_end_times=500.0,
            birth_rates=0.05,
            death_rates=0.02,
        )
        p2 = BirthDeathProcess(
            creation_times=[100.0, 200.0],
            epoch_end_times=500.0,
            birth_rates=0.05,
            death_rates=0.02,
        )
        assert bool(p1.equals(p2))

    def test_equals_returns_false_when_birth_rates_differ(self) -> None:
        p1 = BirthDeathProcess(
            creation_times=[100.0],
            epoch_end_times=500.0,
            birth_rates=0.05,
            death_rates=0.02,
        )
        p2 = BirthDeathProcess(
            creation_times=[100.0],
            epoch_end_times=500.0,
            birth_rates=0.06,
            death_rates=0.02,
        )
        assert not bool(p1.equals(p2))

    def test_equals_returns_false_for_non_process_input(self) -> None:
        p = BirthDeathProcess(creation_times=[], epoch_end_times=500.0)
        assert p.equals("not a process") is False
        assert p.equals(None) is False
