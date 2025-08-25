"""
Test suite for BirthDeathProcess class initialization and basic functionality.
"""

import numpy as np
from numpy.testing import assert_array_equal

from inf_bp_texts.models import BirthDeathProcess


class TestBirthDeathProcessInit:
    """Test BirthDeathProcess initialization and basic methods."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        creation_times = []
        final_time = 1000.0
        process = BirthDeathProcess(creation_times, final_time)

        assert process.birth_rate is None  # Optional, not provided
        assert process.death_rate is None  # Optional, not provided
        assert process.creation_times == []
        assert process.final_time == 1000.0
        assert (
            not process.fitted
        )  # Not fitted since birth_rate and death_rate not provided
        assert process.fit_results is None
        assert len(process.get_manuscript_ages()) == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        creation_times = [100.0, 200.0, 300.0]
        final_time = 500.0

        process = BirthDeathProcess(
            creation_times=creation_times,
            final_time=final_time,
        )

        assert process.birth_rate is None  # Optional, not provided
        assert process.death_rate is None  # Optional, not provided
        assert process.creation_times == creation_times
        assert process.final_time == final_time

        # Ages should be calculated on-demand
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_ages)

    def test_add_manuscript(self):
        """Test adding manuscripts."""
        process = BirthDeathProcess([], 1000.0)

        # Add manuscripts
        creation_times = [100, 300, 700]
        for ct in creation_times:
            process.add_manuscript(ct)

        assert process.creation_times == creation_times
        expected_ages = np.array([900, 700, 300], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_ages)

    def test_set_final_time(self):
        """Test setting final time."""
        process = BirthDeathProcess([], 1000.0)
        process.creation_times = [100, 200, 300]

        # Initially final_time is 1000, so ages should be calculated
        expected_initial_ages = np.array([900, 800, 700], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_initial_ages)

        # Update final time using the property
        process.final_time = 500
        assert process.final_time == 500

        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_ages)

    def test_ages_calculated_on_demand(self):
        """Test that ages are calculated correctly on-demand when creation_times or final_time change."""
        process = BirthDeathProcess([], 1000.0)

        # Add manuscripts and verify ages are calculated correctly
        process.add_manuscript(100)
        assert_array_equal(process.get_manuscript_ages(), [900])

        process.add_manuscript(300)
        assert_array_equal(process.get_manuscript_ages(), [900, 700])

        # Change final time and verify ages are recalculated correctly
        process.set_final_time(1200)
        assert_array_equal(process.get_manuscript_ages(), [1100, 900])

    def test_get_manuscript_ages(self):
        """Test get_manuscript_ages method."""
        process = BirthDeathProcess(creation_times=[100, 200, 300], final_time=500)

        ages = process.get_manuscript_ages()
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(ages, expected_ages)

        # Verify multiple calls return consistent results
        ages2 = process.get_manuscript_ages()
        assert_array_equal(ages, ages2)

    def test_get_process_summary(self):
        """Test process summary generation."""
        process = BirthDeathProcess(
            creation_times=[100, 200, 300, 500],
            final_time=1000,
            birth_rate=0.02,
            death_rate=0.005,
        )

        summary = process.get_process_summary()

        assert summary["birth_rate"] == 0.02
        assert summary["death_rate"] == 0.005
        assert summary["n_manuscripts"] == 4
        assert summary["final_time"] == 1000
        assert summary["min_age"] == 500.0  # 1000 - 500
        assert summary["max_age"] == 900.0  # 1000 - 100
        assert (
            summary["mean_age"] == 725.0
        )  # Mean of [900, 800, 700, 500] = 2900/4 = 725

    def test_empty_process_summary(self):
        """Test summary with no manuscripts."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.01, death_rate=0.002)

        summary = process.get_process_summary()

        assert summary["birth_rate"] == 0.01
        assert summary["death_rate"] == 0.002
        assert summary["n_manuscripts"] == 0
        assert summary["final_time"] == 1000.0  # final_time is now always set
        assert "min_age" not in summary
        assert "max_age" not in summary
        assert "mean_age" not in summary

    def test_type_consistency(self):
        """Test that attributes maintain correct types."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.02, death_rate=0.005)

        assert isinstance(process.birth_rate, np.float64)
        assert isinstance(process.death_rate, np.float64)
        assert isinstance(process.creation_times, list)

        # Test that get_manuscript_ages returns correct type
        ages = process.get_manuscript_ages()
        assert isinstance(ages, np.ndarray)
        assert ages.dtype == np.float64

    def test_manuscript_filtering_on_final_time_update(self):
        """Test that manuscripts after final_time are removed when final_time is updated."""
        # Start with manuscripts that span before and after a future final_time
        creation_times = [100.0, 200.0, 300.0, 600.0, 800.0]
        initial_final_time = 1000.0

        process = BirthDeathProcess(creation_times, initial_final_time)

        # Initially all manuscripts should be present
        assert len(process.creation_times) == 5
        assert process.creation_times == [100.0, 200.0, 300.0, 600.0, 800.0]

        # Update final_time to 500.0 (should remove manuscripts at 600.0 and 800.0)
        process.final_time = 500.0

        # Should have only 3 manuscripts now
        assert len(process.creation_times) == 3
        assert process.creation_times == [100.0, 200.0, 300.0]
        assert process.final_time == 500.0
