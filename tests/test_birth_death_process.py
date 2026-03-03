"""Test suite for BirthDeathProcess class initialization and basic functionality."""

# ruff: noqa: S101

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from inf_bp_texts.models import BirthDeathProcess


class TestBirthDeathProcessInit:
    """Test BirthDeathProcess initialization and basic methods."""

    def test_default_initialization(self) -> None:
        """Test default parameter initialization."""
        creation_times = []
        epoch_end_times = 1000.0
        process = BirthDeathProcess(creation_times, epoch_end_times)

        assert process.birth_rate is None  # Optional, not provided
        assert process.death_rate is None  # Optional, not provided
        assert process.creation_times == []
        assert process.epoch_end_times[-1] == 1000.0
        assert (
            not process.fitted
        )  # Not fitted since birth_rate and death_rate not provided
        assert process.fit_results is None
        assert len(process.get_manuscript_ages()) == 0

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        creation_times = [100.0, 200.0, 300.0]
        epoch_end_times = 500.0

        process = BirthDeathProcess(
            creation_times=creation_times,
            epoch_end_times=epoch_end_times,
        )

        assert process.birth_rate is None  # Optional, not provided
        assert process.death_rate is None  # Optional, not provided
        assert process.creation_times == creation_times
        assert process.epoch_end_times[-1] == epoch_end_times

        # Ages should be calculated on-demand
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_ages)

    def test_add_manuscript(self) -> None:
        """Test adding manuscripts."""
        process = BirthDeathProcess([], 1000.0)

        # Add manuscripts
        creation_times = [100, 300, 700]
        for ct in creation_times:
            process.add_manuscript(ct)

        assert process.creation_times == creation_times
        expected_ages = np.array([900, 700, 300], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_ages)

    def test_set_epoch_end_times(self) -> None:
        """Test setting epoch end times."""
        process = BirthDeathProcess([], 1000.0)
        process.creation_times = [100, 200, 300]

        # Initially epoch end time is 1000, so ages should be calculated
        expected_initial_ages = np.array([900, 800, 700], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_initial_ages)

        # Update epoch end time using setter
        process.set_epoch_end_times(500)
        assert process.epoch_end_times[-1] == 500

        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.get_manuscript_ages(), expected_ages)

    def test_ages_calculated_on_demand(self) -> None:
        """Test that ages are calculated correctly on-demand when inputs change."""
        process = BirthDeathProcess([], 1000.0)

        # Add manuscripts and verify ages are calculated correctly
        process.add_manuscript(100)
        assert_array_equal(process.get_manuscript_ages(), [900])

        process.add_manuscript(300)
        assert_array_equal(process.get_manuscript_ages(), [900, 700])

        # Change epoch end time and verify ages are recalculated correctly
        process.set_epoch_end_times(1200)
        assert_array_equal(process.get_manuscript_ages(), [1100, 900])

    def test_get_manuscript_ages(self) -> None:
        """Test get_manuscript_ages method."""
        process = BirthDeathProcess(creation_times=[100, 200, 300], epoch_end_times=500)

        ages = process.get_manuscript_ages()
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(ages, expected_ages)

        # Verify multiple calls return consistent results
        ages2 = process.get_manuscript_ages()
        assert_array_equal(ages, ages2)

    def test_get_process_summary(self) -> None:
        """Test process summary generation."""
        process = BirthDeathProcess(
            creation_times=[100, 200, 300, 500],
            epoch_end_times=1000,
            birth_rate=0.02,
            death_rate=0.005,
        )

        summary = process.get_process_summary()

        assert summary["birth_rate"] == 0.02
        assert summary["death_rate"] == 0.005
        assert summary["n_manuscripts"] == 4
        assert summary["epoch_end_times"] == [1000.0]
        assert summary["min_age"] == 500.0  # 1000 - 500
        assert summary["max_age"] == 900.0  # 1000 - 100
        assert (
            summary["mean_age"] == 725.0
        )  # Mean of [900, 800, 700, 500] = 2900/4 = 725

    def test_empty_process_summary(self) -> None:
        """Test summary with no manuscripts."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.01, death_rate=0.002)

        summary = process.get_process_summary()

        assert summary["birth_rate"] == 0.01
        assert summary["death_rate"] == 0.002
        assert summary["n_manuscripts"] == 0
        assert summary["epoch_end_times"] == [1000.0]
        assert "min_age" not in summary
        assert "max_age" not in summary
        assert "mean_age" not in summary

    def test_type_consistency(self) -> None:
        """Test that attributes maintain correct types."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.02, death_rate=0.005)

        assert isinstance(process.birth_rate, np.ndarray)
        assert isinstance(process.death_rate, np.ndarray)
        assert process.birth_rate.dtype == np.float64
        assert process.death_rate.dtype == np.float64
        assert process.birth_rates is not None
        assert process.death_rates is not None
        assert process.n_epochs == 1
        assert isinstance(process.creation_times, list)

        # Test that get_manuscript_ages returns correct type
        ages = process.get_manuscript_ages()
        assert isinstance(ages, np.ndarray)
        assert ages.dtype == np.float64

    def test_manuscript_filtering_on_epoch_end_time_update(self) -> None:
        """Test that manuscripts after epoch end time are removed when updated."""
        # Start with manuscripts that span before and after a future end time
        creation_times = [100.0, 200.0, 300.0, 600.0, 800.0]
        initial_epoch_end_times = 1000.0

        process = BirthDeathProcess(creation_times, initial_epoch_end_times)

        # Initially all manuscripts should be present
        assert len(process.creation_times) == 5
        assert process.creation_times == [100.0, 200.0, 300.0, 600.0, 800.0]

        # Update epoch end time to 500.0 (should remove manuscripts at 600.0 and 800.0)
        process.set_epoch_end_times(500.0)

        # Should have only 3 manuscripts now
        assert len(process.creation_times) == 3
        assert process.creation_times == [100.0, 200.0, 300.0]
        assert process.epoch_end_times[-1] == 500.0

    def test_vector_epoch_times_and_rates(self) -> None:
        """Test vector epoch end times and vector rates with matching lengths."""
        process = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0, 700.0],
            epoch_end_times=[400.0, 800.0, 1200.0],
            birth_rate=[0.02, 0.03, 0.04],
            death_rate=[0.005, 0.007, 0.009],
        )

        assert process.n_epochs == 3
        assert process.epoch_end_times[-1] == 1200.0
        assert_array_equal(process.epoch_end_times, np.array([400.0, 800.0, 1200.0]))
        assert_array_equal(process.birth_rates, np.array([0.02, 0.03, 0.04]))
        assert_array_equal(process.death_rates, np.array([0.005, 0.007, 0.009]))

        with pytest.raises(ValueError, match="birth_rate length"):
            process.set_epoch_end_times([400.0, 800.0])
