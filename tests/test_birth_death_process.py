"""
Test suite for BirthDeathProcess class initialization and basic functionality.
"""

import numpy as np
from numpy.testing import assert_array_equal

from inf_bp_texts.inference import BirthDeathProcess


class TestBirthDeathProcessInit:
    """Test BirthDeathProcess initialization and basic methods."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        process = BirthDeathProcess()

        assert process.birth_rate == 1.0
        assert process.death_rate == 0.1
        assert process.creation_times == []
        assert process.final_time is None
        assert not process.fitted
        assert process.fit_results is None
        assert len(process.ages) == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        birth_rate = 0.05
        death_rate = 0.01
        creation_times = [100, 200, 300]
        final_time = 500

        process = BirthDeathProcess(
            birth_rate=birth_rate,
            death_rate=death_rate,
            creation_times=creation_times,
            final_time=final_time,
        )

        assert process.birth_rate == birth_rate
        assert process.death_rate == death_rate
        assert process.creation_times == creation_times
        assert process.final_time == final_time

        # Ages should be calculated automatically
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.ages, expected_ages)

    def test_add_manuscript(self):
        """Test adding manuscripts."""
        process = BirthDeathProcess()
        process.set_final_time(1000)

        # Add manuscripts
        creation_times = [100, 300, 700]
        for ct in creation_times:
            process.add_manuscript(ct)

        assert process.creation_times == creation_times
        expected_ages = np.array([900, 700, 300], dtype=np.float64)
        assert_array_equal(process.ages, expected_ages)

    def test_set_final_time(self):
        """Test setting final time."""
        process = BirthDeathProcess()
        process.creation_times = [100, 200, 300]

        # Initially no final time, so no ages
        assert len(process.ages) == 0

        # Set final time
        process.set_final_time(500)
        assert process.final_time == 500

        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(process.ages, expected_ages)

    def test_ages_update_automatically(self):
        """Test that ages update when creation_times or final_time change."""
        process = BirthDeathProcess(final_time=1000)

        # Add manuscripts and verify ages update
        process.add_manuscript(100)
        assert_array_equal(process.ages, [900])

        process.add_manuscript(300)
        assert_array_equal(process.ages, [900, 700])

        # Change final time and verify ages update
        process.set_final_time(1200)
        assert_array_equal(process.ages, [1100, 900])

    def test_get_manuscript_ages(self):
        """Test get_manuscript_ages method."""
        process = BirthDeathProcess(creation_times=[100, 200, 300], final_time=500)

        ages = process.get_manuscript_ages()
        expected_ages = np.array([400, 300, 200], dtype=np.float64)
        assert_array_equal(ages, expected_ages)

        # Verify it returns the same array as self.ages
        assert_array_equal(ages, process.ages)

    def test_get_process_summary(self):
        """Test process summary generation."""
        process = BirthDeathProcess(
            birth_rate=0.02,
            death_rate=0.005,
            creation_times=[100, 200, 300, 500],
            final_time=1000,
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
        process = BirthDeathProcess(birth_rate=0.01, death_rate=0.002)

        summary = process.get_process_summary()

        assert summary["birth_rate"] == 0.01
        assert summary["death_rate"] == 0.002
        assert summary["n_manuscripts"] == 0
        assert summary["final_time"] is None
        assert "min_age" not in summary
        assert "max_age" not in summary
        assert "mean_age" not in summary

    def test_survival_probability(self):
        """Test survival probability calculation."""
        process = BirthDeathProcess(death_rate=0.01)

        # Test known values
        assert np.isclose(process.survival_probability(0), 1.0)
        assert np.isclose(process.survival_probability(100), np.exp(-1.0))
        assert np.isclose(process.survival_probability(200), np.exp(-2.0))

    def test_expected_manuscripts(self):
        """Test expected manuscript calculation."""
        process = BirthDeathProcess(birth_rate=0.1, death_rate=0.01)

        # Expected = birth_rate * time_window * exp(-death_rate * age)
        age = 100
        time_window = 10
        expected = 0.1 * 10 * np.exp(-0.01 * 100)

        result = process.expected_manuscripts(age, time_window)
        assert np.isclose(result, expected)

    def test_type_consistency(self):
        """Test that attributes maintain correct types."""
        process = BirthDeathProcess(birth_rate=0.02, death_rate=0.005)

        assert isinstance(process.birth_rate, np.float64)
        assert isinstance(process.death_rate, np.float64)
        assert isinstance(process.creation_times, list)
        assert isinstance(process.ages, np.ndarray)
        assert process.ages.dtype == np.float64
