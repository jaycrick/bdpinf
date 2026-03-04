"""Test suite for BirthDeathProcess class initialization and basic functionality."""

# ruff: noqa: S101

import numpy as np
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
        process = BirthDeathProcess(creation_times, epoch_end_times)

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
        process = BirthDeathProcess([], 1000.0)

        # Add manuscripts
        creation_times = [100, 300, 700]
        for ct in creation_times:
            process.add_manuscript(ct)

        assert process.creation_times == creation_times
        expected_ages = np.array([900, 700, 300], dtype=np.float64)
        assert_array_equal(process.manuscript_ages, expected_ages)

    def test_set_epoch_end_times(self) -> None:
        """Test setting epoch end times."""
        process = BirthDeathProcess([], 1000.0)
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
        process = BirthDeathProcess([], 1000.0)

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

        summary = process.get_process_summary()

        assert summary["birth_rates"] == [0.02]
        assert summary["death_rates"] == [0.005]
        assert summary["n_manuscripts"] == 4
        assert summary["epoch_end_times"] == [1000.0]
        assert summary["adjusted_epoch_end_times"] == [1000.0]
        assert summary["initial_population"] == 2
        assert summary["min_age"] == 500.0  # 1000 - 500
        assert summary["max_age"] == 900.0  # 1000 - 100
        assert (
            summary["mean_age"] == 725.0
        )  # Mean of [900, 800, 700, 500] = 2900/4 = 725

    def test_initial_population_validation(self) -> None:
        """Test initial_population requires a positive integer."""
        with pytest.raises(TypeError, match="initial_population must be an integer"):
            BirthDeathProcess([], 1000.0, initial_population=1.0)

        with pytest.raises(
            ValueError,
            match="initial_population must be a positive integer",
        ):
            BirthDeathProcess([], 1000.0, initial_population=0)

    def test_empty_process_summary(self) -> None:
        """Test summary with no manuscripts."""
        process = BirthDeathProcess([], 1000.0, birth_rates=0.01, death_rates=0.002)

        summary = process.get_process_summary()

        assert summary["birth_rates"] == [0.01]
        assert summary["death_rates"] == [0.002]
        assert summary["n_manuscripts"] == 0
        assert summary["epoch_end_times"] == [1000.0]
        assert "min_age" not in summary
        assert "max_age" not in summary
        assert "mean_age" not in summary

    def test_type_consistency(self) -> None:
        """Test that attributes maintain correct types."""
        process = BirthDeathProcess([], 1000.0, birth_rates=0.02, death_rates=0.005)

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

        process = BirthDeathProcess(creation_times, initial_epoch_end_times)

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
