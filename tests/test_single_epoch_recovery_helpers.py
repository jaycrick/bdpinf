"""Tests for the single-epoch bulk-inference recovery-study helpers."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import bdpinf.analysis.single_epoch_recovery as ser
from bdpinf.inference import FitResult


class _FakeSampledAges:
    """Stand-in for the BirthDeathProcess-like object simulate_bdp_ages returns."""

    def __init__(self, spine: list[float], non_spine: list[float]) -> None:
        self.spine_ages = np.asarray(spine, dtype=np.float64)
        self.non_spine_ages = np.asarray(non_spine, dtype=np.float64)


class TestSampleSingleEpochAges:
    """Test sample_single_epoch_ages."""

    def test_concatenates_spine_and_non_spine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_simulate(*_args: Any, **_kwargs: Any) -> _FakeSampledAges:
            return _FakeSampledAges([1.0, 2.0], [3.0])

        monkeypatch.setattr(ser, "simulate_bdp_ages_from_distributions", fake_simulate)

        ages = ser.sample_single_epoch_ages(
            seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert isinstance(ages, np.ndarray)
        np.testing.assert_array_equal(np.sort(ages), [1.0, 2.0, 3.0])


class TestRunSingleEpochRecoveryReplicate:
    """Test run_single_epoch_recovery_replicate."""

    def test_successful_replicate_row_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_fit_result: Callable[..., FitResult],
    ) -> None:
        def fake_sample(*_args: Any, **_kwargs: Any) -> np.ndarray:
            return np.array([1.0, 2.0, 15.0])  # includes a boundary age

        def fake_fit(*_args: Any, **_kwargs: Any) -> FitResult:
            return make_fit_result([0.48], [0.09], method="L-BFGS-B")

        monkeypatch.setattr(ser, "sample_single_epoch_ages", fake_sample)
        monkeypatch.setattr(ser, "fit_single_model", fake_fit)

        row = ser.run_single_epoch_recovery_replicate(
            simulation_index=7, seed=123, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )

        assert row["simulation_index"] == 7
        assert row["success"] is True
        assert row["population"] == 3
        assert row["mean_age"] == pytest.approx(6.0)
        assert row["has_boundary_age"] is True
        assert row["birth_rate"] == pytest.approx(0.48)
        assert row["death_rate"] == pytest.approx(0.09)
        assert row["optimizer"] == "L-BFGS-B"

    def test_fit_is_configured_from_true_rates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_fit_result: Callable[..., FitResult],
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_fit(_ages: Any, config: Any) -> FitResult:
            captured["config"] = config
            return make_fit_result([0.5], [0.1])

        monkeypatch.setattr(
            ser, "sample_single_epoch_ages", lambda *_a, **_k: np.array([1.0, 2.0])
        )
        monkeypatch.setattr(ser, "fit_single_model", fake_fit)

        ser.run_single_epoch_recovery_replicate(
            simulation_index=0, seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )

        config = captured["config"]
        assert config.times == 15.0
        assert config.is_ordered is True
        np.testing.assert_array_equal(config.initial_birth_rates, [0.5])
        np.testing.assert_array_equal(config.initial_death_rates, [0.1])
        assert config.method == "trust-exact"

    def test_non_boundary_replicate_row(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_fit_result: Callable[..., FitResult],
    ) -> None:
        def fake_sample(*_args: Any, **_kwargs: Any) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])  # max < time_span

        def fake_fit(*_args: Any, **_kwargs: Any) -> FitResult:
            return make_fit_result([0.5], [0.1])

        monkeypatch.setattr(ser, "sample_single_epoch_ages", fake_sample)
        monkeypatch.setattr(ser, "fit_single_model", fake_fit)

        row = ser.run_single_epoch_recovery_replicate(
            simulation_index=0, seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert row["has_boundary_age"] is False

    def test_empty_ages_replicate_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ser,
            "sample_single_epoch_ages",
            lambda *_a, **_k: np.array([], dtype=np.float64),
        )

        row = ser.run_single_epoch_recovery_replicate(
            simulation_index=0, seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert row["success"] is False
        assert row["population"] == 0
        assert np.isnan(row["mean_age"])
        assert row["optimizer"] == "none"

    def test_failed_fit_replicate_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ser, "sample_single_epoch_ages", lambda *_a, **_k: np.array([1.0, 2.0])
        )
        monkeypatch.setattr(ser, "fit_single_model", lambda *_a, **_k: None)

        row = ser.run_single_epoch_recovery_replicate(
            simulation_index=0, seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert row["success"] is False
        assert row["population"] == 2
        assert np.isnan(row["birth_rate"])
        assert row["optimizer"] == "none"
