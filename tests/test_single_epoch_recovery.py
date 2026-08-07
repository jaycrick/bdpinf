"""Tests for the single-epoch bulk-inference recovery-study helpers."""

from typing import Any

import numpy as np
import pytest

import bdpinf.analysis.single_epoch_recovery as ser


class _FakeSampledAges:
    """Stand-in for the BirthDeathProcess-like object simulate_bdp_ages returns."""

    def __init__(self, spine: list[float], non_spine: list[float]) -> None:
        self.spine_ages = np.asarray(spine, dtype=np.float64)
        self.non_spine_ages = np.asarray(non_spine, dtype=np.float64)


def _fake_success_result(
    birth: float, death: float, optimizer: str = "lbfgs"
) -> dict[str, Any]:
    return {
        "success": True,
        "birth_rates": np.array([birth]),
        "death_rates": np.array([death]),
        "optimization_result": {
            "attempts": [{"optimizer": optimizer}],
            "selected_attempt_index": 0,
        },
    }


class TestSampleSingleEpochAges:
    """Test sample_single_epoch_ages."""

    def test_concatenates_spine_and_non_spine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_simulate(*args: Any, **kwargs: Any) -> _FakeSampledAges:
            return _FakeSampledAges([1.0, 2.0], [3.0])

        monkeypatch.setattr(ser, "simulate_bdp_ages_from_distributions", fake_simulate)

        ages = ser.sample_single_epoch_ages(
            seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert isinstance(ages, np.ndarray)
        np.testing.assert_array_equal(np.sort(ages), [1.0, 2.0, 3.0])


class TestFitSingleEpochReplicate:
    """Test fit_single_epoch_replicate."""

    def test_empty_ages_returns_none(self) -> None:
        result = ser.fit_single_epoch_replicate(
            np.array([], dtype=np.float64),
            birth_rate=0.5,
            death_rate=0.1,
            time_span=15.0,
        )
        assert result is None

    def test_success_delegates_to_fit_birth_death_processes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_fit(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return _fake_success_result(0.5, 0.1)

        monkeypatch.setattr(ser, "fit_birth_death_processes", fake_fit)

        ages = np.array([1.0, 2.0, 3.0])
        result = ser.fit_single_epoch_replicate(
            ages, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )

        assert result is not None
        assert result["success"] is True
        assert captured["times"] == 15.0
        assert captured["is_ordered"] is True
        assert captured["perturb_initial_params_std"] == 0.0
        np.testing.assert_array_equal(captured["initial_birth_rates"], [0.5])
        np.testing.assert_array_equal(captured["initial_death_rates"], [0.1])
        assert captured["optimizers"] == ["lbfgs", "adam"]

    def test_exception_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_fit(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("optimizer blew up")

        monkeypatch.setattr(ser, "fit_birth_death_processes", fake_fit)

        result = ser.fit_single_epoch_replicate(
            np.array([1.0]), birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert result is None


class TestRunSingleEpochRecoveryReplicate:
    """Test run_single_epoch_recovery_replicate."""

    def test_successful_replicate_row_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_sample(*args: Any, **kwargs: Any) -> np.ndarray:
            return np.array([1.0, 2.0, 15.0])  # includes a boundary age

        def fake_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return _fake_success_result(0.48, 0.09, optimizer="adam")

        monkeypatch.setattr(ser, "sample_single_epoch_ages", fake_sample)
        monkeypatch.setattr(ser, "fit_single_epoch_replicate", fake_fit)

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
        assert row["optimizer"] == "adam"

    def test_non_boundary_replicate_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_sample(*args: Any, **kwargs: Any) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])  # max < time_span

        def fake_fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return _fake_success_result(0.5, 0.1)

        monkeypatch.setattr(ser, "sample_single_epoch_ages", fake_sample)
        monkeypatch.setattr(ser, "fit_single_epoch_replicate", fake_fit)

        row = ser.run_single_epoch_recovery_replicate(
            simulation_index=0, seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert row["has_boundary_age"] is False

    def test_empty_ages_replicate_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ser,
            "sample_single_epoch_ages",
            lambda *a, **k: np.array([], dtype=np.float64),
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
            ser, "sample_single_epoch_ages", lambda *a, **k: np.array([1.0, 2.0])
        )
        monkeypatch.setattr(ser, "fit_single_epoch_replicate", lambda *a, **k: None)

        row = ser.run_single_epoch_recovery_replicate(
            simulation_index=0, seed=0, birth_rate=0.5, death_rate=0.1, time_span=15.0
        )
        assert row["success"] is False
        assert row["population"] == 2
        assert np.isnan(row["birth_rate"])
        assert row["optimizer"] == "none"
