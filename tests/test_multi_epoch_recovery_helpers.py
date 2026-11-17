"""Tests for the multi-epoch bulk-inference recovery-study helpers."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import bdpinf.analysis.multi_epoch_recovery as mer
import bdpinf.analysis.validation_helpers as vh
from bdpinf.analysis.validation_helpers import FitModelConfig
from bdpinf.inference import Constraints, FitResult


class TestRunTwoRateReplicate:
    """Test run_two_rate_replicate (03_bulk_inference.qmd)."""

    _TIMES = np.array([7.5, 15.0])
    _FIT_CONFIG = FitModelConfig(times=_TIMES, is_ordered=True)

    def test_dropped_when_ages_dont_span_both_epochs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            mer, "simulate_birth_death", lambda *_args, **_kwargs: object()
        )
        monkeypatch.setattr(
            mer, "extract_alive_ages", lambda *_a, **_k: np.array([1.0, 2.0])
        )
        monkeypatch.setattr(mer, "has_ages_in_each_epoch", lambda *_a, **_k: False)

        row = mer.run_two_rate_replicate(
            simulation_index=0,
            sim_seed=0,
            birth_rates_true=np.array([0.5, 0.7]),
            death_rates_true=np.array([0.1, 0.15]),
            times=self._TIMES,
            fit_config=self._FIT_CONFIG,
        )
        assert row is None

    def test_kept_replicate_row_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_fit_result: Callable[..., FitResult],
    ) -> None:
        monkeypatch.setattr(
            mer, "simulate_birth_death", lambda *_args, **_kwargs: object()
        )
        monkeypatch.setattr(
            mer, "extract_alive_ages", lambda *_a, **_k: np.array([1.0, 10.0])
        )
        monkeypatch.setattr(mer, "has_ages_in_each_epoch", lambda *_a, **_k: True)
        monkeypatch.setattr(
            mer,
            "fit_single_model",
            lambda *_a, **_k: make_fit_result(
                [0.51, 0.72], [0.11, 0.16], times=[7.5, 15.0]
            ),
        )

        row = mer.run_two_rate_replicate(
            simulation_index=3,
            sim_seed=0,
            birth_rates_true=np.array([0.5, 0.7]),
            death_rates_true=np.array([0.1, 0.15]),
            times=self._TIMES,
            fit_config=self._FIT_CONFIG,
        )
        assert row is not None
        assert row["simulation_index"] == 3
        assert row["epoch_1_birth"] == pytest.approx(0.51)
        assert row["epoch_1_death"] == pytest.approx(0.11)
        assert row["epoch_2_birth"] == pytest.approx(0.72)
        assert row["epoch_2_death"] == pytest.approx(0.16)

    def test_dropped_when_fit_unsuccessful(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            mer, "simulate_birth_death", lambda *_args, **_kwargs: object()
        )
        monkeypatch.setattr(
            mer, "extract_alive_ages", lambda *_a, **_k: np.array([1.0, 10.0])
        )
        monkeypatch.setattr(mer, "has_ages_in_each_epoch", lambda *_a, **_k: True)
        monkeypatch.setattr(mer, "fit_single_model", lambda *_a, **_k: None)

        row = mer.run_two_rate_replicate(
            simulation_index=0,
            sim_seed=0,
            birth_rates_true=np.array([0.5, 0.7]),
            death_rates_true=np.array([0.1, 0.15]),
            times=self._TIMES,
            fit_config=self._FIT_CONFIG,
        )
        assert row is None


class TestRunSingleRateReplicate:
    """Test run_single_rate_replicate (04_bulk_inference_single_rate.qmd)."""

    _FIT_TIMES = np.array([7.5, 15.0])
    _FIT_CONFIG = FitModelConfig(times=_FIT_TIMES, is_ordered=True)

    def test_dropped_when_ages_dont_span_both_epochs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            mer, "sample_single_epoch_ages", lambda *_a, **_k: np.array([1.0])
        )
        monkeypatch.setattr(mer, "has_ages_in_each_epoch", lambda *_a, **_k: False)

        row = mer.run_single_rate_replicate(
            simulation_index=0,
            sim_seed=0,
            birth_rate=0.5,
            death_rate=0.1,
            time_span=15.0,
            fit_times=self._FIT_TIMES,
            fit_config=self._FIT_CONFIG,
        )
        assert row is None

    def test_kept_replicate_row_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_fit_result: Callable[..., FitResult],
    ) -> None:
        monkeypatch.setattr(
            mer, "sample_single_epoch_ages", lambda *_a, **_k: np.array([1.0, 10.0])
        )
        monkeypatch.setattr(mer, "has_ages_in_each_epoch", lambda *_a, **_k: True)
        monkeypatch.setattr(
            mer,
            "fit_single_model",
            lambda *_a, **_k: make_fit_result(
                [0.49, 0.48], [0.09, 0.11], times=[7.5, 15.0]
            ),
        )

        row = mer.run_single_rate_replicate(
            simulation_index=1,
            sim_seed=0,
            birth_rate=0.5,
            death_rate=0.1,
            time_span=15.0,
            fit_times=self._FIT_TIMES,
            fit_config=self._FIT_CONFIG,
        )
        assert row is not None
        assert row["simulation_index"] == 1
        assert row["epoch_1_birth"] == pytest.approx(0.49)
        assert row["epoch_2_death"] == pytest.approx(0.11)


class TestSimulateZeroBirthSecondEpoch:
    """Test simulate_zero_birth_second_epoch."""

    def test_alive_survivors_get_death_or_survive_epoch_2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        split_time, final_time = 7.5, 15.0
        records_df = pd.DataFrame(
            {
                "death_time": [np.nan, 3.0, np.nan],
                "creation_time": [0.0, 0.0, 0.0],
                "is_spine": [True, False, False],
            }
        )

        class _FakeFirstEpoch:
            records = records_df

        monkeypatch.setattr(
            mer, "simulate_birth_death", lambda *_args, **_kwargs: _FakeFirstEpoch()
        )

        result = mer.simulate_zero_birth_second_epoch(
            first_epoch_seed=1,
            death_time_seed=2,
            birth_rate_epoch_1=1.0,
            death_rate=0.1,
            times=np.array([split_time, final_time]),
        )

        assert len(result) == 3
        # Row 1 already died in epoch 1: untouched.
        assert result.loc[1, "death_time"] == pytest.approx(3.0)
        # Rows 0 and 2 were alive at the split: either still alive (NaN) or
        # died strictly within epoch 2's window.
        for idx in (0, 2):
            death_time = result.loc[idx, "death_time"]
            if not np.isnan(death_time):
                assert split_time < death_time < final_time


class TestFitZeroBirthReplicate:
    """Test fit_zero_birth_replicate."""

    _CONFIG = FitModelConfig(
        times=np.array([7.5, 15.0]), is_ordered=True, condition_on_survival=True
    )
    _CONSTRAINTS = Constraints(
        fixed={"birth.v1": 0.0}, equal=[["death.v0", "death.v1"]]
    )

    def test_empty_ages_returns_none(self) -> None:
        result = mer.fit_zero_birth_replicate(
            np.array([], dtype=np.float64), self._CONFIG, self._CONSTRAINTS
        )
        assert result is None

    def test_forwards_constraints_to_the_fit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_fit_result: Callable[..., FitResult],
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_fit(*_args: Any, **kwargs: Any) -> FitResult:
            captured.update(kwargs)
            return make_fit_result([0.98, 0.0], [0.11, 0.11], times=[7.5, 15.0])

        monkeypatch.setattr(vh, "fit", fake_fit)

        result = mer.fit_zero_birth_replicate(
            np.array([1.0, 2.0]), self._CONFIG, self._CONSTRAINTS
        )
        assert result is not None
        assert captured["constraints"] is self._CONSTRAINTS
        assert result.birth_values[1] == 0.0

    def test_exception_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_fit(*_args: Any, **_kwargs: Any) -> FitResult:
            raise RuntimeError("optimizer blew up")

        monkeypatch.setattr(vh, "fit", fake_fit)

        result = mer.fit_zero_birth_replicate(
            np.array([1.0]), self._CONFIG, self._CONSTRAINTS
        )
        assert result is None


class TestRunZeroBirthReplicate:
    """Test run_zero_birth_replicate."""

    _TIMES = np.array([7.5, 15.0])
    _CONFIG = FitModelConfig(times=_TIMES, is_ordered=True)
    _CONSTRAINTS = Constraints(
        fixed={"birth.v1": 0.0}, equal=[["death.v0", "death.v1"]]
    )

    def _fake_simulated(self, ages: list[float]) -> pd.DataFrame:
        n = len(ages)
        return pd.DataFrame(
            {
                "death_time": [np.nan] * n,
                "creation_time": [15.0 - age for age in ages],
                "is_spine": [i == 0 for i in range(n)],
            }
        )

    def test_successful_replicate_row_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_fit_result: Callable[..., FitResult],
    ) -> None:
        monkeypatch.setattr(
            mer,
            "simulate_zero_birth_second_epoch",
            lambda **_kwargs: self._fake_simulated([1.0, 15.0]),
        )
        monkeypatch.setattr(
            mer,
            "fit_zero_birth_replicate",
            lambda *_a, **_k: make_fit_result(
                [0.98, 0.0], [0.11, 0.11], times=[7.5, 15.0]
            ),
        )

        row = mer.run_zero_birth_replicate(
            simulation_index=4,
            first_epoch_seed=1,
            death_time_seed=2,
            birth_rate_epoch_1=1.0,
            death_rate=0.1,
            times=self._TIMES,
            fit_config=self._CONFIG,
            param_constraints=self._CONSTRAINTS,
        )
        assert row["simulation_index"] == 4
        assert row["success"] is True
        assert row["population"] == 2
        assert row["mean_age"] == pytest.approx(8.0)
        assert row["has_boundary_age"] is True
        assert row["birth_rate"] == pytest.approx(0.98)
        assert row["death_rate"] == pytest.approx(0.11)

    def test_failed_fit_replicate_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            mer,
            "simulate_zero_birth_second_epoch",
            lambda **_kwargs: self._fake_simulated([1.0, 2.0]),
        )
        monkeypatch.setattr(mer, "fit_zero_birth_replicate", lambda *_a, **_k: None)

        row = mer.run_zero_birth_replicate(
            simulation_index=0,
            first_epoch_seed=1,
            death_time_seed=2,
            birth_rate_epoch_1=1.0,
            death_rate=0.1,
            times=self._TIMES,
            fit_config=self._CONFIG,
            param_constraints=self._CONSTRAINTS,
        )
        assert row["success"] is False
        assert row["population"] == 2
        assert np.isnan(row["birth_rate"])

    def test_no_survivors_replicate_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            mer,
            "simulate_zero_birth_second_epoch",
            lambda **_kwargs: self._fake_simulated([]),
        )

        row = mer.run_zero_birth_replicate(
            simulation_index=0,
            first_epoch_seed=1,
            death_time_seed=2,
            birth_rate_epoch_1=1.0,
            death_rate=0.1,
            times=self._TIMES,
            fit_config=self._CONFIG,
            param_constraints=self._CONSTRAINTS,
        )
        assert row["success"] is False
        assert row["population"] == 0
        assert np.isnan(row["mean_age"])
