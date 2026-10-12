"""Tests for recovery-study persistence utilities (inference/persistence.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bdpinf.inference.persistence import (
    load_recovery_results,
    save_recovery_results,
)


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simulation_index": [0, 1, 2],
            "success": [True, False, True],
            "birth_rate": [0.5, np.nan, 0.6],
            "optimizer": ["lbfgs", "none", "adam"],
        }
    )


class TestSaveLoadRecoveryResults:
    """Test save_recovery_results and load_recovery_results."""

    def test_round_trip_columns_and_values(self) -> None:
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.npz"
            save_recovery_results(df, path)
            loaded = load_recovery_results(path)
        pd.testing.assert_frame_equal(loaded["results"], df)

    def test_column_order_preserved(self) -> None:
        df = _make_df()[["optimizer", "birth_rate", "simulation_index", "success"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.npz"
            save_recovery_results(df, path)
            loaded = load_recovery_results(path)
        assert list(loaded["results"].columns) == list(df.columns)

    def test_dtypes_preserved(self) -> None:
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.npz"
            save_recovery_results(df, path)
            loaded = load_recovery_results(path)
        result_df = loaded["results"]
        assert result_df["simulation_index"].dtype == np.int64
        assert result_df["success"].dtype == bool
        assert result_df["birth_rate"].dtype == np.float64

    def test_metadata_round_trip(self) -> None:
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.npz"
            save_recovery_results(df, path, metadata={"n_simulated": 10, "n_kept": 3})
            loaded = load_recovery_results(path)
        assert loaded["metadata"] == {"n_simulated": 10, "n_kept": 3}

    def test_metadata_defaults_to_empty(self) -> None:
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.npz"
            save_recovery_results(df, path)
            loaded = load_recovery_results(path)
        assert loaded["metadata"] == {}

    def test_load_without_extension(self) -> None:
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results"
            save_recovery_results(df, path)
            loaded = load_recovery_results(path)
        assert len(loaded["results"]) == len(df)

    def test_file_not_found_raises(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            pytest.raises(FileNotFoundError),
        ):
            load_recovery_results(Path(tmp) / "nonexistent.npz")

    def test_parent_dir_created(self) -> None:
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "results.npz"
            save_recovery_results(df, path)
            assert path.exists()

    def test_empty_dataframe_round_trip(self) -> None:
        df = pd.DataFrame({"a": pd.Series([], dtype=np.float64)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.npz"
            save_recovery_results(df, path)
            loaded = load_recovery_results(path)
        assert len(loaded["results"]) == 0
        assert list(loaded["results"].columns) == ["a"]
