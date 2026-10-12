"""Tests for single-epoch analysis helper utilities."""

from typing import Any

import numpy as np
import pandas as pd

from bdpinf.analysis import (
    extract_alive_ages,
)
from bdpinf.process import BirthDeathProcess


def _layer_mark_types(spec: dict[str, Any]) -> set[str]:
    """Collect the mark-type strings from every layer of an Altair spec dict."""
    marks: set[str] = set()
    for layer in spec.get("layer", []):
        mark = layer.get("mark")
        if isinstance(mark, dict):
            marks.add(mark["type"])
        elif isinstance(mark, str):
            marks.add(mark)
    return marks


class TestSingleEpochAnalysis:
    """Test single-epoch analysis helpers."""

    def test_extract_alive_ages_sorts_by_spine_flag(self) -> None:
        simulation_df = pd.DataFrame(
            {
                "creation_time": [100.0, 50.0, 10.0],
                "death_time": [np.nan, np.nan, 80.0],
                "is_spine": [False, True, False],
            },
        )
        process = BirthDeathProcess(
            creation_times=[100.0, 50.0],
            epoch_end_times=200.0,
            records=simulation_df,
        )

        ages = extract_alive_ages(process, time_span=200.0)
        # Alive rows are creation times [50.0, 100.0] after spine-first sorting.
        assert np.allclose(ages, np.array([150.0, 100.0], dtype=np.float64))
