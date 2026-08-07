"""Tests for visualization.genealogy: tree plotting."""

import altair as alt
import pandas as pd
import pytest

from bdpinf.process import BirthDeathProcess
from bdpinf.visualization.genealogy import plot_genealogy_tree


def _layer_mark_types(spec: dict) -> set[str]:
    """Collect the mark-type strings from every layer of an Altair spec dict."""
    marks: set[str] = set()
    for layer in spec.get("layer", []):
        mark = layer.get("mark")
        marks.add(mark["type"] if isinstance(mark, dict) else mark)
    return marks


_T = 500.0


def _simple_records() -> pd.DataFrame:
    """Two-generation tree: root → child (both survived)."""
    return pd.DataFrame(
        {
            "individual_id": [1, 2],
            "parent_id": [None, 1],
            "creation_time": [0.0, 100.0],
            "death_time": [_T, _T],
            "survived": [True, True],
            "is_spine": [False, False],
        }
    )


class TestGenealogy:
    """Test genealogy tree rendering."""

    def test_plot_genealogy_tree_returns_layered_rule_chart(self) -> None:
        records = _simple_records()
        chart = plot_genealogy_tree(records, start_time=0.0, simulation_end_time=_T)
        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        assert _layer_mark_types(spec) == {"rule"}
        # y axis is labelled by individual_id with the root (id 1) on top.
        assert spec["layer"][0]["encoding"]["y"]["scale"]["domain"] == ["1", "2"]

    def test_plot_genealogy_tree_missing_times_raises(self) -> None:
        with pytest.raises(ValueError, match="start_time and simulation_end_time"):
            plot_genealogy_tree(_simple_records())

    def test_plot_genealogy_tree_accepts_process_directly(self) -> None:
        records = _simple_records()
        records["age"] = [_T, _T - 100.0]
        records["lifespan"] = [float("nan"), float("nan")]
        process = BirthDeathProcess.from_records(
            records,
            epoch_end_times=_T,
            start_time=0.0,
        )
        chart = plot_genealogy_tree(process)
        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        assert spec["layer"][0]["encoding"]["y"]["scale"]["domain"] == ["1", "2"]

    def test_plot_genealogy_tree_process_without_records_raises(self) -> None:
        process = BirthDeathProcess(creation_times=[0.0], epoch_end_times=_T)
        with pytest.raises(ValueError, match="requires records"):
            plot_genealogy_tree(process)
