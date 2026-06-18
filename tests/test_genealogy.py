"""Tests for visualization.genealogy: DFS ordering and tree plotting."""

import altair as alt
import pandas as pd
import pytest

from inf_bp_texts.visualization.genealogy import (
    genealogy_dfs_order,
    plot_genealogy_tree,
)


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
            "manuscript_id": [1, 2],
            "parent_id": [None, 1],
            "creation_time": [0.0, 100.0],
            "death_time": [_T, _T],
            "survived": [True, True],
            "is_spine": [False, False],
        }
    )


class TestGenealogy:
    """Test genealogy DFS ordering and tree rendering."""

    def test_genealogy_dfs_order_correct_for_simple_tree(self) -> None:
        records = _simple_records()
        frame = genealogy_dfs_order(records, start_time=0.0, simulation_end_time=_T)
        assert len(frame) == 2
        assert list(frame["manuscript_id"]) == [1, 2]
        assert list(frame["dfs_index"]) == [0, 1]

    def test_genealogy_dfs_order_marks_spine(self) -> None:
        """DFS order result has an is_spine column with boolean values."""
        records = _simple_records()
        frame = genealogy_dfs_order(records, start_time=0.0, simulation_end_time=_T)
        # is_spine column should exist with boolean dtype
        assert "is_spine" in frame.columns
        assert (
            frame["is_spine"].dtype == bool
            or frame["is_spine"].map(type).eq(bool).all()
            or frame["is_spine"].isin([True, False]).all()
        )

    def test_plot_genealogy_tree_returns_layered_rule_chart(self) -> None:
        records = _simple_records()
        chart = plot_genealogy_tree(records, start_time=0.0, simulation_end_time=_T)
        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        assert _layer_mark_types(spec) == {"rule"}
        # y axis is labelled by manuscript_id with the root (id 1) on top.
        assert spec["layer"][0]["encoding"]["y"]["scale"]["domain"] == ["1", "2"]

    def test_genealogy_requires_single_root(self) -> None:
        two_roots = pd.DataFrame(
            {
                "manuscript_id": [1, 2],
                "parent_id": [None, None],
                "creation_time": [0.0, 0.0],
                "death_time": [_T, _T],
                "survived": [True, True],
                "is_spine": [False, False],
            }
        )
        with pytest.raises(ValueError, match="exactly one root"):
            genealogy_dfs_order(two_roots, start_time=0.0, simulation_end_time=_T)

    def test_genealogy_dfs_order_empty_records(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "manuscript_id",
                "parent_id",
                "creation_time",
                "death_time",
                "survived",
                "is_spine",
            ]
        )
        with pytest.raises(ValueError):
            genealogy_dfs_order(empty, start_time=0.0, simulation_end_time=_T)
