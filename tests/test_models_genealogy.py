"""Tests for models.genealogy: pure-pandas DFS ordering."""

import pandas as pd
import pytest

from bdpinf.models.genealogy import genealogy_dfs_order

_T = 500.0


class TestModelsGenealogy:
    """Test genealogy DFS ordering."""

    def test_genealogy_dfs_order_correct_for_simple_tree(
        self, simple_records: pd.DataFrame
    ) -> None:
        frame = genealogy_dfs_order(
            simple_records, start_time=0.0, simulation_end_time=_T
        )
        assert len(frame) == 2
        assert list(frame["individual_id"]) == [1, 2]
        assert list(frame["dfs_index"]) == [0, 1]

    def test_genealogy_dfs_order_marks_spine(
        self, simple_records: pd.DataFrame
    ) -> None:
        """DFS order result has an is_spine column with boolean values."""
        frame = genealogy_dfs_order(
            simple_records, start_time=0.0, simulation_end_time=_T
        )
        # is_spine column should exist with boolean dtype
        assert "is_spine" in frame.columns
        assert (
            frame["is_spine"].dtype == bool
            or frame["is_spine"].map(type).eq(bool).all()
            or frame["is_spine"].isin([True, False]).all()
        )

    def test_genealogy_requires_single_root(self) -> None:
        two_roots = pd.DataFrame(
            {
                "individual_id": [1, 2],
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
                "individual_id",
                "parent_id",
                "creation_time",
                "death_time",
                "survived",
                "is_spine",
            ]
        )
        with pytest.raises(
            ValueError, match=r"records must contain exactly one root individual"
        ):
            genealogy_dfs_order(empty, start_time=0.0, simulation_end_time=_T)
