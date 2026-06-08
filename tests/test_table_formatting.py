"""Tests for reusable table formatting helpers."""

import numpy as np
import pandas as pd

from inf_bp_texts.utils import format_table_dataframe, format_table_value


class TestTableFormatting:
    """Test table formatting helpers."""

    def test_format_table_value_preserves_special_and_non_numeric_values(self) -> None:
        assert format_table_value(None) is None
        assert np.isnan(format_table_value(np.nan))
        assert format_table_value(np.inf) == np.inf
        assert format_table_value(-np.inf) == -np.inf
        assert format_table_value("text") == "text"
        assert format_table_value({"a": 1}) == {"a": 1}

    def test_format_table_value_formats_integers_and_floats(self) -> None:
        assert format_table_value(42) == 42
        assert format_table_value(np.int64(7)) == 7
        assert format_table_value(5.0) == 5
        assert format_table_value(1.2) == "1.20"
        assert format_table_value(3.456) == "3.46"

    def test_format_table_value_uses_scientific_notation_outside_decimal_range(
        self,
    ) -> None:
        assert format_table_value(0.04567) == "4.57e-02"
        assert format_table_value(123456) == "1.23e+05"
        assert format_table_value(123456.0) == "1.23e+05"

    def test_format_table_dataframe_formats_cell_values(self) -> None:
        dataframe = pd.DataFrame(
            {
                "integer": [42, np.int64(7)],
                "float": [1.2, 0.04567],
                "large": [123456.0, np.nan],
                "text": ["alpha", None],
            },
        )

        formatted = format_table_dataframe(dataframe)

        assert formatted["integer"].tolist() == [42, 7]
        assert formatted["float"].tolist() == ["1.20", "4.57e-02"]
        assert formatted.loc[0, "large"] == "1.23e+05"
        assert pd.isna(formatted.loc[1, "large"])
        assert formatted.loc[0, "text"] == "alpha"
        assert pd.isna(formatted.loc[1, "text"])
        assert dataframe.loc[0, "float"] == 1.2
