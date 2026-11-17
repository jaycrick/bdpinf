"""Tests for reusable table formatting helpers."""

import datetime
import decimal
import fractions

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from bdpinf.utils import format_table_dataframe, format_table_value


class _Uncomparable:
    def __ne__(self, other):
        raise RuntimeError("uncomparable")

    def __eq__(self, other):
        raise RuntimeError("uncomparable")

    __hash__ = None


_INPUTS: dict[str, object] = {
    "none": None,
    "str": "text",
    "empty_str": "",
    "bytes": b"ab",
    "nan_str": "nan",
    "true": True,
    "false": False,
    "np_true": np.True_,
    "np_false": np.False_,
    "int_0": 0,
    "int_neg": -42,
    "int_9999": 9999,
    "int_10000": 10000,
    "int_big": 123456,
    "int_neg_big": -123456,
    "int_huge": 10**30,
    "np_int64": np.int64(7),
    "np_int64_big": np.int64(250000),
    "np_uint8": np.uint8(200),
    "np_int8_neg": np.int8(-5),
    "float_int": 5.0,
    "float_neg_int": -3.0,
    "float_zero": 0.0,
    "float_neg_zero": -0.0,
    "float_1e4": 1e4,
    "float_9999_5": 9999.5,
    "float_9999_999": 9999.999,
    "float_0_1": 0.1,
    "float_0_5": 0.5,
    "float_1_2": 1.2,
    "float_3_456": 3.456,
    "float_neg_2_5": -2.5,
    "float_0_0999": 0.0999,
    "float_0_04567": 0.04567,
    "float_tiny": 1e-12,
    "float_neg_small": -0.05,
    "float_huge": 1.234e15,
    "float_neg_huge": -2.5e9,
    "float_1e4_int": 20000.0,
    "nan": float("nan"),
    "inf": float("inf"),
    "neg_inf": float("-inf"),
    "np_float64": np.float64(1.2345),
    "np_float64_big": np.float64(123456.789),
    "np_float32": np.float32(2.5),
    "np_float32_int": np.float32(8.0),
    "np_float16": np.float16(0.25),
    "np_nan": np.float64("nan"),
    "np_inf": np.float64("inf"),
    "fraction": fractions.Fraction(1, 3),
    "fraction_int": fractions.Fraction(4, 1),
    "decimal": decimal.Decimal("1.5"),
    "decimal_nan": decimal.Decimal("NaN"),
    "complex": 1 + 2j,
    "complex_nan": complex("nan"),
    "pd_na": pd.NA,
    "pd_nat": pd.NaT,
    "pd_ts": pd.Timestamp("2020-01-01"),
    "datetime": datetime.datetime(2020, 1, 1),
    "np_datetime_nat": np.datetime64("NaT", "D"),
    "np_datetime": np.datetime64("2020-01-01"),
    "list": [1, 2],
    "list_empty": [],
    "tuple": (1.5, 2),
    "dict": {"a": 1},
    "set": {1},
    "np_array_1": np.array([1.5]),
    "np_array_2": np.array([1.5, 2.5]),
    "jax_scalar": jnp.asarray(1.2345),
    "object": object(),
    "weird": _Uncomparable(),
}

# Recorded from the pre-refactor ``format_table_value``: (type name of the
# result, its repr, or ``"<same>"`` when the input itself is returned).
_EXPECTED: dict[str, tuple[str, str]] = {
    "none": ("NoneType", "None"),
    "str": ("str", "<same>"),
    "empty_str": ("str", "<same>"),
    "bytes": ("bytes", "<same>"),
    "nan_str": ("str", "<same>"),
    "true": ("bool", "<same>"),
    "false": ("bool", "<same>"),
    "np_true": ("bool", "<same>"),
    "np_false": ("bool", "<same>"),
    "int_0": ("int", "<same>"),
    "int_neg": ("int", "<same>"),
    "int_9999": ("int", "<same>"),
    "int_10000": ("str", "'1.00e+04'"),
    "int_big": ("str", "'1.23e+05'"),
    "int_neg_big": ("str", "'-1.23e+05'"),
    "int_huge": ("str", "'1.00e+30'"),
    "np_int64": ("int", "7"),
    "np_int64_big": ("str", "'2.50e+05'"),
    "np_uint8": ("int", "200"),
    "np_int8_neg": ("int", "-5"),
    "float_int": ("int", "5"),
    "float_neg_int": ("int", "-3"),
    "float_zero": ("int", "0"),
    "float_neg_zero": ("int", "0"),
    "float_1e4": ("str", "'1.00e+04'"),
    "float_9999_5": ("str", "'9999.50'"),
    "float_9999_999": ("str", "'10000.00'"),
    "float_0_1": ("str", "'0.10'"),
    "float_0_5": ("str", "'0.50'"),
    "float_1_2": ("str", "'1.20'"),
    "float_3_456": ("str", "'3.46'"),
    "float_neg_2_5": ("str", "'-2.50'"),
    "float_0_0999": ("str", "'9.99e-02'"),
    "float_0_04567": ("str", "'4.57e-02'"),
    "float_tiny": ("str", "'1.00e-12'"),
    "float_neg_small": ("str", "'-5.00e-02'"),
    "float_huge": ("str", "'1.23e+15'"),
    "float_neg_huge": ("str", "'-2.50e+09'"),
    "float_1e4_int": ("str", "'2.00e+04'"),
    "nan": ("float", "<same>"),
    "inf": ("float", "<same>"),
    "neg_inf": ("float", "<same>"),
    "np_float64": ("str", "'1.23'"),
    "np_float64_big": ("str", "'1.23e+05'"),
    "np_float32": ("str", "'2.50'"),
    "np_float32_int": ("int", "8"),
    "np_float16": ("str", "'0.25'"),
    "np_nan": ("float64", "<same>"),
    "np_inf": ("float64", "<same>"),
    "fraction": ("str", "'0.33'"),
    "fraction_int": ("int", "4"),
    "decimal": ("Decimal", "<same>"),
    "decimal_nan": ("Decimal", "<same>"),
    "complex": ("complex", "<same>"),
    "complex_nan": ("complex", "<same>"),
    "pd_na": ("NAType", "<same>"),
    "pd_nat": ("NaTType", "<same>"),
    "pd_ts": ("Timestamp", "<same>"),
    "datetime": ("datetime", "<same>"),
    "np_datetime_nat": ("datetime64", "<same>"),
    "np_datetime": ("datetime64", "<same>"),
    "list": ("list", "<same>"),
    "list_empty": ("list", "<same>"),
    "tuple": ("tuple", "<same>"),
    "dict": ("dict", "<same>"),
    "set": ("set", "<same>"),
    "np_array_1": ("ndarray", "<same>"),
    "np_array_2": ("ndarray", "<same>"),
    "jax_scalar": ("ArrayImpl", "<same>"),
    "object": ("object", "<same>"),
    "weird": ("_Uncomparable", "<same>"),
}


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

    @pytest.mark.parametrize("case", list(_INPUTS))
    def test_format_table_value_outputs_are_unchanged(self, case: str) -> None:
        value = _INPUTS[case]
        result = format_table_value(value)
        type_name, shown = _EXPECTED[case]
        assert type(result).__name__ == type_name
        if shown == "<same>":
            assert result is value
        else:
            assert repr(result) == shown
