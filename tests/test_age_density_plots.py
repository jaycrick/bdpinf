"""Test the rates-above-age-density panel builders."""

import altair as alt
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate
from bdpinf.visualization.age_density_plots import (
    age_density_frame,
    rate_frame,
    scenario_panel,
    scenario_row,
)

_T = 3.0
_BIRTH = PiecewiseLinearRate.continuous(jnp.array([0.4, 1.6]), jnp.array([_T]))
_DEATH = PiecewiseConstantRate.constant(0.7, _T)


class TestFrames:
    def test_rate_frame_columns_and_length(self) -> None:
        frame = rate_frame("s", _BIRTH, _DEATH, n=50)
        assert list(frame.columns) == ["scenario", "t", "rate", "kind"]
        assert len(frame) == 100
        assert set(frame["kind"]) == {"λ (birth)", "μ (death)"}

    @pytest.mark.parametrize("which", ["non_spine", "spine"])
    def test_density_frame_columns(self, which: str) -> None:
        frame = age_density_frame("s", _BIRTH, _DEATH, which=which)  # pyright: ignore[reportArgumentType]
        assert list(frame.columns) == ["scenario", "age", "density"]
        assert np.all(np.isfinite(frame["density"]))

    def test_non_spine_frame_integrates_to_one(self) -> None:
        ages = jnp.linspace(0.0, _T, 20001)
        frame = age_density_frame("s", _BIRTH, _DEATH, ages, which="non_spine")
        assert np.trapezoid(frame["density"], frame["age"]) == pytest.approx(
            1.0, abs=1e-5
        )


class TestCharts:
    @pytest.mark.parametrize("primary", ["non_spine", "spine"])
    @pytest.mark.parametrize("show_secondary", [True, False])
    def test_panel_serialises(self, primary: str, show_secondary: bool) -> None:
        chart = scenario_panel(
            "s",
            _BIRTH,
            _DEATH,
            y_max=1.0,
            r_max=2.0,
            primary=primary,  # pyright: ignore[reportArgumentType]
            show_secondary=show_secondary,
        )
        assert isinstance(chart, alt.VConcatChart)
        assert chart.to_dict()

    def test_row_has_one_panel_per_scenario(self) -> None:
        row = scenario_row({"a": (_BIRTH, _DEATH), "b": (_DEATH, _DEATH)})
        assert isinstance(row, alt.HConcatChart)
        assert len(row.hconcat) == 2
        assert row.to_dict()
