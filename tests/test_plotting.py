"""Tests for visualization plotting utilities."""

# ruff: noqa: S101

import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from inf_bp_texts.visualization.plotting import (
    apply_professional_styling,
    apply_tufte_axes,
    apply_tufte_legend,
    plot_bootstrap_confidence_intervals,
    plot_parameter_comparison,
    plot_simulation_results,
    save_figure,
)


class DummyTrace:
    """Simple trace container used by fake Plotly classes."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the trace with arbitrary keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments to store in the trace.

        """
        self.kwargs = kwargs


class DummyFigure:
    """Minimal figure object that captures plotting calls for assertions."""

    def __init__(self, *, should_fail_write: bool = False) -> None:
        """Initialize the figure with optional failure mode for write operations.

        Parameters
        ----------
        should_fail_write : bool, optional
            Whether write_image should raise an error, by default False.

        """
        self.should_fail_write = should_fail_write
        self.layout_updates: list[dict[str, Any]] = []
        self.xaxes_updates: list[dict[str, Any]] = []
        self.yaxes_updates: list[dict[str, Any]] = []
        self.traces: list[tuple[DummyTrace, int | None, int | None]] = []
        self.show_called = False
        self.write_image_path: str | None = None

    def update_layout(self, **kwargs: dict[str, Any]) -> None:
        """Update the layout with the provided keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Layout properties to update.

        """
        self.layout_updates.append(kwargs)

    def update_xaxes(self, **kwargs: dict[str, Any]) -> None:
        """Update the x-axes with the provided keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            x-axes properties to update.

        """
        self.xaxes_updates.append(kwargs)

    def update_yaxes(self, **kwargs: dict[str, Any]) -> None:
        """Update the y-axes with the provided keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            y-axes properties to update.

        """
        self.yaxes_updates.append(kwargs)

    def add_trace(
        self,
        trace: DummyTrace,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        """Add a trace to the figure.

        Parameters
        ----------
        trace : DummyTrace
            The trace object to add.
        row : int | None, optional
            The row index for the trace, by default None.
        col : int | None, optional
            The column index for the trace, by default None.

        """
        self.traces.append((trace, row, col))

    def show(self) -> None:
        """Mark that show was called on this figure."""
        self.show_called = True

    def write_image(self, path: str) -> None:
        """Write an image to the specified path.

        Parameters
        ----------
        path : str
            The file path where the image should be written.

        Raises
        ------
        RuntimeError
            If should_fail_write is True.

        """
        if self.should_fail_write:
            msg = "write failed"
            raise RuntimeError(msg)
        self.write_image_path = path
        Path(path).write_text("pdf", encoding="utf-8")


@pytest.fixture
def fake_plotly(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[DummyFigure]]:
    """Install fake plotly modules in sys.modules and expose created figures."""
    created_subplot_figures: list[DummyFigure] = []
    created_figure_figures: list[DummyFigure] = []

    plotly_module = types.ModuleType("plotly")
    go_module = types.ModuleType("plotly.graph_objects")
    subplots_module = types.ModuleType("plotly.subplots")

    go_module_any = cast("Any", go_module)
    subplots_module_any = cast("Any", subplots_module)
    plotly_module_any = cast("Any", plotly_module)

    go_module_any.Histogram = DummyTrace
    go_module_any.Scatter = DummyTrace
    go_module_any.Bar = DummyTrace

    def figure_factory() -> DummyFigure:
        fig = DummyFigure()
        created_figure_figures.append(fig)
        return fig

    go_module_any.Figure = figure_factory

    def make_subplots(**kwargs: object) -> DummyFigure:
        del kwargs
        fig = DummyFigure()
        created_subplot_figures.append(fig)
        return fig

    subplots_module_any.make_subplots = make_subplots

    # Optional but helpful for "from plotly import graph_objects/subplots" import styles
    plotly_module_any.graph_objects = go_module
    plotly_module_any.subplots = subplots_module

    monkeypatch.setitem(sys.modules, "plotly", plotly_module)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go_module)
    monkeypatch.setitem(sys.modules, "plotly.subplots", subplots_module)

    return {
        "subplot_figures": created_subplot_figures,
        "figure_figures": created_figure_figures,
    }


def test_apply_professional_styling_updates_layout() -> None:
    """apply_professional_styling should populate core layout fields."""
    height = 420
    top_margin = 15

    fig = DummyFigure()
    returned = apply_professional_styling(fig, "Title", "X", "Y", height=height)

    assert returned is fig
    assert len(fig.layout_updates) == 1
    update = fig.layout_updates[0]
    assert update["title"] == "Title"
    assert update["xaxis_title"] == "X"
    assert update["yaxis_title"] == "Y"
    assert update["height"] == height
    assert update["margin"]["t"] == top_margin


def test_apply_tufte_legend_fallback_position() -> None:
    """apply_tufte_legend should fall back to top-right for unknown positions."""
    fallback_position = 0.98

    fig = DummyFigure()
    apply_tufte_legend(fig, position="unknown", showlegend=False)

    update = fig.layout_updates[0]
    assert update["showlegend"] is False
    assert update["legend"]["x"] == fallback_position
    assert update["legend"]["y"] == fallback_position
    assert update["legend"]["xanchor"] == "right"
    assert update["legend"]["yanchor"] == "top"


def test_apply_tufte_axes_updates_both_axes() -> None:
    """apply_tufte_axes should apply minimalist settings to both axes."""
    axis_color = "#111111"
    tick_color = "#222222"
    tick_len = 4

    fig = DummyFigure()
    apply_tufte_axes(
        fig,
        axis_color=axis_color,
        tick_color=tick_color,
        tick_len=tick_len,
    )

    assert len(fig.xaxes_updates) == 1
    assert len(fig.yaxes_updates) == 1
    assert fig.xaxes_updates[0]["showgrid"] is False
    assert fig.yaxes_updates[0]["linecolor"] == axis_color
    assert fig.xaxes_updates[0]["tickcolor"] == tick_color
    assert fig.xaxes_updates[0]["ticklen"] == tick_len


def test_save_figure_success_returns_path(tmp_path: Path) -> None:
    """save_figure should return the output PDF path on successful export."""
    fig = DummyFigure()

    saved = save_figure(fig, "my_plot", width=220, height=180, output_dir=str(tmp_path))

    assert saved == tmp_path / "my_plot.pdf"
    assert saved is not None
    assert saved.exists()
    first_update = fig.layout_updates[0]
    assert first_update["margin"] == {"l": 30, "r": 8, "t": 8, "b": 35, "pad": 2}


def test_save_figure_failure_returns_none(tmp_path: Path) -> None:
    """save_figure should return None when image writing fails."""
    fig = DummyFigure(should_fail_write=True)

    saved = save_figure(fig, "bad_plot", output_dir=str(tmp_path))

    assert saved is None


def test_plot_parameter_comparison_length_mismatch_raises() -> None:
    """plot_parameter_comparison should reject mismatched input lengths."""
    with pytest.raises(ValueError, match="must have same length"):
        plot_parameter_comparison(
            fit_results_list=[
                {
                    "success": True,
                    "birth_rate": 1,
                    "death_rate": 1,
                    "log_likelihood": -1,
                },
            ],
            parameter_names=["a", "b"],
        )


def test_plot_parameter_comparison_no_successful_fits_prints(
    capsys: pytest.CaptureFixture[str],
    fake_plotly: dict[str, list[DummyFigure]],
) -> None:
    """plot_parameter_comparison should print and exit when all fits fail."""
    plot_parameter_comparison(
        fit_results_list=[
            {
                "success": False,
                "birth_rate": 0.1,
                "death_rate": 0.05,
                "log_likelihood": -10,
            },
        ],
        parameter_names=["set1"],
    )

    out = capsys.readouterr().out
    assert "No successful fits to plot." in out
    assert fake_plotly["subplot_figures"] == []


def test_plot_parameter_comparison_success_builds_traces(
    fake_plotly: dict[str, list[DummyFigure]],
) -> None:
    """plot_parameter_comparison should create traces for successful fit entries."""
    expected_traces = 3
    plot_parameter_comparison(
        fit_results_list=[
            {
                "success": True,
                "birth_rate": 0.2,
                "death_rate": 0.1,
                "log_likelihood": -12.3,
            },
            {
                "success": False,
                "birth_rate": 0.3,
                "death_rate": 0.2,
                "log_likelihood": -10.0,
            },
            {
                "success": True,
                "birth_rate": 0.4,
                "death_rate": 0.2,
                "log_likelihood": -8.1,
            },
        ],
        parameter_names=["A", "B", "C"],
    )

    fig = fake_plotly["subplot_figures"][0]
    assert len(fig.traces) == expected_traces
    assert fig.show_called is True


def test_plot_simulation_results_no_survivors_prints(
    capsys: pytest.CaptureFixture[str],
    fake_plotly: dict[str, list[DummyFigure]],
) -> None:
    """plot_simulation_results should report when no manuscripts survive."""
    df = pd.DataFrame(
        {
            "survived": [False, False],
            "age": [10.0, 20.0],
            "creation_time": [1.0, 2.0],
        },
    )

    plot_simulation_results(df)

    out = capsys.readouterr().out
    assert "No surviving manuscripts to plot." in out
    assert fake_plotly["subplot_figures"] == []


def test_plot_simulation_results_success_creates_expected_traces(
    fake_plotly: dict[str, list[DummyFigure]],
) -> None:
    """plot_simulation_results should build expected traces and show the figure."""
    expected_traces = 5
    df = pd.DataFrame(
        {
            "survived": [True, True, False],
            "age": [50.0, 100.0, 150.0],
            "creation_time": [10.0, 20.0, 30.0],
        },
    )

    plot_simulation_results(df, age_bins=np.array([0.0, 50.0, 100.0, 150.0]))

    fig = fake_plotly["subplot_figures"][0]
    assert len(fig.traces) == expected_traces
    assert fig.show_called is True


def test_plot_bootstrap_confidence_intervals_failed_prints(
    capsys: pytest.CaptureFixture[str],
    fake_plotly: dict[str, list[DummyFigure]],
) -> None:
    """plot_bootstrap_confidence_intervals should print when bootstrap failed."""
    plot_bootstrap_confidence_intervals({"success": False})
    out = capsys.readouterr().out
    assert "Bootstrap analysis failed, cannot plot." in out
    assert fake_plotly["figure_figures"] == []


def test_plot_bootstrap_confidence_intervals_success_builds_figure(
    fake_plotly: dict[str, list[DummyFigure]],
) -> None:
    """plot_bootstrap_confidence_intervals should render two parameter traces."""
    expected_traces = 2
    results = {
        "success": True,
        "original_fit": {"birth_rate": 0.5, "death_rate": 0.2},
        "birth_rate_ci": (0.4, 0.6),
        "death_rate_ci": (0.1, 0.3),
        "confidence_level": 0.95,
        "n_bootstrap": 500,
    }

    plot_bootstrap_confidence_intervals(results)

    fig = fake_plotly["figure_figures"][0]
    assert len(fig.traces) == expected_traces
    assert fig.show_called is True
    title = fig.layout_updates[-1]["title"]
    assert "95% CI" in title
    assert "500 bootstrap samples" in title
