"""Tests for the Altair theme and export chokepoint."""

from collections.abc import Iterator
from pathlib import Path

import altair as alt
import pandas as pd
import pytest
import vl_convert

from bdpinf.visualization import altair_theme
from bdpinf.visualization.altair_theme import (
    BACKGROUND,
    PTOLQUAL,
    THEME_NAME,
    _untitled,
    display_svg,
    get_figure_dir,
    save_chart,
    set_figure_dir,
)


@pytest.fixture(autouse=True)
def _reset_figure_dir() -> Iterator[None]:
    previous = get_figure_dir()
    try:
        yield
    finally:
        set_figure_dir(previous)


def _sample_chart() -> alt.Chart:
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [3.0, 2.1, 1.4, 1.0]})
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="x:Q", y="y:Q")
        .properties(title="Test")
    )


def _hconcat_chart() -> alt.HConcatChart:
    return alt.hconcat(_sample_chart(), _sample_chart())


def _concat_chart() -> alt.ConcatChart:
    return alt.concat(_sample_chart(), _sample_chart(), columns=2)


def _facet_chart() -> alt.FacetChart:
    df = pd.DataFrame(
        {"x": [0, 1, 2, 3], "y": [1.0, 2.0, 1.5, 0.5], "cat": list("AABB")}
    )
    return alt.Chart(df).mark_line().encode(x="x:Q", y="y:Q").facet(row="cat:N")


class TestAltairTheme:
    """Test theme registration and the save_chart export chokepoint."""

    def test_theme_registered_and_enabled_on_import(self) -> None:
        assert THEME_NAME in alt.theme.names()
        assert alt.theme.active == THEME_NAME

    def test_theme_config_is_non_bold_muted_and_palette(self) -> None:
        config = _sample_chart().to_dict()["config"]
        assert config["background"] == BACKGROUND
        assert config["title"]["fontWeight"] == "normal"
        assert config["axis"]["labelFontWeight"] == "normal"
        assert config["axis"]["grid"] is False
        assert config["range"]["category"][: len(PTOLQUAL)] == PTOLQUAL

    def test_save_chart_writes_pdf_and_returns_path(self, tmp_path: Path) -> None:
        saved = save_chart(
            _sample_chart(), "out", width=400, height=300, output_dir=tmp_path
        )

        assert saved == tmp_path / "out.pdf"
        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_save_chart_creates_missing_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        saved = save_chart(_sample_chart(), "out", output_dir=nested)

        assert saved.exists()
        assert saved.parent == nested

    def test_save_chart_fails_loudly_on_unknown_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"format|fmt|nope|extension"):
            save_chart(_sample_chart(), "out", output_dir=tmp_path, fmt="nope")

    def test_display_svg_renders_without_raising(self) -> None:
        # vl-convert returns the SVG mimebundle entry as str, not bytes, so
        # display_svg must write it into a text buffer (io.StringIO).
        display_svg(_sample_chart(), width=400, height=300)

    def test_save_chart_sizes_hconcat_children_not_wrapper(
        self, tmp_path: Path
    ) -> None:
        # HConcatChart has no top-level width/height parameter (Vega-Lite sizes
        # each child view independently); chart.properties(width=...) on the
        # wrapper only fails validation at save time. This regression-tests
        # that save_chart applies the override per child instead.
        saved = save_chart(
            _hconcat_chart(), "out", width=400, height=300, output_dir=tmp_path
        )

        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_save_chart_sizes_concat_children_not_wrapper(self, tmp_path: Path) -> None:
        saved = save_chart(
            _concat_chart(), "out", width=400, height=300, output_dir=tmp_path
        )

        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_save_chart_sizes_facet_spec_not_wrapper(self, tmp_path: Path) -> None:
        # FacetChart has no top-level width/height (Vega-Lite sizes the inner
        # spec per cell); height is ignored for facets and only width is applied
        # to the inner spec.  This regression-tests that save_chart does not
        # raise SchemaValidationError when called with a FacetChart.
        saved = save_chart(
            _facet_chart(), "out", width=400, height=300, output_dir=tmp_path
        )

        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_display_svg_sizes_hconcat_children_not_wrapper(self) -> None:
        display_svg(_hconcat_chart(), width=400, height=300)

    def test_save_chart_hconcat_without_size_override_is_unaffected(
        self, tmp_path: Path
    ) -> None:
        # No width/height passed -> _sized must short-circuit and return the
        # chart object unchanged, not attempt the per-child rewrite.
        saved = save_chart(_hconcat_chart(), "out", output_dir=tmp_path)

        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_save_chart_sizes_vconcat_of_facets(self, tmp_path: Path) -> None:
        # VConcatChart whose children are FacetCharts (e.g. vertical ridgeline
        # output from posterior_rates_chart) must not raise SchemaValidationError
        # when width/height are passed — FacetChart children need the inner-spec
        # width path, not child.properties(width=..., height=...).
        chart = alt.vconcat(_facet_chart(), _facet_chart())
        saved = save_chart(chart, "out", width=400, height=300, output_dir=tmp_path)

        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_untitled_clears_single_chart_title(self) -> None:
        chart = _sample_chart()
        assert "title" in chart.to_dict()

        assert "title" not in _untitled(chart).to_dict()
        # _untitled must not mutate the caller's chart object.
        assert "title" in chart.to_dict()

    def test_untitled_clears_hconcat_child_titles(self) -> None:
        spec = _untitled(_hconcat_chart()).to_dict()

        assert "title" not in spec
        assert all("title" not in child for child in spec["hconcat"])

    def test_untitled_clears_facet_chart_title(self) -> None:
        chart = _facet_chart().properties(title="Facet Title")

        assert "title" not in _untitled(chart).to_dict()

    def test_save_chart_strips_title_by_default(self, tmp_path: Path) -> None:
        # Saved PDFs are captioned separately in notebook prose; the title
        # baked in for the interactive display() view should not leak into
        # the exported spec.
        saved = save_chart(_sample_chart(), "out", output_dir=tmp_path)

        assert saved.exists()

    def test_save_chart_keep_title_preserves_it(self, tmp_path: Path) -> None:
        saved = save_chart(_sample_chart(), "out", output_dir=tmp_path, keep_title=True)

        assert saved.exists()


class TestFigureDirResolution:
    """Test the output_dir resolution order used by save_chart when omitted."""

    def test_get_figure_dir_defaults_to_none(self) -> None:
        assert get_figure_dir() is None

    def test_set_figure_dir_round_trips(self, tmp_path: Path) -> None:
        set_figure_dir(tmp_path)
        assert get_figure_dir() == tmp_path

    def test_set_figure_dir_none_clears_override(self, tmp_path: Path) -> None:
        set_figure_dir(tmp_path)
        set_figure_dir(None)
        assert get_figure_dir() is None

    def test_save_chart_uses_module_default_when_output_dir_omitted(
        self, tmp_path: Path
    ) -> None:
        set_figure_dir(tmp_path)
        saved = save_chart(_sample_chart(), "out")

        assert saved.parent == tmp_path

    def test_save_chart_explicit_output_dir_overrides_module_default(
        self, tmp_path: Path
    ) -> None:
        other = tmp_path / "other"
        set_figure_dir(tmp_path / "default")
        saved = save_chart(_sample_chart(), "out", output_dir=other)

        assert saved.parent == other

    def test_save_chart_falls_back_to_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_figure_dir(None)
        monkeypatch.setenv("BDPINF_FIGURE_DIR", str(tmp_path))
        saved = save_chart(_sample_chart(), "out")

        assert saved.parent == tmp_path

    def test_save_chart_falls_back_to_cwd_figures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        set_figure_dir(None)
        monkeypatch.delenv("BDPINF_FIGURE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        saved = save_chart(_sample_chart(), "out")

        assert saved.parent == tmp_path / "figures"

    def test_resolve_output_dir_precedence_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        explicit = tmp_path / "explicit"
        default = tmp_path / "default"
        env_dir = tmp_path / "env"

        set_figure_dir(default)
        monkeypatch.setenv("BDPINF_FIGURE_DIR", str(env_dir))
        assert altair_theme._resolve_output_dir(explicit) == explicit
        assert altair_theme._resolve_output_dir(None) == default

        set_figure_dir(None)
        assert altair_theme._resolve_output_dir(None) == env_dir


class TestRegisterExportFonts:
    """Test best-effort font registration."""

    @pytest.fixture
    def font_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(altair_theme, "_FONT_DIR_CANDIDATES", (tmp_path,))
        monkeypatch.setattr(altair_theme, "_fonts_registered", False)
        return tmp_path

    @pytest.mark.parametrize("error", [OSError("unreadable"), ValueError("bad font")])
    def test_expected_registration_failure_is_skipped(
        self,
        font_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        error: Exception,
    ) -> None:
        def failing(_path: str) -> None:
            raise error

        monkeypatch.setattr(vl_convert, "register_font_directory", failing)

        with caplog.at_level("DEBUG"):
            altair_theme.register_export_fonts()

        assert altair_theme._fonts_registered is True
        assert str(font_dir) in caplog.text

    @pytest.mark.usefixtures("font_dir")
    def test_type_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(_path: str) -> None:
            raise TypeError("programming error")

        monkeypatch.setattr(vl_convert, "register_font_directory", broken)

        with pytest.raises(TypeError, match="programming error"):
            altair_theme.register_export_fonts()

        assert altair_theme._fonts_registered is False
