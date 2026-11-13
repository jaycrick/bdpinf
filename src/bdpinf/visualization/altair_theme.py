"""Altair theme and export chokepoint for birth-death process figures.

This module is the single source of truth for the project's figure house
style: every per-figure layout block (palette, fonts, axis styling) is
centralized here instead of scattered across plotting functions.

Importing this module **registers and enables** the ``bdpinf`` Altair theme
process-wide, so every chart created afterwards inherits the palette
(``ptolqual``), the serif font stack, the muted ``#FAFAFA`` background, and
minimalist Tufte axes (no gridlines, thin domain lines, small outward muted
ticks). Titles and labels use a normal (non-bold) weight by design.

:func:`save_chart` is the export chokepoint: it renders a chart to PDF (or
SVG/PNG) via vl-convert — a pure-Rust converter requiring no Chromium — and it
does **not** swallow export errors and does **not** mutate display state.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import altair as alt

from .figure_manifest import record_figure

logger = logging.getLogger(__name__)

_FIGURE_DIR_ENV_VAR = "BDPINF_FIGURE_DIR"
_figure_dir: Path | None = None


def set_figure_dir(path: str | Path | None) -> None:
    """Set the module-wide default output directory for :func:`save_chart`.

    Pass ``None`` to clear the override and fall back to the environment
    variable / cwd resolution described in :func:`save_chart`.
    """
    global _figure_dir
    _figure_dir = Path(path) if path is not None else None


def get_figure_dir() -> Path | None:
    """Return the module-wide default output directory set by :func:`set_figure_dir`."""
    return _figure_dir


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    """Resolve the effective output directory for a chart export.

    Order: explicit ``output_dir`` argument, then the module default set via
    :func:`set_figure_dir`, then the ``BDPINF_FIGURE_DIR`` environment
    variable, then ``Path.cwd() / "figures"``. This module deliberately makes
    no assumption about Quarto or any other project layout — it does not
    import anything from ``bdpinf`` itself.
    """
    if output_dir is not None:
        return Path(output_dir)
    if _figure_dir is not None:
        return _figure_dir
    env_dir = os.environ.get(_FIGURE_DIR_ENV_VAR)
    if env_dir:
        return Path(env_dir)
    return Path.cwd() / "figures"


# House style constants shared by every figure builder in this package.
FONT: str = "Libertinus Serif, Latin Modern Roman, Times New Roman, Times, serif"
FONT_SIZE: int = 13
LABEL_FONT_SIZE: int = 11
TITLE_FONT_SIZE: int = 15

BACKGROUND: str = "#FAFAFA"  # very light grey, high luminance, low saturation
AXIS_COLOR: str = "#BBBBBB"  # thin domain line
TICK_COLOR: str = "#777777"  # muted outward ticks
GRID_COLOR: str = "#D0D0D0"  # only used where a figure opts back into gridlines
GRID_COLOR_LIGHT: str = "#E8E8E8"  # faint Tufte horizontal gridlines

# Paul Tol qualitative palette — the project's categorical colour order.
PTOLQUAL: list[str] = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Violet
    "#BBBBBB",  # Gray
]

# Colourblind-friendly diverging palette (IBM/ColorBrewer RdBu), ordered from
# +1 (blue) through 0 (white) to -1 (red). The project's correlation-matrix
# heatmaps use this as an explicit `range=` (reversed, since Vega-Lite scales
# read low-to-high domain against low-to-high range) rather than a named Vega
# scheme, matching the house convention of a shared Python colour-list
# constant sliced/ordered per chart instead of `alt.Scale(scheme=...)`.
DIVERGING_RDBU: list[str] = [
    "#2166AC",  # +1  Blue
    "#4393C3",
    "#92C5DE",
    "#D1E5F0",
    "#F7F7F7",  # 0   White
    "#FDDBC7",
    "#F4A582",
    "#D6604D",
    "#B2182B",  # -1  Red
]

THEME_NAME: str = "bdpinf"

_NORMAL = "normal"


@alt.theme.register(THEME_NAME, enable=True)
def bdpinf_theme() -> alt.theme.ThemeConfig:
    """Register the project house style as an Altair theme.

    Enabled on import. Titles and labels are deliberately non-bold and the
    background is the muted ``#FAFAFA`` used across the rest of the site.
    """
    return {
        "config": {
            "background": BACKGROUND,
            "font": FONT,
            "view": {
                # Leave the inner data rectangle unfilled (the Vega-Lite default)
                # rather than painting it BACKGROUND: config.background already
                # tints the whole canvas, so the figure still reads as uniformly
                # muted, but a single knob (background) controls the look in
                # every context — override it for a paper/slide or a transparent
                # export and the whole figure recolours, with no orphaned grey
                # data rectangle left behind.
                "fill": None,
                "stroke": None,
                "continuousWidth": 480,
                "continuousHeight": 320,
            },
            "title": {
                "font": FONT,
                "fontSize": TITLE_FONT_SIZE,
                "fontWeight": _NORMAL,
                "color": "black",
                "anchor": "middle",
            },
            "axis": {
                "grid": False,
                "domainColor": AXIS_COLOR,
                "domainWidth": 1,
                "ticks": True,
                "tickColor": TICK_COLOR,
                "tickSize": 3,
                "tickWidth": 1,
                "labelColor": "black",
                "titleColor": "black",
                "labelFont": FONT,
                "titleFont": FONT,
                "labelFontWeight": _NORMAL,
                "titleFontWeight": _NORMAL,
                "labelFontSize": LABEL_FONT_SIZE,
                "titleFontSize": FONT_SIZE,
            },
            # Faint horizontal (y) gridlines only — they aid reading counts/values
            # while the x-axis stays clean (Tufte-style). The base axis config
            # keeps grids off, so this override applies to the y-axis alone.
            "axisY": {
                "grid": True,
                "gridColor": GRID_COLOR_LIGHT,
                "gridWidth": 0.5,
            },
            "legend": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelColor": "black",
                "titleColor": "black",
                "labelFontWeight": _NORMAL,
                "titleFontWeight": _NORMAL,
                "labelFontSize": LABEL_FONT_SIZE,
                "titleFontSize": LABEL_FONT_SIZE,
                "fillColor": "rgba(250,250,250,0.6)",
                "strokeColor": "rgba(0,0,0,0.08)",
                "padding": 4,
            },
            "header": {  # facet/grid panel titles
                "labelFont": FONT,
                "titleFont": FONT,
                "labelFontWeight": _NORMAL,
                "titleFontWeight": _NORMAL,
                "labelColor": "black",
                "titleColor": "black",
                "labelFontSize": LABEL_FONT_SIZE,
                "titleFontSize": FONT_SIZE,
            },
            "range": {"category": PTOLQUAL},
        },
    }


# The figures here are static publication artifacts, not interactive dashboards,
# so the 5000-row guard (meant to stop browsers choking on huge interactive specs)
# only gets in the way — e.g. multi-trajectory splitting plots. Disable it globally.
alt.data_transformers.disable_max_rows()

_fonts_registered = False

# Directories searched (in order) for the serif font files vl-convert needs to
# embed in vector exports. vl-convert does not see macOS user fonts by default,
# so the Libertinus / Latin Modern OTFs in the user font dir must be registered.
_FONT_DIR_CANDIDATES: tuple[Path, ...] = (
    Path.home() / "Library" / "Fonts",
    Path("/Library/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
)


def _size_one(
    child: alt.TopLevelMixin,
    size_props: dict[str, int],
    width: int | None,
) -> alt.TopLevelMixin:
    """Apply size overrides to a single chart, handling FacetChart children.

    When a child inside a concat/vconcat is itself a ``FacetChart``, the usual
    ``child.properties(width=..., height=...)`` path fails at schema validation
    because ``FacetChart`` accepts neither at its top level.  Mirror the same
    special-case used in :func:`_sized` for top-level facets: apply ``width``
    only, to the inner ``spec``.
    """
    if isinstance(child, alt.FacetChart):
        if width is None:
            return child
        sized = child.copy()
        inner = getattr(child, "spec", None)
        if inner is not None:
            sized.spec = inner.properties(width=width)
        return sized
    return child.properties(**size_props)


def _untitled_one(child: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Clear a single chart's ``title``, recursing into ``FacetChart.spec``.

    ``.properties(title=alt.Undefined)`` does not work to *clear* a title —
    Altair validates the passed value against the title schema even when it is
    the ``Undefined`` sentinel, which raises. Setting the attribute directly on
    a ``.copy()`` (the same trick :func:`_size_one` uses for facet ``spec``)
    is the way to actually omit the key from the serialized spec.
    """
    untitled = child.copy()
    if isinstance(untitled, alt.FacetChart):
        inner = getattr(untitled, "spec", None)
        if inner is not None:
            untitled.spec = _untitled_one(inner)
    untitled.title = alt.Undefined
    return untitled


def _untitled(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Strip the chart title (and any concat/facet child titles) for export.

    Every figure builder in this package attaches its title via
    ``.properties(title=...)`` so the *interactive* ``display()`` view stays
    self-describing in a rendered notebook. Saved PDFs are meant to be
    captioned separately in the surrounding prose, so :func:`save_chart` clears
    the title by default. Mirrors the concat/facet walk in :func:`_sized`,
    since ``HConcatChart``/``VConcatChart``/``ConcatChart`` carry titles on
    each child rather than (only) at the top level.
    """
    if isinstance(chart, alt.HConcatChart | alt.VConcatChart | alt.ConcatChart):
        untitled_chart = chart.copy()
        children_attr = (
            "hconcat"
            if isinstance(chart, alt.HConcatChart)
            else "vconcat"
            if isinstance(chart, alt.VConcatChart)
            else "concat"
        )
        setattr(
            untitled_chart,
            children_attr,
            [_untitled_one(child) for child in getattr(chart, children_attr)],
        )
        untitled_chart.title = alt.Undefined
        return untitled_chart

    return _untitled_one(chart)


def _sized(
    chart: alt.TopLevelMixin,
    width: int | None,
    height: int | None,
) -> alt.TopLevelMixin:
    """Apply width/height overrides, honouring concat-family and facet charts.

    ``HConcatChart``/``VConcatChart``/``ConcatChart`` have no top-level
    ``width``/``height`` parameter — Vega-Lite sizes each child view
    independently — so ``chart.properties(width=..., height=...)`` only fails
    validation when the chart is finally serialized, not when called. The
    override is applied to every child instead for those wrapper types.  When a
    child is itself a ``FacetChart`` (e.g. a vconcat of ridgeline facets),
    :func:`_size_one` handles it with the same facet-specific logic below.

    ``FacetChart`` similarly lacks top-level ``width``/``height``; width is
    applied to the inner ``spec`` (each cell's view width) and height is
    intentionally ignored because facet row heights are determined by the
    y-scale range and the number of rows, not by a single pixel value.
    """
    if width is None and height is None:
        return chart

    size_props: dict[str, int] = {}
    if width is not None:
        size_props["width"] = width
    if height is not None:
        size_props["height"] = height

    if isinstance(chart, alt.HConcatChart | alt.VConcatChart | alt.ConcatChart):
        sized_chart = chart.copy()
        children_attr = (
            "hconcat"
            if isinstance(chart, alt.HConcatChart)
            else "vconcat"
            if isinstance(chart, alt.VConcatChart)
            else "concat"
        )
        setattr(
            sized_chart,
            children_attr,
            [
                _size_one(child, size_props, width)
                for child in getattr(chart, children_attr)
            ],
        )
        return sized_chart

    if isinstance(chart, alt.FacetChart):
        if width is None:
            return chart
        sized_chart = chart.copy()
        inner = getattr(chart, "spec", None)
        if inner is not None:
            sized_chart.spec = inner.properties(width=width)
        return sized_chart

    return chart.properties(**size_props)


def register_export_fonts() -> None:
    """Register local font directories with vl-convert for serif PDF export.

    Best-effort and idempotent: missing directories are skipped silently and a
    failure to register a directory logs a debug message rather than raising,
    because a font fallback degrades quality but should never break an export.
    """
    global _fonts_registered
    if _fonts_registered:
        return

    import vl_convert as vlc

    for candidate in _FONT_DIR_CANDIDATES:
        if not candidate.is_dir():
            continue
        try:
            vlc.register_font_directory(str(candidate))
        except Exception as exc:  # font registration is best-effort
            logger.debug("Could not register font directory %s: %s", candidate, exc)
    _fonts_registered = True


def save_chart(
    chart: alt.TopLevelMixin,
    filename: str,
    *,
    width: int | None = None,
    height: int | None = None,
    output_dir: str | Path | None = None,
    fmt: str = "pdf",
    keep_title: bool = False,
) -> Path:
    """Export an Altair chart to ``output_dir/filename.fmt`` via vl-convert.

    This is the project's single export chokepoint: it is **pure** (it does not
    mutate display state) and it **fails loudly** — an export error propagates
    with the target filename rather than being swallowed and returning ``None``.

    Parameters
    ----------
    chart : alt.TopLevelMixin
        Any Altair chart (single, layered, faceted, or concatenated).
    filename : str
        Output filename stem (without extension).
    width, height : int | None
        Optional overrides for the chart's view dimensions in pixels. When
        ``None`` the chart's own sizing (or the theme default) is used.
    output_dir : str | Path | None
        Directory to write into; created if absent. Resolution order when
        omitted: the module default set via :func:`set_figure_dir`, then the
        ``BDPINF_FIGURE_DIR`` environment variable, then
        ``Path.cwd() / "figures"``.
    fmt : str
        Output format understood by vl-convert: ``"pdf"`` (default), ``"svg"``,
        or ``"png"``.
    keep_title : bool
        Saved figures are captioned separately in the surrounding notebook
        prose, so by default the chart's ``title`` (baked in for the
        interactive ``display()`` view) is stripped before export. Pass
        ``True`` to keep it in the exported file.

    Returns
    -------
    Path
        Path to the written file, also recorded in the directory's figure
        manifest (see :mod:`bdpinf.visualization.figure_manifest`).
    """
    output_path = _resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    register_export_fonts()

    sized = _sized(chart, width, height)
    if not keep_title:
        sized = _untitled(sized)

    file_path = output_path / f"{filename}.{fmt}"
    sized.save(str(file_path), engine="vl-convert")
    record_figure(file_path)
    return file_path


def save_chart_html(
    chart: alt.TopLevelMixin,
    filename: str,
    *,
    width: int | None = None,
    height: int | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Export an Altair chart to a self-contained interactive HTML file.

    Uses Altair's built-in HTML serialiser, which embeds the Vega-Lite
    specification together with the Vega, Vega-Lite, and vega-embed JavaScript
    libraries.  The resulting file opens in any modern browser with no server
    required and preserves all interactivity (legend-filter, brush, tooltips,
    pan/zoom).

    Mirrors :func:`save_chart` in signature and directory handling, but writes
    ``.html`` instead of a vector format and does not call
    :func:`register_export_fonts` (font rendering is left to the browser).

    Parameters
    ----------
    chart : alt.TopLevelMixin
        Any Altair chart (single, layered, faceted, or concatenated).
    filename : str
        Output filename stem (without extension).
    width, height : int | None
        Optional overrides for the chart's view dimensions in pixels. When
        ``None`` the chart's own sizing (or the theme default) is used.
    output_dir : str | Path | None
        Directory to write into; created if absent. See :func:`save_chart`
        for the resolution order when omitted.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = _resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sized = _sized(chart, width, height)

    file_path = output_path / f"{filename}.html"
    sized.save(str(file_path))
    return file_path


def display_svg(
    chart: alt.TopLevelMixin,
    *,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Render an Altair chart to static SVG and display it inline.

    Charts that inline large datasets (long per-iteration histories across many
    series, thousands of scatter points) emit their full Vega-Lite data spec
    as JSON when displayed interactively via ``IPython.display.display``. In a
    rendered Quarto notebook this lands as a single multi-megabyte line in the
    intermediate markdown, which pandoc's reader parses pathologically slowly.
    An SVG is one bounded image node instead, so pandoc sees ordinary markdown.

    Use this in place of ``display(chart)`` for data-heavy charts; keep using
    plain ``display`` for small/aggregated charts where interactivity is cheap
    and worth preserving. Mirrors :func:`save_chart`'s engine and font setup so
    the embedded image matches the house style used in PDF/SVG exports.

    Parameters
    ----------
    chart : alt.TopLevelMixin
        Any Altair chart (single, layered, faceted, or concatenated).
    width, height : int | None
        Optional overrides for the chart's view dimensions in pixels.
    """
    import io

    from IPython.display import SVG, display

    register_export_fonts()

    sized = _sized(chart, width, height)

    buf = io.StringIO()
    sized.save(buf, format="svg", engine="vl-convert")
    display(SVG(buf.getvalue()))


__all__ = [
    "BACKGROUND",
    "DIVERGING_RDBU",
    "FONT",
    "FONT_SIZE",
    "LABEL_FONT_SIZE",
    "PTOLQUAL",
    "THEME_NAME",
    "TITLE_FONT_SIZE",
    "bdpinf_theme",
    "display_svg",
    "get_figure_dir",
    "register_export_fonts",
    "save_chart",
    "save_chart_html",
    "set_figure_dir",
]
