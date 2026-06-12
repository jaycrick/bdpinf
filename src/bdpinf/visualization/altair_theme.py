"""Altair theme and export chokepoint for birth-death process figures.

This module is the single source of truth for the project's figure house style
under Altair, replacing the scattered Plotly ``apply_professional_styling`` /
``apply_tufte_axes`` helpers and the per-figure ``go.Figure`` layout blocks.

Importing this module **registers and enables** the ``inf_bp_texts`` Altair theme
process-wide, so every chart created afterwards inherits the palette
(``ptolqual``), the serif font stack, the muted ``#FAFAFA`` background, and
minimalist Tufte axes (no gridlines, thin domain lines, small outward muted
ticks). Titles and labels use a normal (non-bold) weight by design.

:func:`save_chart` is the export chokepoint: it renders a chart to PDF (or SVG/PNG)
via vl-convert — a pure-Rust converter requiring no Chromium — and, unlike the old
Plotly ``save_figure``, it does **not** swallow export errors and does **not**
mutate display state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import altair as alt

logger = logging.getLogger(__name__)

# House style constants (ported verbatim from the Plotly ``plotting`` module so
# migrated figures match the existing site palette exactly).
FONT: str = "Libertinus Serif, Latin Modern Roman, Times New Roman, Times, serif"
FONT_SIZE: int = 13
LABEL_FONT_SIZE: int = 11
TITLE_FONT_SIZE: int = 15

BACKGROUND: str = "#FAFAFA"  # very light grey, high luminance, low saturation
AXIS_COLOR: str = "#BBBBBB"  # thin domain line
TICK_COLOR: str = "#777777"  # muted outward ticks
GRID_COLOR: str = "#D0D0D0"  # only used where a figure opts back into gridlines

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

THEME_NAME: str = "inf_bp_texts"

_NORMAL = "normal"


@alt.theme.register(THEME_NAME, enable=True)
def inf_bp_texts_theme() -> alt.theme.ThemeConfig:
    """Register the project house style as an Altair theme.

    Enabled on import. Titles and labels are deliberately non-bold and the
    background is the muted ``#FAFAFA`` used across the rest of the site.
    """
    return {
        "config": {
            "background": BACKGROUND,
            "font": FONT,
            "view": {
                "fill": BACKGROUND,
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


def register_export_fonts() -> None:
    """Register local font directories with vl-convert for serif PDF export.

    Best-effort and idempotent: missing directories are skipped silently and a
    failure to register a directory logs a debug message rather than raising,
    because a font fallback degrades quality but should never break an export.
    """
    global _fonts_registered  # noqa: PLW0603
    if _fonts_registered:
        return

    import vl_convert as vlc  # noqa: PLC0415

    for candidate in _FONT_DIR_CANDIDATES:
        if not candidate.is_dir():
            continue
        try:
            vlc.register_font_directory(str(candidate))
        except Exception as exc:  # noqa: BLE001 - font registration is best-effort
            logger.debug("Could not register font directory %s: %s", candidate, exc)
    _fonts_registered = True


def save_chart(
    chart: alt.TopLevelMixin,
    filename: str,
    *,
    width: int | None = None,
    height: int | None = None,
    output_dir: str | Path = "figures",
    fmt: str = "pdf",
) -> Path:
    """Export an Altair chart to ``output_dir/filename.fmt`` via vl-convert.

    The export chokepoint that replaces the Plotly ``save_figure``. Unlike the
    old helper it is **pure** (it does not mutate display state) and it **fails
    loudly** — an export error propagates with the target filename rather than
    being swallowed and returning ``None``.

    Parameters
    ----------
    chart : alt.TopLevelMixin
        Any Altair chart (single, layered, faceted, or concatenated).
    filename : str
        Output filename stem (without extension).
    width, height : int | None
        Optional overrides for the chart's view dimensions in pixels. When
        ``None`` the chart's own sizing (or the theme default) is used.
    output_dir : str | Path
        Directory to write into; created if absent. Defaults to ``"figures"``.
    fmt : str
        Output format understood by vl-convert: ``"pdf"`` (default), ``"svg"``,
        or ``"png"``.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    register_export_fonts()

    sized = chart
    if width is not None or height is not None:
        size_props: dict[str, int] = {}
        if width is not None:
            size_props["width"] = width
        if height is not None:
            size_props["height"] = height
        sized = chart.properties(**size_props)

    file_path = output_path / f"{filename}.{fmt}"
    sized.save(str(file_path), engine="vl-convert")
    return file_path


__all__ = [
    "BACKGROUND",
    "FONT",
    "FONT_SIZE",
    "LABEL_FONT_SIZE",
    "PTOLQUAL",
    "THEME_NAME",
    "TITLE_FONT_SIZE",
    "inf_bp_texts_theme",
    "register_export_fonts",
    "save_chart",
]
