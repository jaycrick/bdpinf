"""Genealogy tree plotting helpers for birth-death process simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import altair as alt
import pandas as pd

from bdpinf.models.genealogy import genealogy_dfs_order
from bdpinf.visualization.altair_theme import PTOLQUAL

if TYPE_CHECKING:
    from bdpinf.process import BirthDeathProcess

# Map a named dash style onto a Vega-Lite strokeDash array.
_DASH_ARRAYS: dict[str, list[int]] = {"dot": [2, 2], "dash": [6, 4], "solid": []}


@dataclass(frozen=True, slots=True)
class GenealogyPlotConfig:
    """Configuration for genealogy-tree plots."""

    title: str = "Genealogy Tree"
    xaxis_title: str = "Time"
    yaxis_title: str = "Individual ID"
    width: int = 1100
    height: int | None = None
    lifetime_color: str = "#000000"
    survivor_color: str = PTOLQUAL[2]
    spine_color: str = PTOLQUAL[2]
    connector_color: str = "#666666"
    lifetime_width: float = 2.0
    survivor_width: float = 2.5
    spine_width: float = 4.0
    connector_width: float = 1.25
    connector_dash: str = "dot"


def _genealogy_category(*, survived: bool, is_spine: bool) -> str:
    """Classify a lineage as extinct, surviving, or the spine (first survivor)."""
    if not survived:
        return "extinct"
    return "spine" if is_spine else "survivor"


def plot_genealogy_tree(
    records: pd.DataFrame | BirthDeathProcess,
    *,
    start_time: float | None = None,
    simulation_end_time: float | None = None,
    config: GenealogyPlotConfig | None = None,
) -> alt.LayerChart:
    """Render a genealogy tree with lifetimes and parent-child connectors.

    Accepts either a raw records ``pd.DataFrame`` with explicit ``start_time``
    and ``simulation_end_time``, or a single :class:`~bdpinf.process.BirthDeathProcess`
    instance, from which ``.records``, ``.start_time``, and
    ``.latest_epoch_end_time`` are read directly.

    Lifetimes are horizontal ``mark_rule`` segments (birth → death) on a per-row
    ordinal axis ordered by DFS index and labelled by ``individual_id``; their
    colour and thickness distinguish extinct / surviving / spine lineages.
    Parent→child connectors are dashed vertical ``mark_rule`` segments at the
    child's birth time, spanning the parent's and child's rows.

    Raises
    ------
    ValueError
        If a ``BirthDeathProcess`` is passed with no ``records`` table, or if
        a raw ``pd.DataFrame`` is passed without both ``start_time`` and
        ``simulation_end_time``.
    """
    if isinstance(records, pd.DataFrame):
        records_df = records
        if start_time is None or simulation_end_time is None:
            msg = (
                "start_time and simulation_end_time are required when records "
                "is a plain DataFrame"
            )
            raise ValueError(msg)
    else:
        process = records
        if process.records is None:
            msg = (
                "plot_genealogy_tree requires records with individual_id and parent_id"
            )
            raise ValueError(msg)
        records_df = process.records
        start_time = process.start_time
        simulation_end_time = float(process.latest_epoch_end_time)

    plot_config = config or GenealogyPlotConfig()
    frame = genealogy_dfs_order(
        records_df,
        start_time=start_time,
        simulation_end_time=simulation_end_time,
    )

    if frame.empty:
        empty = pd.DataFrame({"birth_time": [], "plot_death_time": [], "row": []})
        return cast(
            "alt.LayerChart",
            alt.layer(alt.Chart(empty).mark_rule()).properties(title=plot_config.title),
        )

    frame = frame.copy()
    frame["category"] = [
        _genealogy_category(survived=bool(s), is_spine=bool(sp))
        for s, sp in zip(frame["survived"], frame["is_spine"], strict=True)
    ]
    # Ordinal y domain: individual ids in DFS order, so the root (index 0) is
    # on top.
    order = [
        str(mid) for mid in frame.sort_values("dfs_index")["individual_id"].tolist()
    ]
    frame["row"] = frame["individual_id"].astype(str)
    y_scale = alt.Scale(domain=order)

    categories = ["extinct", "survivor", "spine"]
    color_scale = alt.Scale(
        domain=categories,
        range=[
            plot_config.lifetime_color,
            plot_config.survivor_color,
            plot_config.spine_color,
        ],
    )
    size_scale = alt.Scale(
        domain=categories,
        range=[
            plot_config.lifetime_width,
            plot_config.survivor_width,
            plot_config.spine_width,
        ],
    )

    lifetimes = (
        alt.Chart(frame)
        .mark_rule()
        .encode(
            x=alt.X("birth_time:Q", title=plot_config.xaxis_title),
            x2="plot_death_time:Q",
            y=alt.Y("row:N", scale=y_scale, title=plot_config.yaxis_title),
            color=alt.Color("category:N", scale=color_scale, legend=None),
            size=alt.Size("category:N", scale=size_scale, legend=None),
        )
    )

    layers: list[alt.Chart] = [lifetimes]

    connector_frame = frame[frame["parent_id"].notna()].copy()
    if not connector_frame.empty:
        connector_frame["parent_row"] = (
            connector_frame["parent_id"].astype(int).astype(str)
        )
        connectors = (
            alt.Chart(connector_frame)
            .mark_rule(
                color=plot_config.connector_color,
                strokeWidth=plot_config.connector_width,
                strokeDash=_DASH_ARRAYS.get(plot_config.connector_dash, [2, 2]),
            )
            .encode(
                x="birth_time:Q",
                y=alt.Y("parent_row:N", scale=y_scale),
                y2="row:N",
            )
        )
        layers.append(connectors)

    computed_height = (
        max(420, 28 * len(frame) + 120)
        if plot_config.height is None
        else plot_config.height
    )
    return cast(
        "alt.LayerChart",
        alt.layer(*layers).properties(
            title=plot_config.title,
            width=plot_config.width,
            height=computed_height,
        ),
    )


__all__ = ["GenealogyPlotConfig", "plot_genealogy_tree"]
