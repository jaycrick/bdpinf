"""Genealogy tree plotting helpers for birth-death process simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import altair as alt
import numpy as np
import pandas as pd

from inf_bp_texts.visualization.altair_theme import PTOLQUAL

# Map a Plotly-style dash keyword onto a Vega-Lite strokeDash array.
_DASH_ARRAYS: dict[str, list[int]] = {"dot": [2, 2], "dash": [6, 4], "solid": []}


@dataclass(frozen=True, slots=True)
class GenealogyPlotConfig:
    """Configuration for genealogy-tree plots."""

    title: str = "Genealogy Tree"
    xaxis_title: str = "Time"
    yaxis_title: str = "Manuscript ID"
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


@dataclass(slots=True)
class _GeneRecord:
    dfs_index: int
    manuscript_id: int
    parent_id: int | None
    creation_time: float
    birth_time: float
    death_time: float
    plot_death_time: float
    survived: bool
    is_spine: bool = False


def _build_genealogy_maps(
    df: pd.DataFrame,
) -> tuple[dict[int, _GeneRecord], dict[int, list[int]]]:
    """Build id->_GeneRecord and parent->children maps."""
    raw_rows = df.to_dict(orient="records")
    id_to_row: dict[int, _GeneRecord] = {}
    child_map: dict[int, list[int]] = {}

    for raw_row in raw_rows:
        manuscript_id = int(cast("int", raw_row["manuscript_id"]))
        parent_raw = raw_row.get("parent_id")
        parent_id = None if pd.isna(parent_raw) else int(cast("int", parent_raw))
        creation_time = float(cast("float", raw_row["creation_time"]))
        death_raw = raw_row.get("death_time")
        death_time = (
            float(cast("float", death_raw)) if not pd.isna(death_raw) else float("nan")
        )
        survived = pd.isna(death_raw)

        id_to_row[manuscript_id] = _GeneRecord(
            dfs_index=-1,
            manuscript_id=manuscript_id,
            parent_id=parent_id,
            creation_time=creation_time,
            birth_time=creation_time,
            death_time=death_time,
            plot_death_time=death_time,
            survived=survived,
            is_spine=False,
        )

    for record in id_to_row.values():
        if record.parent_id is None:
            continue
        if record.parent_id not in id_to_row:
            msg = (
                f"parent_id {record.parent_id} does not reference a known manuscript_id"
            )
            raise ValueError(msg)
        child_map.setdefault(record.parent_id, []).append(record.manuscript_id)

    for child_ids in child_map.values():
        child_ids.sort(
            key=lambda manuscript_id: (
                id_to_row[manuscript_id].creation_time,
                manuscript_id,
            ),
        )

    return id_to_row, child_map


def _traverse_genealogy_preorder(
    root_id: int,
    id_to_row: dict[int, _GeneRecord],
    child_map: dict[int, list[int]],
    root_birth_time: float,
    simulation_end_time: float,
) -> tuple[list[_GeneRecord], int | None]:
    ordered: list[_GeneRecord] = []
    visited: set[int] = set()
    stack: list[int] = [root_id]
    dfs_idx = 0
    spine_id: int | None = None

    while stack:
        manuscript_id = stack.pop()
        if manuscript_id in visited:
            msg = "manuscript_records contain a cycle or repeated parent link"
            raise ValueError(msg)
        visited.add(manuscript_id)

        base = id_to_row[manuscript_id]
        if spine_id is None and base.survived:
            spine_id = manuscript_id

        ordered.append(
            _GeneRecord(
                dfs_index=dfs_idx,
                manuscript_id=manuscript_id,
                parent_id=base.parent_id,
                creation_time=base.creation_time,
                birth_time=(
                    root_birth_time if manuscript_id == root_id else base.creation_time
                ),
                death_time=base.death_time,
                plot_death_time=(
                    float(simulation_end_time) if base.survived else base.death_time
                ),
                survived=base.survived,
                is_spine=False,
            ),
        )
        dfs_idx += 1

        stack.extend(list(child_map.get(manuscript_id, [])))

    if len(visited) != len(id_to_row):
        msg = "manuscript_records must describe a single connected genealogy tree"
        raise ValueError(msg)

    return ordered, spine_id


def _require_genealogy_columns(records: pd.DataFrame) -> None:
    required_columns = {"manuscript_id", "parent_id", "creation_time", "death_time"}
    missing_columns = required_columns.difference(records.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        msg = (
            f"manuscript_records is missing required genealogy columns: {missing_text}"
        )
        raise ValueError(msg)


def _normalize_genealogy_records(records: pd.DataFrame) -> pd.DataFrame:
    normalized = records.copy()
    normalized["manuscript_id"] = normalized["manuscript_id"].astype(int)
    normalized["parent_id"] = normalized["parent_id"].astype("Int64")
    normalized["creation_time"] = normalized["creation_time"].astype(float)
    normalized["death_time"] = normalized["death_time"].astype(float)
    if "survived" in normalized.columns:
        normalized["survived"] = normalized["survived"].astype(bool)
    return normalized


def genealogy_dfs_order(
    records: pd.DataFrame,
    *,
    start_time: float,
    simulation_end_time: float,
) -> pd.DataFrame:
    """Return a preorder DFS ordering table for a rooted genealogy."""
    _require_genealogy_columns(records)
    normalized = _normalize_genealogy_records(records)

    if normalized["manuscript_id"].duplicated().any():
        msg = "manuscript_id values must be unique"
        raise ValueError(msg)

    root_rows = normalized[normalized["parent_id"].isna()]
    if len(root_rows) != 1:
        msg = "manuscript_records must contain exactly one root individual"
        raise ValueError(msg)

    root_row = root_rows.iloc[0]
    root_id = int(root_row["manuscript_id"])
    root_birth_time = float(start_time)
    if not np.isclose(float(root_row["creation_time"]), root_birth_time):
        msg = "root creation_time must match the process start_time"
        raise ValueError(msg)

    id_to_row, child_map = _build_genealogy_maps(normalized)
    ordered_records, spine_id = _traverse_genealogy_preorder(
        root_id,
        id_to_row,
        child_map,
        root_birth_time,
        simulation_end_time,
    )

    frame = pd.DataFrame([asdict(r) for r in ordered_records])
    if spine_id is not None:
        frame.loc[frame["manuscript_id"] == spine_id, "is_spine"] = True

    return frame.sort_values("dfs_index").reset_index(drop=True)


def _genealogy_category(*, survived: bool, is_spine: bool) -> str:
    """Classify a lineage as extinct, surviving, or the spine (first survivor)."""
    if not survived:
        return "extinct"
    return "spine" if is_spine else "survivor"


def plot_genealogy_tree(
    records: pd.DataFrame,
    *,
    start_time: float,
    simulation_end_time: float,
    config: GenealogyPlotConfig | None = None,
) -> alt.LayerChart:
    """Render a genealogy tree with lifetimes and parent-child connectors.

    Lifetimes are horizontal ``mark_rule`` segments (birth → death) on a per-row
    ordinal axis ordered by DFS index and labelled by ``manuscript_id``; their
    colour and thickness distinguish extinct / surviving / spine lineages.
    Parent→child connectors are dashed vertical ``mark_rule`` segments at the
    child's birth time, spanning the parent's and child's rows.
    """
    plot_config = config or GenealogyPlotConfig()
    frame = genealogy_dfs_order(
        records,
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
    # Ordinal y domain: manuscript ids in DFS order, so the root (index 0) is on
    # top, matching the previous reversed-axis Plotly rendering.
    order = [
        str(mid) for mid in frame.sort_values("dfs_index")["manuscript_id"].tolist()
    ]
    frame["row"] = frame["manuscript_id"].astype(str)
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


__all__ = ["GenealogyPlotConfig", "genealogy_dfs_order", "plot_genealogy_tree"]
