"""Pure-pandas genealogy record ordering (no plotting dependency)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass(slots=True)
class _GeneRecord:
    dfs_index: int
    individual_id: int
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
        individual_id = int(cast("int", raw_row["individual_id"]))
        parent_raw = raw_row.get("parent_id")
        parent_id = None if pd.isna(parent_raw) else int(cast("int", parent_raw))
        creation_time = float(cast("float", raw_row["creation_time"]))
        death_raw = raw_row.get("death_time")
        death_time = (
            float(cast("float", death_raw)) if not pd.isna(death_raw) else float("nan")
        )
        survived = pd.isna(death_raw)

        id_to_row[individual_id] = _GeneRecord(
            dfs_index=-1,
            individual_id=individual_id,
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
                f"parent_id {record.parent_id} does not reference a known individual_id"
            )
            raise ValueError(msg)
        child_map.setdefault(record.parent_id, []).append(record.individual_id)

    for child_ids in child_map.values():
        child_ids.sort(
            key=lambda individual_id: (
                id_to_row[individual_id].creation_time,
                individual_id,
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
        individual_id = stack.pop()
        if individual_id in visited:
            msg = "records contain a cycle or repeated parent link"
            raise ValueError(msg)
        visited.add(individual_id)

        base = id_to_row[individual_id]
        if spine_id is None and base.survived:
            spine_id = individual_id

        ordered.append(
            _GeneRecord(
                dfs_index=dfs_idx,
                individual_id=individual_id,
                parent_id=base.parent_id,
                creation_time=base.creation_time,
                birth_time=(
                    root_birth_time if individual_id == root_id else base.creation_time
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

        stack.extend(list(child_map.get(individual_id, [])))

    if len(visited) != len(id_to_row):
        msg = "records must describe a single connected genealogy tree"
        raise ValueError(msg)

    return ordered, spine_id


def _require_genealogy_columns(records: pd.DataFrame) -> None:
    required_columns = {"individual_id", "parent_id", "creation_time", "death_time"}
    missing_columns = required_columns.difference(records.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        msg = f"records is missing required genealogy columns: {missing_text}"
        raise ValueError(msg)


def _normalize_genealogy_records(records: pd.DataFrame) -> pd.DataFrame:
    normalized = records.copy()
    normalized["individual_id"] = normalized["individual_id"].astype(int)
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

    if normalized["individual_id"].duplicated().any():
        msg = "individual_id values must be unique"
        raise ValueError(msg)

    root_rows = normalized[normalized["parent_id"].isna()]
    if len(root_rows) != 1:
        msg = "records must contain exactly one root individual"
        raise ValueError(msg)

    root_row = root_rows.iloc[0]
    root_id = int(root_row["individual_id"])
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
        frame.loc[frame["individual_id"] == spine_id, "is_spine"] = True

    return frame.sort_values("dfs_index").reset_index(drop=True)


__all__ = ["genealogy_dfs_order"]
