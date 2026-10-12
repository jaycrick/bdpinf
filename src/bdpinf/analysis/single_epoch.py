"""Helper utilities for single-epoch validation analyses."""

from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class _HasRecords(Protocol):
    @property
    def records(self) -> pd.DataFrame | None: ...


def extract_alive_ages(
    simulation_df: _HasRecords,
    time_span: float,
) -> NDArray[np.float64]:
    """Return individual ages for extant records sorted by spine flag."""
    records = simulation_df.records
    if records is None:
        msg = "extract_alive_ages requires records"
        raise ValueError(msg)

    alive_mask = records["death_time"].isna()
    alive = records.loc[alive_mask].sort_values("is_spine", ascending=False)
    ages = time_span - alive["creation_time"].to_numpy(dtype=np.float64)
    return np.asarray(ages, dtype=np.float64)
