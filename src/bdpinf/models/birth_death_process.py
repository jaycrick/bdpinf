"""Core Birth-Death Process model for manuscript survival and creation analysis.

This module contains the main BirthDeathProcess class with basic data management,
initialization, and core functionality. Complex mathematical computations and
fitting procedures are handled by separate modules.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from inf_bp_texts.models.validation import (
    ValidationConfig,
    normalize_and_validate_inputs,
)
from inf_bp_texts.utils import ensure_1d_tensor

if TYPE_CHECKING:
    import altair as alt

    from inf_bp_texts.visualization.genealogy import GenealogyPlotConfig

NumericInput = float | list[float] | NDArray[np.float64] | jax.Array
RateLabel = Literal["birth", "death"]


class BirthDeathProcess:
    r"""Canonical birth--death process representation for manuscript data.

    This class is the canonical output for the simulators in this package and
    represents a birth--death process used to model manuscript creation and
    loss. It stores observation data (creation times or a full manuscript
    record table), optional sampled spine/non-spine ages, epoch boundaries, and
    per-epoch rate vectors when available.

    Key attributes and accessors
    - `creation_times` (list[float]): times when manuscripts were created.
    - `epoch_end_times` (jax.Array): strictly increasing epoch end times.
    - `manuscript_ages` (jax.Array): ages computed as latest epoch end time
        minus creation times.
    - `manuscript_records` (pd.DataFrame | None): full event table when
        available; may include `death_time`, `age`, `lifespan`, `is_spine`, and
        `survived`. When present, survivors' `death_time` values may be NaN.
    - `spine_ages`, `non_spine_ages` (jax.Array | None): sampled age
        vectors produced by the age-sampling simulator.
    - `death_times` (jax.Array | None): vector of death times extracted
        from `manuscript_records` when present.
    - `surviving_manuscripts`, `extinct_manuscripts` (pd.DataFrame | None):
        convenience views into the record table.

    Construction
    - `BirthDeathProcess(...)` accepts `creation_times` and `epoch_end_times`.
    - `from_age_samples(...)` builds a process from sampled spine and
        non-spine ages (useful when simulators return only sampled ages).
    - `from_manuscript_records(...)` builds a process from a full event table
        (useful when simulators return detailed records including extinct
        manuscripts).

    Key methods

    - `to_newick`: export the full genealogy as a strict Newick-format tree
        string (requires `manuscript_records` with `manuscript_id` and
        `parent_id`).
    - `genealogy_dfs_order`: validate the manuscript record table and return
        a preorder DFS ordering for the rooted genealogy used by the
        serializer and plotting helpers.
    - `plot_genealogy_tree`: return an Altair chart showing manuscript
        lifetimes and the genealogy (requires `manuscript_records`).
    - `log_likelihood` / `negative_log_likelihood`: compute the likelihood
        given per-epoch `birth_rates` and `death_rates` (stored internally as
        `jax.Array`).
    - `update_parameters`: update the model parameters after fitting.

    Examples
    --------
            Construct from sampled ages and inspect results::

                    >>> proc = BirthDeathProcess.from_age_samples(
                    ...     spine_ages=[10.0, 5.0], non_spine_ages=[2.0],
                    ...     epoch_end_times=50
                    ... )
                    >>> proc.spine_ages
                    Array([10.,  5.], dtype=float64)

            Construct from a manuscript record table and access death times::

                    >>> df = proc.manuscript_records  # may be None if not provided
                    >>> if df is not None:
                    ...     print(proc.death_times)

    """

    _NEWICK_LENGTH_EPSILON = 1e-12
    __hash__ = None  # pyright: ignore[reportAssignmentType]  # intentionally unhashable: __eq__ is defined without __hash__

    def __init__(
        self,
        *,
        creation_times: list[float],
        epoch_end_times: NumericInput,
        start_time: float = 0.0,
        initial_population: int = 1,
        birth_rates: NumericInput | None = None,
        death_rates: NumericInput | None = None,
        manuscript_records: pd.DataFrame | None = None,
        spine_ages: NumericInput | None = None,
        non_spine_ages: NumericInput | None = None,
    ) -> None:
        """Initialize birth-death process with observation data.

        Args:
            creation_times: List of times when manuscripts were created
            epoch_end_times: Scalar or vector of epoch end times
            start_time: Start time of the observation period
            initial_population: Initial population size (must be a positive integer)
            birth_rates: Manuscript creation-rate vector (λ) - optional
            death_rates: Manuscript loss-rate vector (μ) - optional

        """
        self.creation_times: list[float] = creation_times.copy()
        self._epoch_end_times: jax.Array = self._normalize_epoch_end_times(
            epoch_end_times,
        )
        self.start_time: float = self._normalize_start_time(start_time)
        self.initial_population: int = self._normalize_initial_population(
            initial_population,
        )
        self._validate_start_time_window()

        # Parameters can be provided at initialization or fitted later
        self._birth_rates: jax.Array | None = (
            self._normalize_rate_vector(birth_rates, label="birth")
            if birth_rates is not None
            else None
        )
        self._death_rates: jax.Array | None = (
            self._normalize_rate_vector(death_rates, label="death")
            if death_rates is not None
            else None
        )

        self._validate_rate_lengths()

        self._validate_creation_times()
        self._manuscript_ages: jax.Array = self._compute_manuscript_ages()
        self._manuscript_records: pd.DataFrame | None = (
            self._normalize_manuscript_records(
                manuscript_records,
            )
        )
        self._spine_ages: jax.Array | None = (
            self._normalize_optional_age_vector(spine_ages)
            if spine_ages is not None
            else None
        )
        self._non_spine_ages: jax.Array | None = (
            self._normalize_optional_age_vector(non_spine_ages)
            if non_spine_ages is not None
            else None
        )

        # Set fitted status based on whether parameters were provided
        self.fitted: bool = (
            self._birth_rates is not None and self._death_rates is not None
        )
        self.fit_results: dict[str, Any] | None = None

    @classmethod
    def from_manuscript_records(
        cls,
        manuscripts: pd.DataFrame,
        *,
        epoch_end_times: NumericInput,
        start_time: float = 0.0,
        initial_population: int = 1,
        birth_rates: NumericInput | None = None,
        death_rates: NumericInput | None = None,
    ) -> BirthDeathProcess:
        """Build a process from a full manuscript event table."""
        required_columns = {
            "manuscript_id",
            "parent_id",
            "creation_time",
            "death_time",
            "is_spine",
            "survived",
            "age",
            "lifespan",
        }
        missing_columns = required_columns.difference(manuscripts.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            msg = f"manuscripts is missing required columns: {missing_text}"
            raise ValueError(msg)

        observed = manuscripts[manuscripts["survived"]].copy()
        process = cls(
            creation_times=observed["creation_time"].astype(float).tolist(),
            epoch_end_times=epoch_end_times,
            start_time=start_time,
            initial_population=initial_population,
            birth_rates=birth_rates,
            death_rates=death_rates,
            manuscript_records=manuscripts.copy(),
        )
        return process

    @classmethod
    def from_age_samples(
        cls,
        *,
        spine_ages: list[float],
        non_spine_ages: list[float],
        epoch_end_times: NumericInput,
        start_time: float = 0.0,
        initial_population: int = 1,
        birth_rates: NumericInput | None = None,
        death_rates: NumericInput | None = None,
    ) -> BirthDeathProcess:
        """Build a process from sampled spine and non-spine manuscript ages."""
        epoch_end_times_arr = ensure_1d_tensor(epoch_end_times)
        max_time = float(epoch_end_times_arr[-1])
        creation_times = [
            max_time - float(age) for age in [*spine_ages, *non_spine_ages]
        ]
        process = cls(
            creation_times=creation_times,
            epoch_end_times=epoch_end_times,
            start_time=start_time,
            initial_population=initial_population,
            birth_rates=birth_rates,
            death_rates=death_rates,
            spine_ages=spine_ages,
            non_spine_ages=non_spine_ages,
        )
        return process

    @property
    def epoch_end_times(self) -> jax.Array:
        """Get epoch end times as a vector."""
        return self._epoch_end_times

    @property
    def adjusted_epoch_end_times(self) -> jax.Array:
        """Get epoch end times shifted by start_time."""
        return self._epoch_end_times - self.start_time

    @property
    def n_epochs(self) -> int:
        """Get number of epochs."""
        return len(self._epoch_end_times)

    @property
    def birth_rates(self) -> jax.Array | None:
        """Get birth-rate vector (one value per epoch)."""
        return self._birth_rates

    @birth_rates.setter
    def birth_rates(self, value: NumericInput | None) -> None:
        """Set birth-rate vector."""
        if value is None:
            self._birth_rates = None
            return
        self._birth_rates = self._normalize_rate_vector(value, label="birth")
        self._validate_rate_lengths()

    @property
    def death_rates(self) -> jax.Array | None:
        """Get death-rate vector (one value per epoch)."""
        return self._death_rates

    @death_rates.setter
    def death_rates(self, value: NumericInput | None) -> None:
        """Set death-rate vector."""
        if value is None:
            self._death_rates = None
            return
        self._death_rates = self._normalize_rate_vector(value, label="death")
        self._validate_rate_lengths()

    def _normalize_epoch_end_times(
        self,
        times: NumericInput,
    ) -> jax.Array:
        """Normalize epoch end times into a strictly increasing vector."""
        config = ValidationConfig(require_times_positive=False)
        times_input = ensure_1d_tensor(times)
        (times_arr,) = normalize_and_validate_inputs(
            times=times_input,
            config=config,
        )
        return times_arr

    def _normalize_start_time(self, start_time: float) -> float:
        """Normalize and validate start time."""
        start_time_value = float(start_time)
        if not np.isfinite(start_time_value):
            msg = "start_time must be finite"
            raise ValueError(msg)

        return start_time_value

    def _normalize_initial_population(self, initial_population: int) -> int:
        """Normalize and validate initial population size."""
        if isinstance(initial_population, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]  # public API boundary: reject float/bool at runtime despite int annotation
            initial_population, int
        ):
            msg = "initial_population must be an integer"
            raise TypeError(msg)

        if initial_population <= 0:
            msg = "initial_population must be a positive integer"
            raise ValueError(msg)

        return initial_population

    def _normalize_optional_age_vector(
        self,
        ages: NumericInput,
    ) -> jax.Array:
        age_arr = ensure_1d_tensor(ages)
        if not bool(jnp.all(jnp.isfinite(age_arr))):
            msg = "age values must be finite"
            raise ValueError(msg)
        if bool(jnp.any(age_arr < 0)):
            msg = "age values must be non-negative"
            raise ValueError(msg)
        return age_arr

    def _normalize_manuscript_records(
        self,
        manuscripts: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if manuscripts is None:
            return None
        records = manuscripts.copy()
        if "death_time" in records.columns:
            records["death_time"] = records["death_time"].astype(float)
        if "age" in records.columns:
            records["age"] = records["age"].astype(float)
        if "lifespan" in records.columns:
            records["lifespan"] = records["lifespan"].astype(float)
        if "parent_id" in records.columns:
            records["parent_id"] = records["parent_id"].astype("Int64")
        if "manuscript_id" in records.columns:
            records["manuscript_id"] = records["manuscript_id"].astype(int)
        self._validate_genealogy_birth_death_order(records)
        return records

    @staticmethod
    def _validate_genealogy_birth_death_order(records: pd.DataFrame) -> None:
        """Validate that each manuscript is born during its parent's lifetime."""
        required_columns = {"manuscript_id", "parent_id", "creation_time"}
        if not required_columns.issubset(records.columns):
            return

        creation_by_id = dict(
            zip(records["manuscript_id"], records["creation_time"], strict=True),
        )
        death_by_id = (
            dict(zip(records["manuscript_id"], records["death_time"], strict=True))
            if "death_time" in records.columns
            else {}
        )

        for manuscript_id, parent_id, child_birth in zip(
            records["manuscript_id"],
            records["parent_id"],
            records["creation_time"],
            strict=True,
        ):
            if pd.isna(parent_id):
                continue
            parent_id = int(parent_id)
            if parent_id not in creation_by_id:
                continue

            parent_birth = creation_by_id[parent_id]
            if child_birth <= parent_birth:
                msg = (
                    f"manuscript {manuscript_id} has creation_time {child_birth} "
                    f"that is not after parent {parent_id}'s creation_time "
                    f"{parent_birth}"
                )
                raise ValueError(msg)

            parent_death = death_by_id.get(parent_id)
            if (
                parent_death is not None
                and not pd.isna(parent_death)
                and child_birth >= parent_death
            ):
                msg = (
                    f"manuscript {manuscript_id} has creation_time {child_birth} "
                    f"that is not before parent {parent_id}'s death_time "
                    f"{parent_death}"
                )
                raise ValueError(msg)

    def _normalize_rate_vector(
        self,
        rates: NumericInput,
        *,
        label: RateLabel,
    ) -> jax.Array:
        """Normalize a scalar/vector rate input into a vector."""
        rate_input = ensure_1d_tensor(rates)

        if label == "birth":
            (rates_arr,) = normalize_and_validate_inputs(birth_rates=rate_input)
        else:
            (rates_arr,) = normalize_and_validate_inputs(death_rates=rate_input)
        return rates_arr

    def _latest_epoch_end_time(self) -> jax.Array:
        """Get the latest epoch end time as an array scalar."""
        return self._epoch_end_times[-1]

    def _validate_start_time_window(self) -> None:
        """Validate start-time relative to epoch end times."""
        latest_epoch_end_time = float(self._latest_epoch_end_time())
        if self.start_time > latest_epoch_end_time:
            msg = "start_time must be less than or equal to latest epoch_end_time"
            raise ValueError(msg)

    def _validate_start_time_against_epoch_end_time(
        self,
        latest_epoch_end_time: float,
    ) -> None:
        """Validate start-time against a candidate latest epoch end time."""
        if self.start_time > latest_epoch_end_time:
            msg = "start_time must be less than or equal to latest epoch_end_time"
            raise ValueError(msg)

    def _validate_and_expand_rate(
        self,
        rate: jax.Array | None,
        *,
        name: str,
    ) -> jax.Array | None:
        """Validate a rate vector against epoch count, expanding scalar rates."""
        if rate is None:
            return None

        if rate.size == 1 and self.n_epochs > 1:
            return jnp.tile(rate, self.n_epochs)

        if len(rate) != self.n_epochs:
            msg = f"{name} length must match number of epoch end times"
            raise ValueError(msg)

        return rate

    def _validate_rate_lengths(self) -> None:
        """Validate that rate vectors match epoch vector length when provided."""
        self._birth_rates = self._validate_and_expand_rate(
            self._birth_rates,
            name="birth_rates",
        )
        self._death_rates = self._validate_and_expand_rate(
            self._death_rates,
            name="death_rates",
        )

    def _validate_creation_times(self) -> None:
        """Validate manuscripts fall within [start_time, latest epoch end time]."""
        if not self.creation_times:
            return

        creation_times_arr = ensure_1d_tensor(self.creation_times)
        if not bool(jnp.all(jnp.isfinite(creation_times_arr))):
            msg = "creation_times must be finite"
            raise ValueError(msg)

        invalid_before_start = [
            ct for ct in self.creation_times if ct < self.start_time
        ]
        if invalid_before_start:
            msg = "creation_times must be greater than or equal to start_time"
            raise ValueError(msg)

        latest_epoch_end_time = float(self._latest_epoch_end_time())
        invalid_after_epoch = [
            ct for ct in self.creation_times if ct > latest_epoch_end_time
        ]
        if invalid_after_epoch:
            msg = "creation_times must be less than or equal to latest epoch_end_time"
            raise ValueError(msg)

    def _compute_manuscript_ages(self) -> jax.Array:
        """Compute manuscript ages from creation times and latest epoch end time."""
        latest_epoch_end_time = self._latest_epoch_end_time()
        if self.creation_times:
            return latest_epoch_end_time - jnp.array(
                self.creation_times,
                dtype=jnp.float64,
            )
        return jnp.array([], dtype=jnp.float64)

    def _recompute_manuscript_ages(self) -> None:
        """Recompute and cache manuscript ages."""
        self._manuscript_ages = self._compute_manuscript_ages()

    @property
    def manuscript_ages(self) -> jax.Array:
        """Get cached manuscript ages from creation times and latest epoch end time."""
        self._recompute_manuscript_ages()
        return self._manuscript_ages

    @property
    def manuscript_records(self) -> pd.DataFrame | None:
        """Get the full manuscript record table when available."""
        if self._manuscript_records is None:
            return None
        return self._manuscript_records.copy()

    @property
    def spine_ages(self) -> jax.Array | None:
        """Get sampled spine ages when the instance originates from that simulator."""
        return self._spine_ages

    @property
    def non_spine_ages(self) -> jax.Array | None:
        """Get sampled non-spine ages when instance originates from the simulator."""
        return self._non_spine_ages

    @property
    def death_times(self) -> jax.Array | None:
        """Get manuscript death times when available."""
        if (
            self._manuscript_records is None
            or "death_time" not in self._manuscript_records
        ):
            return None
        return jnp.array(
            self._manuscript_records["death_time"].to_numpy(dtype=np.float64),
            dtype=jnp.float64,
        )

    @property
    def all_manuscript_ages(self) -> jax.Array | None:
        """Get manuscript ages for all records when a record table exists."""
        if self._manuscript_records is None or "age" not in self._manuscript_records:
            return None
        return jnp.array(
            self._manuscript_records["age"].to_numpy(dtype=np.float64),
            dtype=jnp.float64,
        )

    @property
    def extinct_manuscripts(self) -> pd.DataFrame | None:
        """Get the extinct-manuscript subset when a record table exists."""
        if self._manuscript_records is None:
            return None
        return self._manuscript_records[~self._manuscript_records["survived"]].copy()

    @property
    def surviving_manuscripts(self) -> pd.DataFrame | None:
        """Get the surviving-manuscript subset when a record table exists."""
        if self._manuscript_records is None:
            return None
        return self._manuscript_records[self._manuscript_records["survived"]].copy()

    def genealogy_dfs_order(self) -> pd.DataFrame:
        """Return the preorder DFS ordering for the full genealogy tree."""
        if self._manuscript_records is None:
            msg = (
                "genealogy_dfs_order requires manuscript_records with "
                "manuscript_id and parent_id"
            )
            raise ValueError(msg)

        from inf_bp_texts.visualization.genealogy import (
            genealogy_dfs_order as _genealogy_dfs_order,
        )

        return _genealogy_dfs_order(
            self._manuscript_records,
            start_time=self.start_time,
            simulation_end_time=float(self._latest_epoch_end_time()),
        )

    def plot_genealogy_tree(
        self,
        *,
        config: GenealogyPlotConfig | None = None,
    ) -> alt.LayerChart:
        """Return an Altair chart showing manuscript lifetimes and genealogy."""
        from inf_bp_texts.visualization.genealogy import (
            plot_genealogy_tree as _plot_genealogy_tree,
        )

        if self._manuscript_records is None:
            msg = (
                "plot_genealogy_tree requires manuscript_records with "
                "manuscript_id and parent_id"
            )
            raise ValueError(msg)

        return _plot_genealogy_tree(
            self._manuscript_records,
            start_time=self.start_time,
            simulation_end_time=float(self._latest_epoch_end_time()),
            config=config,
        )

    @classmethod
    def _format_newick_length(cls, length: float) -> str:
        """Format a branch length for deterministic Newick output."""
        length_value = float(length)
        if abs(length_value) < cls._NEWICK_LENGTH_EPSILON:
            length_value = 0.0
        return repr(length_value)

    def _to_newick_fragment(
        self,
        manuscript_id: int,
        *,
        records_by_id: dict[int, dict[Hashable, Any]],
        child_map: dict[int, list[int]],
        simulation_end_time: float,
    ) -> str:
        """Serialize one lineage into a nested Newick fragment."""
        record = records_by_id[manuscript_id]
        birth_time = float(record["creation_time"])
        death_raw = record.get("death_time")
        death_time = (
            float(death_raw)
            if death_raw is not None and not pd.isna(death_raw)
            else float(simulation_end_time)
        )

        children = child_map.get(manuscript_id, [])
        if not children:
            branch_length = self._format_newick_length(death_time - birth_time)
            return f"({manuscript_id}:{branch_length}):0"

        last_child_id = children[-1]
        last_child_birth_time = float(records_by_id[last_child_id]["creation_time"])
        branch_length = self._format_newick_length(
            death_time - last_child_birth_time,
        )
        current = f"({manuscript_id}:{branch_length}):0"

        for index in range(len(children) - 1, -1, -1):
            child_id = children[index]
            child_birth_time = float(records_by_id[child_id]["creation_time"])
            next_time = (
                death_time
                if index == len(children) - 1
                else float(records_by_id[children[index + 1]]["creation_time"])
            )
            child_fragment = self._to_newick_fragment(
                child_id,
                records_by_id=records_by_id,
                child_map=child_map,
                simulation_end_time=simulation_end_time,
            )
            current = (
                f"({child_fragment},{current})"
                f":{self._format_newick_length(next_time - child_birth_time)}"
            )

        return f"({current}):0"

    def to_newick(self) -> str:
        """Export the full genealogy as a strict Newick tree string."""
        if self._manuscript_records is None:
            msg = (
                "to_newick requires manuscript_records with manuscript_id and parent_id"
            )
            raise ValueError(msg)

        # Reuse the genealogy validator to enforce the rooted-tree contract.
        ordered_records = self.genealogy_dfs_order()

        records_by_id = {
            int(row["manuscript_id"]): row.to_dict()
            for _, row in ordered_records.iterrows()
        }

        child_map: dict[int, list[int]] = {}
        for manuscript_id, record in records_by_id.items():
            parent_raw = record.get("parent_id")
            if pd.isna(parent_raw):
                continue
            parent_id = int(parent_raw)
            child_map.setdefault(parent_id, []).append(manuscript_id)

        for child_ids in child_map.values():
            child_ids.sort(
                key=lambda child_id: (
                    float(records_by_id[child_id]["creation_time"]),
                    child_id,
                ),
            )

        root_rows = ordered_records[ordered_records["parent_id"].isna()]
        root_id = int(root_rows.iloc[0]["manuscript_id"])
        root_birth_time = float(root_rows.iloc[0]["creation_time"])
        simulation_end_time = float(self._latest_epoch_end_time())

        root_fragment = self._to_newick_fragment(
            root_id,
            records_by_id=records_by_id,
            child_map=child_map,
            simulation_end_time=simulation_end_time,
        )
        root_branch_length = self._format_newick_length(
            root_birth_time - self.start_time,
        )
        return f"({root_fragment}):{root_branch_length};"

    def equals(self, other: Any) -> bool:
        """Test equality against another BirthDeathProcess instance."""
        if not isinstance(other, BirthDeathProcess):
            return False
        if (
            self._manuscript_records is not None
            and other._manuscript_records is not None
        ):
            return self._manuscript_records.equals(other._manuscript_records)

        def _arrays_equal(a: jax.Array | None, b: jax.Array | None) -> bool:
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return bool(jnp.array_equal(a, b))

        return (
            self.creation_times == other.creation_times
            and _arrays_equal(self._epoch_end_times, other._epoch_end_times)
            and _arrays_equal(self._birth_rates, other._birth_rates)
            and _arrays_equal(self._death_rates, other._death_rates)
        )

    def __eq__(self, other: object) -> bool:
        """Test equality against another BirthDeathProcess instance."""
        if isinstance(other, BirthDeathProcess):
            return self.equals(other)
        return NotImplemented

    def __len__(self) -> int:
        """Return the number of manuscript records when available."""
        if self._manuscript_records is not None:
            return len(self._manuscript_records)
        return len(self.manuscript_ages)

    @property
    def n_manuscripts(self) -> int:
        """Get the number of manuscripts in the process."""
        return len(self.manuscript_ages)

    def add_manuscript(self, creation_time: float) -> None:
        """Add a manuscript with given creation time to the process.

        Args:
            creation_time: Time when the manuscript was created

        """
        if creation_time < self.start_time:
            msg = "creation_time must be greater than or equal to start_time"
            raise ValueError(msg)

        latest_epoch_end_time = float(self._latest_epoch_end_time())
        if creation_time > latest_epoch_end_time:
            msg = "creation_time must be less than or equal to latest epoch_end_time"
            raise ValueError(msg)

        self.creation_times.append(creation_time)
        self._recompute_manuscript_ages()

    def set_epoch_end_times(
        self,
        epoch_end_times: NumericInput,
    ) -> None:
        """Set epoch end times for the process.

        Args:
            epoch_end_times: Scalar end time or vector of epoch end times

        """
        new_epoch_end_times = self._normalize_epoch_end_times(epoch_end_times)
        previous_epoch_end_times = self._epoch_end_times

        try:
            self._epoch_end_times = new_epoch_end_times
            latest_epoch_end_time = float(self._epoch_end_times[-1])
            self._validate_start_time_against_epoch_end_time(latest_epoch_end_time)
            self._validate_rate_lengths()
            self._validate_creation_times()
            self._recompute_manuscript_ages()
        except Exception:
            self._epoch_end_times = previous_epoch_end_times
            raise

    def log_likelihood(self, unbias: bool = True) -> jax.Array:
        """Compute log-likelihood directly from the current model state.

        Raises
        ------
        ValueError
            If birth/death rates are not set on the instance.

        """
        if self._birth_rates is None or self._death_rates is None:
            msg = "birth_rates and death_rates must be set before likelihood calls"
            raise ValueError(msg)

        from inf_bp_texts.inference.likelihood import log_likelihood as _log_likelihood

        return _log_likelihood(
            birth_rates=self._birth_rates,
            death_rates=self._death_rates,
            ages=self.manuscript_ages,
            times=self.adjusted_epoch_end_times,
            unbias=unbias,
        )

    def negative_log_likelihood(self, unbias: bool = True) -> jax.Array:
        """Compute negative log-likelihood directly from the current model state."""
        if self._birth_rates is None or self._death_rates is None:
            msg = "birth_rates and death_rates must be set before likelihood calls"
            raise ValueError(msg)

        from inf_bp_texts.inference.likelihood import (
            negative_log_likelihood as _negative_log_likelihood,
        )

        return _negative_log_likelihood(
            birth_rates=self._birth_rates,
            death_rates=self._death_rates,
            ages=self.manuscript_ages,
            times=self.adjusted_epoch_end_times,
            unbias=unbias,
        )

    def update_parameters(
        self,
        birth_rates: NumericInput,
        death_rates: NumericInput,
        fit_results: dict[str, Any] | None = None,
    ) -> None:
        """Update the model parameters after fitting.

        Use the fitting module directly to obtain parameters, then call this method
        to update the model state.

        Args:
            birth_rates: Fitted birth-rate vector (λ), one value per epoch
            death_rates: Fitted death-rate vector (μ), one value per epoch
            fit_results: Optional fit results dictionary from fitting module

        """
        self._birth_rates = self._normalize_rate_vector(
            birth_rates,
            label="birth",
        )
        self._death_rates = self._normalize_rate_vector(
            death_rates,
            label="death",
        )
        self._validate_rate_lengths()
        self.fitted = True
        self.fit_results = fit_results
