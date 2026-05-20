"""Core Birth-Death Process model for manuscript survival and creation analysis.

This module contains the main BirthDeathProcess class with basic data management,
initialization, and core functionality. Complex mathematical computations and
fitting procedures are handled by separate modules.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from numpy.typing import NDArray

from inf_bp_texts.inference.likelihood import log_likelihood as _log_likelihood
from inf_bp_texts.inference.likelihood import (
    negative_log_likelihood as _negative_log_likelihood,
)
from inf_bp_texts.models.validation import (
    ValidationConfig,
    normalize_and_validate_inputs,
)
from inf_bp_texts.visualization.genealogy import (
    GenealogyPlotConfig,
)
from inf_bp_texts.visualization.genealogy import (
    genealogy_dfs_order as _genealogy_dfs_order,
)
from inf_bp_texts.visualization.genealogy import (
    plot_genealogy_tree as _plot_genealogy_tree,
)

NumericInput = float | list[float] | NDArray[np.float64] | torch.Tensor
RateLabel = Literal["birth", "death"]


class BirthDeathProcess:
    r"""Birth-death process model for manuscript survival and creation.

    This class models manuscript data as a birth-death process where:
    - Birth rate (λ): Rate of manuscript creation
    - Death rate (μ): Rate of manuscript loss/destruction
    - Observed data: Number of surviving manuscripts and their ages

    The likelihood function implements the proper birth-death process:
    1. P_n(T) = probability of n individuals at time T
    2. f(t) = age density for each manuscript at age t
    3. Total likelihood = $P_n(T) * \prod f(t_i)$ for all observed ages

    Mathematical formulations are handled by the mathematical_functions module.

    Examples:
        Basic usage with manuscript creation times:

        >>> from inf_bp_texts.models import BirthDeathProcess
        >>> model = BirthDeathProcess([100, 200, 300], epoch_end_times=500)
        >>> ages = model.manuscript_ages
        >>> print(ages)
        [400. 300. 200.]

        Fitting parameters using the inference module:

        >>> from inf_bp_texts.inference.fitting import fit_birth_death_process
        >>> import numpy as np
        >>> ages = np.array([100, 150, 200])
        >>> result = fit_birth_death_process(ages, observation_time=250)
        >>> if result['success']:
        ...     print(f"Birth rates: {result['birth_rates']}")
        ...     print(f"Death rates: {result['death_rates']}")

    """

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
        self._epoch_end_times: torch.Tensor = self._normalize_epoch_end_times(
            epoch_end_times,
        )
        self.start_time: float = self._normalize_start_time(start_time)
        self.initial_population: int = self._normalize_initial_population(
            initial_population,
        )
        self._validate_start_time_window()

        # Parameters can be provided at initialization or fitted later
        self._birth_rates: torch.Tensor | None = (
            self._normalize_rate_vector(birth_rates, label="birth")
            if birth_rates is not None
            else None
        )
        self._death_rates: torch.Tensor | None = (
            self._normalize_rate_vector(death_rates, label="death")
            if death_rates is not None
            else None
        )

        self._validate_rate_lengths()

        self._validate_creation_times()
        self._manuscript_ages: torch.Tensor = self._compute_manuscript_ages()
        self._manuscript_records: pd.DataFrame | None = (
            self._normalize_manuscript_records(
                manuscript_records,
            )
        )
        self._spine_ages: torch.Tensor | None = (
            self._normalize_optional_age_vector(spine_ages)
            if spine_ages is not None
            else None
        )
        self._non_spine_ages: torch.Tensor | None = (
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
    ) -> "BirthDeathProcess":
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
    ) -> "BirthDeathProcess":
        """Build a process from sampled spine and non-spine manuscript ages."""
        epoch_end_times_tensor = cls._to_static_float64_tensor(epoch_end_times)
        max_time = float(epoch_end_times_tensor.reshape(-1)[-1].item())
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
    def epoch_end_times(self) -> torch.Tensor:
        """Get epoch end times as a vector."""
        return self._epoch_end_times.clone()

    @property
    def adjusted_epoch_end_times(self) -> torch.Tensor:
        """Get epoch end times shifted by start_time."""
        return self._epoch_end_times - self.start_time

    @property
    def n_epochs(self) -> int:
        """Get number of epochs."""
        return len(self._epoch_end_times)

    @property
    def birth_rates(self) -> torch.Tensor | None:
        """Get birth-rate vector (one value per epoch)."""
        if self._birth_rates is None:
            return None
        return self._birth_rates.clone()

    @birth_rates.setter
    def birth_rates(self, value: NumericInput | None) -> None:
        """Set birth-rate vector."""
        if value is None:
            self._birth_rates = None
            return
        self._birth_rates = self._normalize_rate_vector(value, label="birth")
        self._validate_rate_lengths()

    @property
    def death_rates(self) -> torch.Tensor | None:
        """Get death-rate vector (one value per epoch)."""
        if self._death_rates is None:
            return None
        return self._death_rates.clone()

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
    ) -> torch.Tensor:
        """Normalize epoch end times into a strictly increasing vector."""
        config = ValidationConfig(require_times_positive=False)
        times_input = self._to_float64_tensor(times)
        (times_tensor,) = normalize_and_validate_inputs(
            times=times_input,
            config=config,
        )
        return times_tensor.to(dtype=torch.float64)

    def _normalize_start_time(self, start_time: float) -> float:
        """Normalize and validate start time."""
        start_time_value = float(start_time)
        if not np.isfinite(start_time_value):
            msg = "start_time must be finite"
            raise ValueError(msg)

        return start_time_value

    def _normalize_initial_population(self, initial_population: int) -> int:
        """Normalize and validate initial population size."""
        if isinstance(initial_population, bool) or not isinstance(
            initial_population,
            int,
        ):
            msg = "initial_population must be an integer"
            raise TypeError(msg)

        if initial_population <= 0:
            msg = "initial_population must be a positive integer"
            raise ValueError(msg)

        return initial_population

    def _to_float64_tensor(self, values: NumericInput) -> torch.Tensor:
        """Convert numeric input to a float64 tensor."""
        if isinstance(values, torch.Tensor):
            return values.to(dtype=torch.float64)
        if isinstance(values, np.ndarray):
            return torch.as_tensor(values, dtype=torch.float64)
        if isinstance(values, list):
            return torch.tensor(values, dtype=torch.float64)
        return torch.tensor([values], dtype=torch.float64)

    @staticmethod
    def _to_static_float64_tensor(values: NumericInput) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(dtype=torch.float64)
        if isinstance(values, np.ndarray):
            return torch.as_tensor(values, dtype=torch.float64)
        if isinstance(values, list):
            return torch.tensor(values, dtype=torch.float64)
        return torch.tensor([values], dtype=torch.float64)

    def _normalize_optional_age_vector(
        self,
        ages: NumericInput,
    ) -> torch.Tensor:
        age_tensor = self._to_float64_tensor(ages)
        if not torch.all(torch.isfinite(age_tensor)):
            msg = "age values must be finite"
            raise ValueError(msg)
        if torch.any(age_tensor < 0):
            msg = "age values must be non-negative"
            raise ValueError(msg)
        return age_tensor

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
        return records

    def _normalize_rate_vector(
        self,
        rates: NumericInput,
        *,
        label: RateLabel,
    ) -> torch.Tensor:
        """Normalize a scalar/vector rate input into a vector."""
        rate_input = self._to_float64_tensor(rates)

        if label == "birth":
            (rates_tensor,) = normalize_and_validate_inputs(birth_rates=rate_input)
        else:
            (rates_tensor,) = normalize_and_validate_inputs(death_rates=rate_input)
        return rates_tensor.to(dtype=torch.float64)

    def _latest_epoch_end_time(self) -> torch.Tensor:
        """Get the latest epoch end time as a tensor scalar."""
        return self._epoch_end_times[-1]

    def _validate_start_time_window(self) -> None:
        """Validate start-time relative to epoch end times."""
        latest_epoch_end_time = float(self._latest_epoch_end_time().item())
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
        rate: torch.Tensor | None,
        *,
        name: str,
    ) -> torch.Tensor | None:
        """Validate a rate vector against epoch count, expanding scalar rates."""
        if rate is None:
            return None

        if rate.numel() == 1 and self.n_epochs > 1:
            return rate.repeat(self.n_epochs)

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

        creation_times_tensor = self._to_float64_tensor(self.creation_times)
        if not torch.all(torch.isfinite(creation_times_tensor)):
            msg = "creation_times must be finite"
            raise ValueError(msg)

        invalid_before_start = [
            ct for ct in self.creation_times if ct < self.start_time
        ]
        if invalid_before_start:
            msg = "creation_times must be greater than or equal to start_time"
            raise ValueError(msg)

        latest_epoch_end_time = float(self._latest_epoch_end_time().item())
        invalid_after_epoch = [
            ct for ct in self.creation_times if ct > latest_epoch_end_time
        ]
        if invalid_after_epoch:
            msg = "creation_times must be less than or equal to latest epoch_end_time"
            raise ValueError(msg)

    def _compute_manuscript_ages(self) -> torch.Tensor:
        """Compute manuscript ages from creation times and latest epoch end time."""
        latest_epoch_end_time = self._latest_epoch_end_time()
        if self.creation_times:
            return latest_epoch_end_time - torch.tensor(
                self.creation_times,
                dtype=torch.float64,
            )
        return torch.tensor([], dtype=torch.float64)

    def _recompute_manuscript_ages(self) -> None:
        """Recompute and cache manuscript ages."""
        self._manuscript_ages = self._compute_manuscript_ages()

    @property
    def manuscript_ages(self) -> torch.Tensor:
        """Get cached manuscript ages from creation times and latest epoch end time."""
        self._recompute_manuscript_ages()
        return self._manuscript_ages.clone()

    @property
    def manuscript_records(self) -> pd.DataFrame | None:
        """Get the full manuscript record table when available."""
        if self._manuscript_records is None:
            return None
        return self._manuscript_records.copy()

    @property
    def spine_ages(self) -> torch.Tensor | None:
        """Get sampled spine ages when the instance originates from that simulator."""
        if self._spine_ages is None:
            return None
        return self._spine_ages.clone()

    @property
    def non_spine_ages(self) -> torch.Tensor | None:
        """Get sampled non-spine ages when the instance originates from that simulator."""
        if self._non_spine_ages is None:
            return None
        return self._non_spine_ages.clone()

    @property
    def death_times(self) -> torch.Tensor | None:
        """Get manuscript death times when available."""
        if (
            self._manuscript_records is None
            or "death_time" not in self._manuscript_records
        ):
            return None
        return torch.tensor(
            self._manuscript_records["death_time"].to_numpy(dtype=np.float64),
            dtype=torch.float64,
        )

    @property
    def all_manuscript_ages(self) -> torch.Tensor | None:
        """Get manuscript ages for all records when a record table exists."""
        if self._manuscript_records is None or "age" not in self._manuscript_records:
            return None
        return torch.tensor(
            self._manuscript_records["age"].to_numpy(dtype=np.float64),
            dtype=torch.float64,
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

        return _genealogy_dfs_order(
            self._manuscript_records,
            start_time=self.start_time,
            simulation_end_time=float(self._latest_epoch_end_time().item()),
        )

    def plot_genealogy_tree(
        self,
        *,
        config: GenealogyPlotConfig | None = None,
    ) -> go.Figure:
        """Return a Plotly figure showing manuscript lifetimes and genealogy."""
        if self._manuscript_records is None:
            msg = (
                "plot_genealogy_tree requires manuscript_records with "
                "manuscript_id and parent_id"
            )
            raise ValueError(msg)

        return _plot_genealogy_tree(
            self._manuscript_records,
            start_time=self.start_time,
            simulation_end_time=float(self._latest_epoch_end_time().item()),
            config=config,
        )

    def __getitem__(self, key: Any) -> Any:
        """Provide lightweight compatibility with the prior DataFrame/dict outputs."""
        if isinstance(key, str):
            if key == "spine":
                if self._spine_ages is not None:
                    return self._spine_ages.clone().tolist()
                if (
                    self._manuscript_records is not None
                    and "is_spine" in self._manuscript_records
                ):
                    return (
                        self._manuscript_records.loc[
                            self._manuscript_records["is_spine"],
                            "age",
                        ]
                        .astype(float)
                        .tolist()
                    )
            if key == "non_spine":
                if self._non_spine_ages is not None:
                    return self._non_spine_ages.clone().tolist()
                if (
                    self._manuscript_records is not None
                    and "is_spine" in self._manuscript_records
                ):
                    return (
                        self._manuscript_records.loc[
                            ~self._manuscript_records["is_spine"],
                            "age",
                        ]
                        .astype(float)
                        .tolist()
                    )
            if (
                self._manuscript_records is not None
                and key in self._manuscript_records.columns
            ):
                return self._manuscript_records[key]
            raise KeyError(key)

        if self._manuscript_records is not None:
            return self._manuscript_records[key].copy()

        raise TypeError(
            "BirthDeathProcess only supports string and record-style indexing",
        )

    def equals(self, other: Any) -> bool:
        """Compare simulation objects using their manuscript records when available."""
        if not isinstance(other, BirthDeathProcess):
            return False
        if (
            self._manuscript_records is not None
            and other._manuscript_records is not None
        ):
            return self._manuscript_records.equals(other._manuscript_records)
        return (
            self.creation_times == other.creation_times
            and torch.equal(self.epoch_end_times, other.epoch_end_times)
            and self.birth_rates == other.birth_rates
            and self.death_rates == other.death_rates
        )

    def __eq__(self, other: object) -> bool:
        """Compare against legacy dict outputs or another process instance."""
        if isinstance(other, dict):
            expected_keys = {"spine", "non_spine"}
            if set(other) != expected_keys:
                return False
            return list(self["spine"]) == list(other["spine"]) and list(
                self["non_spine"],
            ) == list(other["non_spine"])
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

        latest_epoch_end_time = float(self._latest_epoch_end_time().item())
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
            latest_epoch_end_time = float(self._epoch_end_times[-1].item())
            self._validate_start_time_against_epoch_end_time(latest_epoch_end_time)
            self._validate_rate_lengths()
            self._validate_creation_times()
            self._recompute_manuscript_ages()
        except Exception:
            self._epoch_end_times = previous_epoch_end_times
            raise

    def log_likelihood(self, unbias: bool = True) -> torch.Tensor:
        """Compute log-likelihood directly from the current model state.

        Raises
        ------
        ValueError
            If birth/death rates are not set on the instance.

        """
        if self._birth_rates is None or self._death_rates is None:
            msg = "birth_rates and death_rates must be set before likelihood calls"
            raise ValueError(msg)

        return _log_likelihood(
            birth_rates=self._birth_rates,
            death_rates=self._death_rates,
            ages=self.manuscript_ages,
            times=self.adjusted_epoch_end_times,
            unbias=unbias,
        )

    def negative_log_likelihood(self, unbias: bool = True) -> torch.Tensor:
        """Compute negative log-likelihood directly from the current model state."""
        if self._birth_rates is None or self._death_rates is None:
            msg = "birth_rates and death_rates must be set before likelihood calls"
            raise ValueError(msg)

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
