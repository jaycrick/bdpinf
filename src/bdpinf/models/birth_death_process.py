"""Core Birth-Death Process model for manuscript survival and creation analysis.

This module contains the main BirthDeathProcess class with basic data management,
initialization, and core functionality. Complex mathematical computations and
fitting procedures are handled by separate modules.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
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
        >>> ages = model.get_manuscript_ages()
        >>> print(ages)
        [400. 300. 200.]

        Fitting parameters using the inference module:

        >>> from inf_bp_texts.inference.fitting import fit_birth_death_process
        >>> import numpy as np
        >>> ages = np.array([100, 150, 200])
        >>> result = fit_birth_death_process(ages, observation_time=250)
        >>> if result['success']:
        ...     print(f"Birth rate: {result['birth_rate']:.3f}")
        ...     print(f"Death rate: {result['death_rate']:.3f}")

    """

    def __init__(
        self,
        creation_times: list[float],
        epoch_end_times: float | list[float] | NDArray[np.float64],
        start_time: float | None = None,
        birth_rate: float | list[float] | NDArray[np.float64] | None = None,
        death_rate: float | list[float] | NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize birth-death process with observation data.

        Args:
            creation_times: List of times when manuscripts were created
            epoch_end_times: Scalar or vector of epoch end times
            start_time: Start time of the observation period (optional, defaults to 0)
            birth_rate: Manuscript creation rate (λ) - optional, can be fitted later
            death_rate: Manuscript loss rate (μ) - optional, can be fitted later

        """
        self.creation_times: list[float] = creation_times.copy()
        self.start_time: float = start_time if start_time is not None else 0.0
        self._epoch_end_times: NDArray[np.float64] = self._normalize_epoch_end_times(
            epoch_end_times,
        )

        # Parameters can be provided at initialization or fitted later
        self.birth_rate: NDArray[np.float64] | None = (
            self._normalize_rate_vector(birth_rate) if birth_rate is not None else None
        )
        self.death_rate: NDArray[np.float64] | None = (
            self._normalize_rate_vector(death_rate) if death_rate is not None else None
        )

        self._validate_rate_lengths()

        # Remove manuscripts created after last epoch end time
        self._filter_creation_times()

        # Set fitted status based on whether parameters were provided
        self.fitted: bool = self.birth_rate is not None and self.death_rate is not None
        self.fit_results: dict[str, Any] | None = None

    @property
    def epoch_end_times(self) -> NDArray[np.float64]:
        """Get epoch end times as a vector."""
        return self._epoch_end_times.copy()

    @property
    def n_epochs(self) -> int:
        """Get number of epochs."""
        return len(self._epoch_end_times)

    @property
    def birth_rates(self) -> NDArray[np.float64] | None:
        """Get birth-rate vector (one value per epoch)."""
        if self.birth_rate is None:
            return None
        return self.birth_rate.copy()

    @property
    def death_rates(self) -> NDArray[np.float64] | None:
        """Get death-rate vector (one value per epoch)."""
        if self.death_rate is None:
            return None
        return self.death_rate.copy()

    def _normalize_epoch_end_times(
        self,
        times: float | list[float] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Normalize epoch end times into a strictly increasing vector."""
        if np.isscalar(times):
            array = np.array([times], dtype=np.float64)
        else:
            array = np.asarray(times, dtype=np.float64)

        if array.ndim != 1 or len(array) == 0:
            msg = "epoch_end_times must be a non-empty scalar or 1D vector"
            raise ValueError(msg)
        if not np.all(np.isfinite(array)):
            msg = "epoch_end_times values must be finite"
            raise ValueError(msg)
        if np.any(np.diff(array) <= 0):
            msg = "epoch_end_times values must be strictly increasing"
            raise ValueError(msg)
        return array

    def _normalize_rate_vector(
        self,
        rates: float | list[float] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Normalize a scalar/vector rate input into a vector."""
        if np.isscalar(rates):
            return np.array([rates], dtype=np.float64)

        array = np.asarray(rates, dtype=np.float64)
        if array.ndim != 1 or len(array) == 0:
            msg = "Rate inputs must be non-empty scalars or 1D vectors"
            raise ValueError(msg)
        return array

    def _validate_rate_lengths(self) -> None:
        """Validate that rate vectors match epoch vector length when provided."""
        n_epochs = self.n_epochs

        if self.birth_rate is not None:
            if len(self.birth_rate) == 1 and n_epochs > 1:
                self.birth_rate = np.full(
                    n_epochs,
                    self.birth_rate[0],
                    dtype=np.float64,
                )
            elif len(self.birth_rate) != n_epochs:
                msg = "birth_rate length must match number of epoch end times"
                raise ValueError(msg)

        if self.death_rate is not None:
            if len(self.death_rate) == 1 and n_epochs > 1:
                self.death_rate = np.full(
                    n_epochs,
                    self.death_rate[0],
                    dtype=np.float64,
                )
            elif len(self.death_rate) != n_epochs:
                msg = "death_rate length must match number of epoch end times"
                raise ValueError(msg)

    def _filter_creation_times(self) -> None:
        """Remove manuscripts created after the last epoch end time."""
        if not self.creation_times:
            return

        latest_epoch_end_time = float(self._epoch_end_times[-1])
        self.creation_times = [
            ct for ct in self.creation_times if ct <= latest_epoch_end_time
        ]

    def get_manuscript_ages(self) -> NDArray[np.float64]:
        """Calculate manuscript ages from creation times and last epoch end time.

        Returns:
            Array of manuscript ages (latest_epoch_end_time - creation_time)

        """
        latest_epoch_end_time = float(self._epoch_end_times[-1])
        if self.creation_times:
            return np.array(
                [latest_epoch_end_time - ct for ct in self.creation_times],
                dtype=np.float64,
            )
        return np.array([], dtype=np.float64)

    def add_manuscript(self, creation_time: float) -> None:
        """Add a manuscript with given creation time to the process.

        Args:
            creation_time: Time when the manuscript was created

        """
        self.creation_times.append(creation_time)

    def set_epoch_end_times(
        self,
        epoch_end_times: float | list[float] | NDArray[np.float64],
    ) -> None:
        """Set epoch end times for the process.

        Args:
            epoch_end_times: Scalar end time or vector of epoch end times

        """
        self._epoch_end_times = self._normalize_epoch_end_times(epoch_end_times)
        self._validate_rate_lengths()
        self._filter_creation_times()

    def get_process_summary(self) -> dict[str, Any]:
        """Get a summary of the birth-death process state.

        Returns:
            Dictionary with process information

        """
        summary: dict[str, Any] = {
            "n_manuscripts": len(self.creation_times),
            "epoch_end_times": self.epoch_end_times.tolist(),
            "n_epochs": self.n_epochs,
            "fitted": self.fitted,
        }

        # Include parameters if they've been set
        if self.birth_rate is not None:
            summary["birth_rate"] = float(self.birth_rate[-1])
            summary["birth_rates"] = self.birth_rate.tolist()
        if self.death_rate is not None:
            summary["death_rate"] = float(self.death_rate[-1])
            summary["death_rates"] = self.death_rate.tolist()

        ages = self.get_manuscript_ages()
        if len(ages) > 0:
            summary.update(
                {
                    "min_age": float(ages.min()),
                    "max_age": float(ages.max()),
                    "mean_age": float(ages.mean()),
                },
            )

        return summary

    def validate_parameters(self) -> bool:
        """Validate that model parameters are set and reasonable.

        Returns:
            True if parameters are valid, False otherwise

        """
        if self.birth_rate is None or self.death_rate is None:
            return False

        if np.any(self.birth_rate <= 0) or np.any(self.death_rate <= 0):
            return False

        if (
            len(self.birth_rate) != self.n_epochs
            or len(self.death_rate) != self.n_epochs
        ):
            return False

        return bool(
            np.all(np.isfinite(self.birth_rate))
            and np.all(np.isfinite(self.death_rate)),
        )

    def update_parameters(
        self,
        birth_rate: float | list[float] | NDArray[np.float64],
        death_rate: float | list[float] | NDArray[np.float64],
        fit_results: dict[str, Any] | None = None,
    ) -> None:
        """Update the model parameters after fitting.

        Use the fitting module directly to obtain parameters, then call this method
        to update the model state.

        Args:
            birth_rate: Fitted birth-rate vector (λ), one value per epoch
            death_rate: Fitted death-rate vector (μ), one value per epoch
            fit_results: Optional fit results dictionary from fitting module

        """
        self.birth_rate = self._normalize_rate_vector(birth_rate)
        self.death_rate = self._normalize_rate_vector(death_rate)
        self._validate_rate_lengths()
        self.fitted = True
        self.fit_results = fit_results

    def reset_parameters(self) -> None:
        """Reset fitted parameters and status."""
        self.birth_rate = None
        self.death_rate = None
        self.fitted = False
        self.fit_results = None
