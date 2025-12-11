"""
Core Birth-Death Process model for manuscript survival and creation analysis.

This module contains the main BirthDeathProcess class with basic data management,
initialization, and core functionality. Complex mathematical computations and
fitting procedures are handled by separate modules.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


class BirthDeathProcess:
    """
    Birth-death process model for manuscript survival and creation.

    This class models manuscript data as a birth-death process where:
    - Birth rate (λ): Rate of manuscript creation
    - Death rate (μ): Rate of manuscript loss/destruction
    - Observed data: Number of surviving manuscripts and their ages

    The likelihood function implements the proper birth-death process:
    1. P_n(T) = probability of n individuals at time T
    2. f(t) = age density for each manuscript at age t
    3. Total likelihood = P_n(T) × ∏f(t_i) for all observed ages

    Mathematical formulations are handled by the mathematical_functions module.

    Examples:
        Basic usage with manuscript creation times:

        >>> from inf_bp_texts.models import BirthDeathProcess
        >>> model = BirthDeathProcess([100, 200, 300], final_time=500)
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
        creation_times: List[float],
        final_time: float,
        start_time: Optional[float] = None,
        birth_rate: Optional[float] = None,
        death_rate: Optional[float] = None,
    ):
        """
        Initialize birth-death process with observation data.

        Args:
            creation_times: List of times when manuscripts were created
            final_time: Time when the process was terminated/observed
            start_time: Start time of the observation period (optional, defaults to 0)
            birth_rate: Manuscript creation rate (λ) - optional, can be fitted later
            death_rate: Manuscript loss rate (μ) - optional, can be fitted later
        """
        self.creation_times: List[float] = creation_times.copy()
        self.start_time: float = start_time if start_time is not None else 0.0
        self._final_time: float = final_time

        # Parameters can be provided at initialization or fitted later
        self.birth_rate: Optional[np.float64] = (
            np.float64(birth_rate) if birth_rate is not None else None
        )
        self.death_rate: Optional[np.float64] = (
            np.float64(death_rate) if death_rate is not None else None
        )

        # Remove manuscripts created after final_time
        self._filter_creation_times()

        # Set fitted status based on whether parameters were provided
        self.fitted: bool = self.birth_rate is not None and self.death_rate is not None
        self.fit_results: Optional[Dict[str, Any]] = None

    @property
    def final_time(self) -> float:
        """Get the final observation time."""
        return self._final_time

    @final_time.setter
    def final_time(self, value: float) -> None:
        """Set the final observation time and filter creation times."""
        self._final_time = value
        self._filter_creation_times()

    def _filter_creation_times(self) -> None:
        """Remove manuscripts created after final_time with a warning message."""
        if not self.creation_times:
            return

        original_count = len(self.creation_times)
        self.creation_times = [
            ct for ct in self.creation_times if ct <= self._final_time
        ]
        removed_count = original_count - len(self.creation_times)

        if removed_count > 0:
            print(f"{removed_count} manuscripts after time {self._final_time} removed")

    def get_manuscript_ages(self) -> NDArray[np.float64]:
        """
        Calculate manuscript ages from creation times and final time.

        Returns:
            Array of manuscript ages (final_time - creation_time for each manuscript)
        """
        if self.creation_times:
            return np.array(
                [self.final_time - ct for ct in self.creation_times], dtype=np.float64
            )
        else:
            return np.array([], dtype=np.float64)

    def add_manuscript(self, creation_time: float) -> None:
        """
        Add a manuscript with given creation time to the process.

        Args:
            creation_time: Time when the manuscript was created
        """
        self.creation_times.append(creation_time)

    def set_final_time(self, final_time: float) -> None:
        """
        Set the final observation time for the process.

        Args:
            final_time: Time when the process was observed/terminated
        """
        self.final_time = final_time

    def get_process_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the birth-death process state.

        Returns:
            Dictionary with process information
        """
        summary: Dict[str, Any] = {
            "n_manuscripts": len(self.creation_times),
            "final_time": self.final_time,
            "fitted": self.fitted,
        }

        # Include parameters if they've been set (either at initialization or through fitting)
        if self.birth_rate is not None:
            summary["birth_rate"] = float(self.birth_rate)
        if self.death_rate is not None:
            summary["death_rate"] = float(self.death_rate)

        ages = self.get_manuscript_ages()
        if len(ages) > 0:
            summary.update(
                {
                    "min_age": float(ages.min()),
                    "max_age": float(ages.max()),
                    "mean_age": float(ages.mean()),
                }
            )

        return summary

    def validate_parameters(self) -> bool:
        """
        Validate that model parameters are set and reasonable.

        Returns:
            True if parameters are valid, False otherwise
        """
        if self.birth_rate is None or self.death_rate is None:
            return False

        if self.birth_rate <= 0 or self.death_rate <= 0:
            return False

        if not (np.isfinite(self.birth_rate) and np.isfinite(self.death_rate)):
            return False

        return True

    def update_parameters(
        self,
        birth_rate: float,
        death_rate: float,
        fit_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the model parameters after fitting.

        Use the fitting module directly to obtain parameters, then call this method
        to update the model state.

        Args:
            birth_rate: Fitted birth rate (λ)
            death_rate: Fitted death rate (μ)
            fit_results: Optional fit results dictionary from fitting module
        """
        self.birth_rate = np.float64(birth_rate)
        self.death_rate = np.float64(death_rate)
        self.fitted = True
        self.fit_results = fit_results

    def reset_parameters(self) -> None:
        """Reset fitted parameters and status."""
        self.birth_rate = None
        self.death_rate = None
        self.fitted = False
        self.fit_results = None
