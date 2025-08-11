"""
Inference module for manuscript data analysis using birth-death processes.

This module provides functionality to fit birth-death process models to manuscript
datasets, estimating birth and death rates based on manuscript counts and ages.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize


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

    Mathematical formulations:
    - P_n(T) = (μ-λ)²/λ² × exp[(μ-λ)T] × (1-exp[(μ-λ)T])^(n-1) / (1-μ/λ×exp[(μ-λ)T])^(n+1)
    - f(t) = (λe^(-λt) - μe^(-(λ-μ)T)*e^(-μt)) / (1 - e^(-(λ-μ)T))
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
        if self.final_time is not None and self.creation_times:
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

    def _calculate_population_probability(
        self,
        lambda_param: float,
        mu_param: float,
        time: float,
        n: int,
        log_scale_params: bool = False,
    ) -> np.float64:
        """
        Calculate P_n(T) - probability of n individuals at time T.

        Based on birth-death process theory from:
        Arley, Niels, and Vibeke Borchsenius.
        "On the Theory of Infinite Systems of Differential Equations
        and Their Application to the Theory of Stochastic Processes
        and the Perturbation Theory of Quantum Mechanics."
        Acta Mathematica 76, nos. 3–4 (1945): 261–322.
        https://doi.org/10.1007/BF02551579.

        Formula: log P_n(T) = log(1 - α_t) + log(1 - β_t) + (n-1) * log(β_t)
        where:
        - α_t = μ(e^((λ-μ)T) - 1) / (λe^((λ-μ)T) - μ)
        - β_t = (λ/μ) * α_t

        Args:
            lambda_param: Birth rate (λ) > 0
            mu_param: Death rate (μ) > 0
            time: Time T (float) >= 0
            n: Number of individuals (integer) >= 1
            log_scale_params: If True, parameters are in log scale

        Returns:
            Log probability of n individuals at time T
        """
        try:
            # Input validation
            if n < 1:
                return np.float64(-np.inf)

            if time < 0:
                return np.float64(-np.inf)

            # Handle parameter scaling
            if log_scale_params:
                log_lambda = lambda_param
                log_mu = mu_param
                # Convert log parameters back to original scale
                lambda_val = np.exp(lambda_param)
                mu_val = np.exp(mu_param)
            else:
                # Validate positive parameters
                if lambda_param <= 0 or mu_param <= 0:
                    return np.float64(-np.inf)
                lambda_val = lambda_param
                mu_val = mu_param
                log_lambda = np.log(lambda_param)
                log_mu = np.log(mu_param)

            # Check for numerical stability in log space
            if abs(log_lambda - log_mu) < 1e-12:
                # Limiting case: λ ≈ μ
                rate_time_product = lambda_val * time
                alpha_t = beta_t = rate_time_product / (1 + rate_time_product)
            else:
                # General case: λ ≠ μ
                lambda_minus_mu = lambda_val - mu_val
                exponent = lambda_minus_mu * time

                exp_term = np.exp(exponent)

                # Check for numerical issues
                if not np.isfinite(exp_term):
                    return np.float64(-np.inf)

                # Calculate α_t = μ(e^((λ-μ)T) - 1) / (λe^((λ-μ)T) - μ)
                denominator = lambda_val * exp_term - mu_val
                alpha_t = mu_val * (exp_term - 1) / denominator

                # Calculate β_t = (λ/μ) * α_t
                beta_t = (lambda_val / mu_val) * alpha_t

            # Validate α_t and β_t are in valid range (0, 1)
            if not (0 < alpha_t < 1):
                return np.float64(-np.inf)

            if not (0 < beta_t < 1):
                return np.float64(-np.inf)

            # Calculate final log probability
            log_term1 = np.log(1 - alpha_t)
            log_term2 = np.log(1 - beta_t)
            log_term3 = (n - 1) * np.log(beta_t)

            # Check all terms are finite
            if not (
                np.isfinite(log_term1)
                and np.isfinite(log_term2)
                and np.isfinite(log_term3)
            ):
                return np.float64(-np.inf)

            log_probability = log_term1 + log_term2 + log_term3

            # Final numerical check
            if not np.isfinite(log_probability):
                return np.float64(-np.inf)

            return np.float64(log_probability)

        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return np.float64(-np.inf)

    def _calculate_age_density(
        self,
        lambda_param: float,
        mu_param: float,
        age_t: float,
        T: float,
        log_scale_params: bool = False,
    ) -> np.float64:
        """
        Calculate log age density f(t) for a manuscript of given age.

        The age density function is:
        f(t) = (λe^(-λt) - μe^(-(λ-μ)T)*e^(-μt)) / (1 - e^(-(λ-μ)T))

        This is calculated in log space for numerical stability:
        log f(t) = log(λe^(-λt) - μe^(-(λ-μ)T)*e^(-μt)) - log(1 - e^(-(λ-μ)T))

        Args:
            lambda_param: Birth rate (λ) > 0
            mu_param: Death rate (μ) > 0
            time: Time T (float) >= 0
            n: Number of individuals (integer) >= 1
            log_scale_params: If True, parameters are in log scale

        Returns:
            Log age density f(t)
        """
        try:
            # Input validation
            if age_t < 0:
                return np.float64(-np.inf)

            if T < age_t:
                return np.float64(-np.inf)

            # Handle parameter scaling
            if log_scale_params:
                log_lambda = lambda_param
                log_mu = mu_param
                # Convert log parameters back to original scale
                lambda_val = np.exp(lambda_param)
                mu_val = np.exp(mu_param)
            else:
                # Validate positive parameters
                if lambda_param <= 0 or mu_param <= 0:
                    return np.float64(-np.inf)
                lambda_val = lambda_param
                mu_val = mu_param
                log_lambda = np.log(lambda_param)
                log_mu = np.log(mu_param)

            # Check for numerical stability in parameter space
            if abs(log_lambda - log_mu) < 1e-12:
                # Limiting case: λ ≈ μ
                # Use simplified formula from L'Hôpital's rule

                # Check for potential overflow in rate calculations
                lambda_age_product = lambda_val * age_t
                lambda_time_diff = lambda_val * (T - age_t)

                log_numerator = -lambda_age_product + np.log(1 + lambda_time_diff)
                log_denominator = np.log(T)

                # Check for numerical validity
                if not (np.isfinite(log_numerator) and np.isfinite(log_denominator)):
                    return np.float64(-np.inf)

            else:
                # General case: λ ≠ μ
                lambda_minus_mu = lambda_val - mu_val
                rate_diff_time = lambda_minus_mu * T

                # Calculate numerator terms
                numerator_term1 = lambda_val * np.exp(-lambda_val * age_t)
                numerator_term2 = (
                    mu_val * np.exp(-rate_diff_time) * np.exp(-mu_val * age_t)
                )

                numerator = numerator_term1 - numerator_term2
                denominator = 1 - np.exp(-rate_diff_time)

                if not np.sign(numerator) == np.sign(denominator):
                    return np.float64(-np.inf)

                log_numerator = np.log(abs(numerator))
                log_denominator = np.log(abs(denominator))

                # Check for numerical validity
                if not (np.isfinite(log_numerator) and np.isfinite(log_denominator)):
                    return np.float64(-np.inf)

            # Calculate final log density
            log_density = log_numerator - log_denominator

            # Final numerical check
            if not np.isfinite(log_density):
                return np.float64(-np.inf)

            return np.float64(log_density)

        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return np.float64(-np.inf)

    def log_likelihood(
        self,
        birth_rate: Optional[float] = None,
        death_rate: Optional[float] = None,
        log_scale_params: bool = False,
        ages: Optional[NDArray[np.float64]] = None,
        observation_time: Optional[float] = None,
    ) -> np.float64:
        """
        Calculate log-likelihood for observed manuscript data using birth-death process.

        This implements the proper birth-death process likelihood with:
        1. Probability of n individuals at time T: P_n(T)
        2. Age density for each manuscript: f(t)
        3. Total log-likelihood = log(P_n(T)) + Σlog(f(t_i))

        Args:
            birth_rate: Birth rate (λ > 0) or log(λ) if log_scale_params=True.
                       If None, uses self.birth_rate
            death_rate: Death rate (μ > 0) or log(μ) if log_scale_params=True.
                       If None, uses self.death_rate
            log_scale_params: If True, birth_rate and death_rate are in log scale
            ages: Array of manuscript ages. If None, uses calculated ages from creation_times
            observation_time: Observation time T. If None, uses self.final_time

        Returns:
            Log-likelihood value
        """
        try:
            if (birth_rate is None) or (death_rate is None):
                if log_scale_params:
                    raise ValueError(
                        "Cannot use log_scale_params=True when birth_rate and death_rate are not provided."
                    )
                log_scale_params = False
            # Use provided parameters or fall back to self attributes
            if birth_rate is not None:
                lambda_input = birth_rate
            else:
                if self.birth_rate is None:
                    raise ValueError("birth_rate not provided and not fitted")
                lambda_input = float(self.birth_rate)

            if death_rate is not None:
                mu_input = death_rate
            else:
                if self.death_rate is None:
                    raise ValueError("death_rate not provided and not fitted")
                mu_input = float(self.death_rate)

            if not (np.isfinite(lambda_input) and np.isfinite(mu_input)):
                return np.float64(-np.inf)

            if not log_scale_params:
                # Parameters are in original scale, validate directly
                if lambda_input <= 0 or mu_input <= 0:
                    return np.float64(-np.inf)

            # Validate ages data
            if ages is None:
                ages = self.get_manuscript_ages()

            if len(ages) == 0:
                return np.float64(-np.inf)

            # Check for negative ages (should not happen with proper data)
            if np.any(ages < 0):
                return np.float64(-np.inf)

            # Current time T (observation time)
            if observation_time is not None:
                T = observation_time
            else:
                T = self.final_time

            # Additional validation: T should be positive
            if T is None or T <= 0:
                return np.float64(-np.inf)

            # Total number of observed manuscripts
            n_total: int = int(len(ages))

            if n_total == 0:
                return np.float64(-np.inf)

            # 1. Calculate log(P_n(T)) - log probability of n individuals at time T
            log_P_n_T = self._calculate_population_probability(
                lambda_input,
                mu_input,
                T,
                n_total,
                log_scale_params=log_scale_params,
            )

            # Check if population probability calculation succeeded
            if not np.isfinite(log_P_n_T):
                return np.float64(-np.inf)

            # 2. Calculate age density for each manuscript
            # The helper function now includes comprehensive numerical checks
            log_densities = np.zeros_like(ages, dtype=np.float64)

            for i, age_t in enumerate(ages):
                # Validate individual age vs. total time
                if age_t > T:
                    return np.float64(-np.inf)

                # Calculate log age density using improved helper function
                # Pass log_scale_params to helper function
                log_density = self._calculate_age_density(
                    lambda_input,
                    mu_input,
                    age_t,
                    T,
                    log_scale_params=log_scale_params,
                )

                # Check if age density calculation succeeded
                if not np.isfinite(log_density):
                    return np.float64(-np.inf)

                # Add log density weighted by count
                log_densities[i] = log_density

            # 3. Total log-likelihood = log(P_n(T)) + sum of log densities
            log_density_sum = np.sum(log_densities)

            if not np.isfinite(log_density_sum):
                return np.float64(-np.inf)

            total_log_likelihood = log_P_n_T + log_density_sum

            # Final numerical validation
            if not np.isfinite(total_log_likelihood):
                return np.float64(-np.inf)

            return np.float64(total_log_likelihood)

        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            ZeroDivisionError,
            IndexError,
        ):
            return np.float64(-np.inf)

    def _log_likelihood_external(
        self,
        birth_rate: float,
        death_rate: float,
        ages: NDArray[np.float64],
        counts: NDArray[np.float64],
        log_scale_params: bool = False,
        total_time: Optional[float] = None,
    ) -> np.float64:
        """
        Calculate log-likelihood for external data (used in fitting).

        This is a helper method for calculating log-likelihood with provided
        parameters and data arrays, used during optimization.

        Args:
            birth_rate: Birth rate (λ > 0) or log(λ) if log_scale_params=True
            death_rate: Death rate (μ > 0) or log(μ) if log_scale_params=True
            ages: Array of manuscript ages (all >= 0)
            counts: Array of manuscript counts for each age (all >= 0)
            log_scale_params: If True, birth_rate and death_rate are in log scale
            total_time: Total observation time T. If None, uses max(ages)

        Returns:
            Log-likelihood value
        """
        try:
            # Use provided parameters
            lambda_input = birth_rate
            mu_input = death_rate

            if not (np.isfinite(lambda_input) and np.isfinite(mu_input)):
                return np.float64(-np.inf)

            if not log_scale_params:
                # Parameters are in original scale, validate directly
                if lambda_input <= 0 or mu_input <= 0:
                    return np.float64(-np.inf)

            # Enhanced data validation
            if len(ages) == 0 or len(counts) == 0:
                return np.float64(-np.inf)

            if len(ages) != len(counts):
                return np.float64(-np.inf)

            # Check for negative ages or counts
            if np.any(ages < 0) or np.any(counts < 0):
                return np.float64(-np.inf)

            # Current time T (total observation time)
            if total_time is not None:
                time_T = np.float64(total_time)
            else:
                # Fallback: use maximum age observed as approximation for T
                time_T = np.float64(np.max(ages))

            # Additional validation: T should be positive
            if time_T <= 0:
                return np.float64(-np.inf)

            # Total number of observed manuscripts
            n_total: int = int(np.sum(counts))
            assert isinstance(n_total, int)  # Type hint for mypy

            if n_total == 0:
                return np.float64(-np.inf)

            # Calculate P_n(T) using improved helper method
            # Pass log_scale_params to helper function
            log_P_n_T = self._calculate_population_probability(
                lambda_input,
                mu_input,
                time_T,
                n_total,
                log_scale_params=log_scale_params,
            )

            # Check if population probability calculation succeeded
            if not np.isfinite(log_P_n_T):
                return np.float64(-np.inf)

            # Calculate age density for each manuscript
            # The helper now includes comprehensive numerical checks
            log_densities = np.zeros_like(ages, dtype=np.float64)

            for i, age_t in enumerate(ages):
                # Skip zero-count entries
                if counts[i] == 0:
                    continue

                # Validate individual age vs. total time
                if age_t > time_T:
                    return np.float64(-np.inf)

                # Calculate log age density using improved helper function
                # Pass log_scale_params to helper function
                log_density = self._calculate_age_density(
                    lambda_input,
                    mu_input,
                    age_t,
                    time_T,
                    log_scale_params=log_scale_params,
                )

                # Check if age density calculation succeeded
                if not np.isfinite(log_density):
                    return np.float64(-np.inf)

                # Add log density weighted by count
                log_densities[i] = counts[i] * log_density

            # Total log-likelihood = log(P_n(T)) + sum of log densities
            log_density_sum = np.sum(log_densities)

            # Check intermediate sum is finite
            if not np.isfinite(log_density_sum):
                return np.float64(-np.inf)

            total_log_likelihood = log_P_n_T + log_density_sum

            # Final numerical validation
            if not np.isfinite(total_log_likelihood):
                return np.float64(-np.inf)

            return np.float64(total_log_likelihood)

        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            ZeroDivisionError,
            IndexError,
        ):
            return np.float64(-np.inf)

    def negative_log_likelihood(self, log_params: NDArray[np.float64]) -> float:
        """
        Calculate negative log-likelihood for optimization (public method for testing).

        This method is used internally by fit() methods but is also made public to enable
        comprehensive testing of the optimization objective function. Users should typically
        use log_likelihood() for direct likelihood calculations.

        Uses the proper birth-death process likelihood with:
        1. P_n(T) probability of n individuals at time T
        2. Age density f(t) for each observed manuscript

        Gets ages from self.get_manuscript_ages() and infers counts (1 for each manuscript).
        Optimization is performed over log-scale parameters for better numerical stability.

        Args:
            log_params: Log-scale parameters [log(birth_rate), log(death_rate)]

        Returns:
            Negative log-likelihood value (float for scipy optimization)
        """
        ages = self.get_manuscript_ages()
        if len(ages) == 0:
            return float(np.inf)  # No data available

        # Extract log-scale parameters
        log_birth_rate, log_death_rate = log_params

        result = -self.log_likelihood(
            birth_rate=float(np.exp(log_birth_rate)),
            death_rate=float(np.exp(log_death_rate)),
            log_scale_params=False,
            ages=ages,
            observation_time=self.final_time,
        )
        return float(result)  # Convert to Python float for scipy

    def get_crude_parameter_estimates(
        self, ages: NDArray[np.float64], update_instance: bool = False
    ) -> Tuple[float, float]:
        """
        Get crude initial parameter estimates based on manuscript age data.

        Uses simple heuristics to estimate birth and death rates:
        - Death rate inversely related to max survival age
        - Birth rate related to total observed manuscripts

        Args:
            ages: Array of manuscript ages
            update_instance: If True, updates self.birth_rate and self.death_rate
                           with the estimates (converted from log scale)

        Returns:
            Tuple of (estimated_birth_rate, estimated_death_rate) in log scale
        """
        if len(ages) == 0:
            raise ValueError("Cannot estimate parameters from empty age data")

        mean_age = float(ages.mean())
        n_manuscripts = len(ages)
        T = self.final_time

        # Simple heuristic: death rate inversely related to max survival age
        # birth rate related to total observed manuscripts
        est_birth_rate = 1.0 / mean_age
        est_death_rate = np.max([est_birth_rate - np.log(n_manuscripts) / T, 1e-8])

        # Optionally update instance parameters
        if update_instance:
            self.birth_rate = np.float64(est_birth_rate)
            self.death_rate = np.float64(est_death_rate)
            # Note: Don't set fitted=True since these are just crude estimates

        # Convert to log scale for optimization
        return (np.log(est_birth_rate), np.log(est_death_rate))

    def fit_from_self_data(
        self,
        method: Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ] = "L-BFGS-B",
        initial_params: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit birth-death process parameters using internal ages data.

        Uses self.creation_times and self.final_time to calculate ages and fit the model parameters.
        Optimization is performed over log-scale parameters for better numerical stability.

        Args:
            method: Optimization method for scipy.optimize.minimize
            initial_params: Initial parameter values (birth_rate, death_rate) in original scale

        Returns:
            Dictionary with fit results including parameters and statistics
        """
        ages = self.get_manuscript_ages()
        if len(ages) == 0:
            raise ValueError(
                "No ages data available. Set creation_times and final_time first."
            )

        # Set initial parameters with better estimates based on data
        if initial_params is None:
            initial_params = self.get_crude_parameter_estimates(ages)
        else:
            # Convert provided initial parameters to log scale
            initial_params = (np.log(initial_params[0]), np.log(initial_params[1]))

        # Parameter bounds in log space: log(birth_rate), log(death_rate)
        # birth_rate: 1e-6 to 10.0 -> log bounds: log(1e-6) to log(10.0)
        # death_rate: 1e-8 to 1.0 -> log bounds: log(1e-8) to log(1.0)
        bounds: List[Tuple[Optional[float], Optional[float]]] = [
            (np.log(1e-6), np.log(10.0)),  # log(birth_rate) bounds
            (np.log(1e-8), np.log(1.0)),  # log(death_rate) bounds
        ]

        # Optimize with multiple strategies for robustness
        optimization_methods = [method, "Nelder-Mead", "Powell"]
        best_result = None
        best_likelihood = -np.inf

        for opt_method in optimization_methods:
            try:
                # Ensure parameters are numpy arrays for scipy compatibility
                x0 = np.array(initial_params, dtype=np.float64)

                result = optimize.minimize(
                    self.negative_log_likelihood,
                    x0,
                    method=opt_method,
                    bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-9},
                )

                if result.success:
                    # Check if this is the best result so far
                    likelihood = -result.fun
                    if likelihood > best_likelihood:
                        best_result = result
                        best_likelihood = likelihood

                elif best_result is None:
                    # Keep even failed results if we have nothing better
                    best_result = result

            except Exception as e:
                # Continue trying other methods
                if best_result is None:
                    # Store the error for the first method only
                    self.fit_results = {
                        "success": False,
                        "error": f"All optimization methods failed. First error: {str(e)}",
                    }
                continue

        # Use the best result found
        if best_result is not None:
            result = best_result

            if result.success:
                # Convert log-scale parameters back to original scale
                log_birth_rate, log_death_rate = result.x
                self.birth_rate = np.float64(np.exp(log_birth_rate))
                self.death_rate = np.float64(np.exp(log_death_rate))
                self.fitted = True

                # Calculate additional statistics using internal data
                log_like = (
                    self.log_likelihood()
                )  # Use current self.birth_rate and self.death_rate
                n_data = len(ages)  # Use the ages we calculated at the beginning

                self.fit_results = {
                    "success": True,
                    "birth_rate": self.birth_rate,
                    "death_rate": self.death_rate,
                    "log_likelihood": log_like,
                    "n_data_points": n_data,
                    "optimization_result": result,
                    "ages_used": ages,
                    "counts_used": np.ones_like(ages, dtype=np.float64),
                }

            else:
                self.fit_results = {
                    "success": False,
                    "message": result.message,
                    "optimization_result": result,
                }
        else:
            # No optimization succeeded
            self.fit_results = {
                "success": False,
                "error": "All optimization methods failed",
            }

        return self.fit_results

    def fit(
        self,
        manuscript_data: Optional[pd.DataFrame] = None,
        age_column: str = "age",
        method: Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ] = "L-BFGS-B",
        initial_params: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit birth-death process parameters to manuscript data.

        Optimization is performed over log-scale parameters for better numerical stability.

        Args:
            manuscript_data: DataFrame with manuscript information (if None, uses internal ages)
            age_column: Column name containing manuscript ages
            method: Optimization method for scipy.optimize.minimize
            initial_params: Initial parameter values (birth_rate, death_rate) in original scale

        Returns:
            Dictionary with fit results including parameters and statistics
        """
        # If no manuscript_data provided, use internal ages
        if manuscript_data is None:
            return self.fit_from_self_data(method=method, initial_params=initial_params)

        # Prepare data from provided DataFrame
        # Check if we have creation_times and final_time available
        ages_array = self.get_manuscript_ages()
        if len(ages_array) > 0:
            # Use the calculated process ages directly
            # Create age bins
            age_bins = np.arange(
                0, float(ages_array.max()) + 50, 50, dtype=np.float64
            )  # 50-year bins
            counts, bin_edges = np.histogram(ages_array, bins=age_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            # Assume one manuscript per row, bin by age
            ages = manuscript_data[age_column].values
            # Convert to numpy array to ensure proper typing
            ages_array = np.asarray(ages, dtype=np.float64)

            # Check if external data is empty
            if len(ages_array) == 0:
                raise ValueError("No valid data points for fitting")

            age_bins = np.arange(
                0, float(ages_array.max()) + 50, 50, dtype=np.float64
            )  # 50-year bins
            counts, bin_edges = np.histogram(ages_array, bins=age_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Remove empty bins - convert to numpy arrays for type safety
        counts_array = np.asarray(counts, dtype=np.float64)
        bin_centers_array = np.asarray(bin_centers, dtype=np.float64)
        non_zero_mask = counts_array > 0
        ages_fit: NDArray[np.float64] = bin_centers_array[non_zero_mask]
        counts_fit: NDArray[np.float64] = counts_array[non_zero_mask]

        if len(ages_fit) == 0:
            raise ValueError("No valid data points for fitting")

        # Set final_time if not already set, ensuring it's larger than max age
        if self.final_time is None:
            max_age = float(ages_fit.max())
            self.final_time = max_age + 100.0  # Add buffer beyond maximum observed age

        # Set initial parameters with better estimates based on data
        if initial_params is None:
            initial_params = self.get_crude_parameter_estimates(ages_fit)
        else:
            # Convert provided initial parameters to log scale
            initial_params = (np.log(initial_params[0]), np.log(initial_params[1]))

        # Parameter bounds in log space: log(birth_rate), log(death_rate)
        # birth_rate: 1e-6 to 10.0 -> log bounds: log(1e-6) to log(10.0)
        # death_rate: 1e-8 to 1.0 -> log bounds: log(1e-8) to log(1.0)
        bounds: List[Tuple[Optional[float], Optional[float]]] = [
            (np.log(1e-6), np.log(10.0)),  # log(birth_rate) bounds
            (np.log(1e-8), np.log(1.0)),  # log(death_rate) bounds
        ]

        # Create objective function for external data fitting
        def objective_func(params: NDArray[np.float64]) -> float:
            # Extract log-scale parameters
            log_birth_rate, log_death_rate = params

            # Calculate negative log-likelihood with provided data
            result = -self.log_likelihood(
                birth_rate=float(np.exp(log_birth_rate)),
                death_rate=float(np.exp(log_death_rate)),
                log_scale_params=False,
                ages=ages_fit,
                observation_time=self.final_time,
            )
            return float(result)

        # Optimize with multiple strategies for robustness
        optimization_methods = [method, "Nelder-Mead", "Powell"]
        best_result = None
        best_likelihood = -np.inf

        for opt_method in optimization_methods:
            try:
                # Ensure parameters are numpy arrays for scipy compatibility
                x0 = np.array(initial_params, dtype=np.float64)

                result = optimize.minimize(
                    objective_func,
                    x0,
                    method=opt_method,
                    bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-9},
                )

                if result.success:
                    # Check if this is the best result so far
                    likelihood = -result.fun
                    if likelihood > best_likelihood:
                        best_result = result
                        best_likelihood = likelihood

                elif best_result is None:
                    # Keep even failed results if we have nothing better
                    best_result = result

            except Exception as e:
                # Continue trying other methods
                if best_result is None:
                    # Store the error for the first method only
                    self.fit_results = {
                        "success": False,
                        "error": f"All optimization methods failed. First error: {str(e)}",
                    }
                continue

        # Use the best result found
        if best_result is not None:
            result = best_result

            if result.success:
                # Convert log-scale parameters back to original scale
                log_birth_rate, log_death_rate = result.x
                self.birth_rate = np.float64(np.exp(log_birth_rate))
                self.death_rate = np.float64(np.exp(log_death_rate))
                self.fitted = True

                # Calculate additional statistics using fitted parameters
                birth_rate = float(self.birth_rate)
                death_rate = float(self.death_rate)
                log_like = self._log_likelihood_external(
                    birth_rate, death_rate, ages_fit, counts_fit
                )
                n_data = len(ages_fit)

                self.fit_results = {
                    "success": True,
                    "birth_rate": self.birth_rate,
                    "death_rate": self.death_rate,
                    "log_likelihood": log_like,
                    "n_data_points": n_data,
                    "optimization_result": result,
                    "ages_used": ages_fit,
                    "counts_used": counts_fit,
                }

            else:
                self.fit_results = {
                    "success": False,
                    "message": result.message,
                    "optimization_result": result,
                }
        else:
            # No optimization succeeded
            self.fit_results = {
                "success": False,
                "error": "All optimization methods failed",
            }

        return self.fit_results

    def predict(self, ages: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict expected manuscript counts for given ages.

        Args:
            ages: Array of ages to predict for

        Returns:
            Array of predicted manuscript counts
        """
        if not self.fitted:
            warnings.warn("Model has not been fitted. Using current parameter values.")

        if self.birth_rate is None or self.death_rate is None:
            raise ValueError("Cannot predict without fitted birth_rate and death_rate")

        return self.birth_rate * np.exp(-self.death_rate * ages)

    def simulate(
        self,
        max_time: float = 2000,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Simulate manuscript data from the birth-death process.

        This creates a proper birth-death process simulation where manuscripts
        are born at rate λ and die at rate μ. The simulation runs until max_time
        and records both creation times and survival status.

        Args:
            max_time: Maximum time to simulate the process
            random_seed: Random seed for reproducibility

        Returns:
            DataFrame with simulated manuscript data including creation times and ages
        """
        if self.birth_rate is None or self.death_rate is None:
            raise ValueError("Cannot simulate without fitted birth_rate and death_rate")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Reset the process
        self.creation_times = []
        self.final_time = max_time

        # Simulate birth-death process
        current_time = 0.0
        manuscripts_alive: List[
            Tuple[float, int]
        ] = []  # List of (creation_time, manuscript_id)
        all_manuscripts: List[
            Dict[str, Any]
        ] = []  # Record of all manuscripts ever created
        manuscript_id = 0

        while current_time < max_time:
            # Current population size
            n_alive = len(manuscripts_alive)

            # Total rate = birth_rate + death_rate * n_alive
            total_rate = float(self.birth_rate + self.death_rate * n_alive)

            if total_rate <= 0:
                break

            # Time to next event
            dt = np.random.exponential(1.0 / total_rate)
            current_time += dt

            if current_time >= max_time:
                break

            # Determine if event is birth or death
            if np.random.random() < float(self.birth_rate) / total_rate:
                # Birth event
                manuscripts_alive.append((current_time, manuscript_id))
                all_manuscripts.append(
                    {
                        "manuscript_id": manuscript_id,
                        "creation_time": current_time,
                        "death_time": None,
                        "survived": True,
                    }
                )
                self.creation_times.append(current_time)
                manuscript_id += 1
            else:
                # Death event (if any manuscripts alive)
                if manuscripts_alive:
                    # Choose a random manuscript to die
                    death_idx = np.random.randint(len(manuscripts_alive))
                    creation_time, dead_id = manuscripts_alive.pop(death_idx)

                    # Remove the creation time from self.creation_times
                    # since this manuscript no longer survives
                    if creation_time in self.creation_times:
                        self.creation_times.remove(creation_time)

                    # Update the record
                    for manuscript in all_manuscripts:
                        if manuscript["manuscript_id"] == dead_id:
                            manuscript["death_time"] = current_time
                            manuscript["survived"] = False
                            break

        # Calculate ages for surviving manuscripts (ages are now calculated on-demand)

        # Calculate ages for surviving manuscripts
        surviving_data: List[Dict[str, Any]] = []
        for manuscript in all_manuscripts:
            if manuscript["survived"]:
                creation_time = manuscript["creation_time"]
                assert (
                    creation_time is not None
                )  # Surviving manuscripts must have creation_time
                age = max_time - creation_time
                surviving_data.append(
                    {
                        "manuscript_id": manuscript["manuscript_id"],
                        "creation_time": creation_time,
                        "age": age,
                        "survived": True,
                    }
                )

        # Also include information about dead manuscripts
        dead_data: List[Dict[str, Any]] = []
        for manuscript in all_manuscripts:
            if not manuscript["survived"]:
                death_time = manuscript["death_time"]
                creation_time = manuscript["creation_time"]
                assert (
                    death_time is not None and creation_time is not None
                )  # Dead manuscripts must have both times
                lifespan = death_time - creation_time
                dead_data.append(
                    {
                        "manuscript_id": manuscript["manuscript_id"],
                        "creation_time": creation_time,
                        "death_time": death_time,
                        "lifespan": lifespan,
                        "survived": False,
                    }
                )

        # Combine data
        all_data = surviving_data + dead_data

        if not all_data:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(
                columns=["manuscript_id", "creation_time", "age", "survived"]
            )

        df = pd.DataFrame(all_data)
        return df


def plot_fit_results(
    model: BirthDeathProcess, manuscript_data: pd.DataFrame, age_column: str = "age"
) -> None:
    """
    Plot birth-death process fit results (requires plotly).

    Args:
        model: Fitted BirthDeathProcess instance
        manuscript_data: Manuscript data used for fitting
        age_column: Column name for ages
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available. Cannot create plots.")
        return

    if not model.fitted or model.fit_results is None:
        print("Model must be fitted before plotting.")
        return

    # Get ages and counts used in fitting
    ages_fit = model.fit_results["ages_used"]
    counts_fit = model.fit_results["counts_used"]

    # Generate prediction curve
    age_range = np.linspace(0, ages_fit.max(), 100, dtype=np.float64)
    predicted_counts = model.predict(age_range)

    # Create subplot
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Observed vs Predicted Manuscript Counts", "Residuals"],
        vertical_spacing=0.12,
    )

    # Plot observed data
    fig.add_trace(
        go.Scatter(
            x=ages_fit,
            y=counts_fit,
            mode="markers",
            name="Observed",
            marker=dict(size=8, color="blue"),
        ),
        row=1,
        col=1,
    )

    # Plot predicted curve
    fig.add_trace(
        go.Scatter(
            x=age_range,
            y=predicted_counts,
            mode="lines",
            name="Predicted",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    # Plot residuals
    predicted_at_data = model.predict(ages_fit)
    residuals = counts_fit - predicted_at_data

    fig.add_trace(
        go.Scatter(
            x=ages_fit,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(size=6, color="green"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Add zero line for residuals
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=600,
        title=f"Birth-Death Process Fit<br>λ={model.birth_rate:.4f}, μ={model.death_rate:.4f}",
    )
    fig.update_xaxes(title_text="Manuscript Age (years)", row=2, col=1)
    fig.update_yaxes(title_text="Number of Manuscripts", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)

    fig.show()


# Example usage and testing functions
def example_analysis() -> None:
    """
    Example analysis workflow for birth-death process fitting.
    """
    print("=== Birth-Death Process Analysis Example ===")

    # Create synthetic data for demonstration (medieval manuscript scenario)
    np.random.seed(42)
    true_birth_rate = 0.5
    true_death_rate = 0.001

    # Create manuscripts from years 500-1400 CE (medieval period)
    manuscript_years = np.linspace(500, 1400, 30, dtype=np.float64)
    ages_at_1500 = 1500 - manuscript_years  # Ages when observed in 1500 CE
    true_counts = true_birth_rate * np.exp(-true_death_rate * ages_at_1500)
    observed_counts = np.random.poisson(true_counts)

    synthetic_data = pd.DataFrame(
        {"age": ages_at_1500, "count": observed_counts, "midYear": manuscript_years}
    )

    print("Synthetic medieval manuscript data created with true parameters:")
    print(f"  Birth rate (λ): {true_birth_rate}")
    print(f"  Death rate (μ): {true_death_rate}")
    print(
        f"  Manuscript years: {manuscript_years.min():.0f}-{manuscript_years.max():.0f} CE"
    )
    print("  Analysis cutoff: 1500 CE")

    # Fit model (no initial data, will fit from the synthetic_data)
    model = BirthDeathProcess([], 1500.0)

    fit_results = model.fit(synthetic_data, age_column="age")

    if fit_results["success"]:
        print("\nFit Results:")
        print(f"  Estimated birth rate (λ): {model.birth_rate:.6f}")
        print(f"  Estimated death rate (μ): {model.death_rate:.6f}")
        print(f"  Log-likelihood: {fit_results['log_likelihood']:.2f}")

        # Calculate parameter errors
        if model.birth_rate is not None and model.death_rate is not None:
            birth_error = (
                abs(model.birth_rate - true_birth_rate) / true_birth_rate * 100
            )
            death_error = (
                abs(model.death_rate - true_death_rate) / true_death_rate * 100
            )

            print("\nParameter Estimation Errors:")
            print(f"  Birth rate error: {birth_error:.2f}%")
            print(f"  Death rate error: {death_error:.2f}%")

    else:
        print(f"\nFit failed: {fit_results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    # Run example analysis
    example_analysis()
