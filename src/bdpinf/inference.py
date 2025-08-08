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
        birth_rate: float = 1.0,
        death_rate: float = 0.1,
        creation_times: Optional[List[float]] = None,
        final_time: Optional[float] = None,
    ):
        """
        Initialize birth-death process with parameters and process history.

        Args:
            birth_rate: Manuscript creation rate (λ)
            death_rate: Manuscript loss rate (μ)
            creation_times: List of times when manuscripts were created
            final_time: Time when the process was terminated/observed
        """
        self.birth_rate: np.float64 = np.float64(birth_rate)
        self.death_rate: np.float64 = np.float64(death_rate)
        self.creation_times: List[float] = (
            creation_times if creation_times is not None else []
        )
        self.final_time: Optional[float] = final_time
        self.fitted: bool = False
        self.fit_results: Optional[Dict[str, Any]] = None

        # Calculate ages from creation times and final time
        self._update_ages()

    def _update_ages(self) -> None:
        """Update the ages array when creation_times or final_time change."""
        if self.final_time is not None and self.creation_times:
            self.ages = np.array(
                [self.final_time - ct for ct in self.creation_times], dtype=np.float64
            )
        else:
            self.ages = np.array([], dtype=np.float64)

    def get_manuscript_ages(self) -> NDArray[np.float64]:
        """
        Get manuscript ages based on creation times and final time.

        Returns:
            Array of manuscript ages (final_time - creation_time for each manuscript)
        """
        return self.ages

    def add_manuscript(self, creation_time: float) -> None:
        """
        Add a manuscript with given creation time to the process.

        Args:
            creation_time: Time when the manuscript was created
        """
        self.creation_times.append(creation_time)
        self._update_ages()

    def set_final_time(self, final_time: float) -> None:
        """
        Set the final observation time for the process.

        Args:
            final_time: Time when the process was observed/terminated
        """
        self.final_time = final_time
        self._update_ages()

    def get_process_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the birth-death process state.

        Returns:
            Dictionary with process information
        """
        summary: Dict[str, Any] = {
            "birth_rate": float(self.birth_rate),
            "death_rate": float(self.death_rate),
            "n_manuscripts": len(self.creation_times),
            "final_time": self.final_time,
        }

        if len(self.ages) > 0:
            summary.update(
                {
                    "min_age": float(self.ages.min()),
                    "max_age": float(self.ages.max()),
                    "mean_age": float(self.ages.mean()),
                }
            )

        return summary

    def survival_probability(self, age: float) -> np.float64:
        """
        Calculate survival probability for a manuscript of given age.

        For a birth-death process, the survival probability is:
        P(survival | age) = exp(-μ * age)

        Args:
            age: Age of manuscript in years

        Returns:
            Probability of survival
        """
        return np.float64(np.exp(-self.death_rate * age))

    def expected_manuscripts(self, age: float, time_window: float = 1.0) -> np.float64:
        """
        Calculate expected number of manuscripts created in a time window
        that survive to a given age.

        Args:
            age: Age of manuscripts
            time_window: Time window for manuscript creation

        Returns:
            Expected number of surviving manuscripts
        """
        return np.float64(
            self.birth_rate * time_window * self.survival_probability(age)
        )

    def _calculate_population_probability(
        self,
        lambda_param: float,
        mu_param: float,
        time: float,
        n: int,
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

        Args:
            lambda_param: Birth rate (λ)
            mu_param: Death rate (μ)
            time: Time T (float)
            n: Number of individuals (integer)

        Returns:
            Log probability of n individuals at time T
        """
        try:
            # Handle edge cases
            if lambda_param <= 0 or mu_param <= 0:
                return np.float64(-np.inf)

            lambda_minus_mu = lambda_param - mu_param

            # Handle the case where λ ≈ μ
            if abs(lambda_minus_mu) < 1e-12:
                # Use limiting case formula or small perturbation
                if lambda_param >= mu_param:
                    lambda_minus_mu = 1e-12
                else:
                    lambda_minus_mu = -1e-12

            exp_term = np.exp(lambda_minus_mu * time)

            # Check for numerical overflow/underflow
            if not np.isfinite(exp_term):
                return np.float64(-np.inf)

            # Calculate alpha_t and beta_t with numerical stability checks
            denominator = lambda_param * exp_term - mu_param
            if abs(denominator) < 1e-15:
                return np.float64(-np.inf)

            alpha_t = mu_param * (exp_term - 1) / denominator

            if not np.isfinite(alpha_t) or alpha_t >= 1 or alpha_t <= 0:
                return np.float64(-np.inf)

            beta_t = lambda_param / mu_param * alpha_t

            if not np.isfinite(beta_t) or beta_t >= 1 or beta_t <= 0:
                return np.float64(-np.inf)

            # Calculate log probability: log(1 - alpha_t) + log(1 - beta_t) + (n - 1) * log(beta_t)
            term1 = np.log(1 - alpha_t)
            term2 = np.log(1 - beta_t)
            term3 = (n - 1) * np.log(beta_t)

            # Check each term for validity
            if not (np.isfinite(term1) and np.isfinite(term2) and np.isfinite(term3)):
                return np.float64(-np.inf)

            log_P_n_T = term1 + term2 + term3

        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return np.float64(-np.inf)

        return np.float64(log_P_n_T)

    def _calculate_age_density(
        self,
        lambda_param: float,
        mu_param: float,
        age_t: float,
        T: float,
    ) -> np.float64:
        """
        Calculate log age density f(t) for a manuscript of given age.

        The age density function is:
        f(t) = (λe^(-λt) - μe^(-(λ-μ)T)*e^(-μt)) / (1 - e^(-(λ-μ)T))

        This is calculated in log space for numerical stability:
        log f(t) = log(λe^(-λt) - μe^(-(λ-μ)T)*e^(-μt)) - log(1 - e^(-(λ-μ)T))

        Args:
            lambda_param: Birth rate (λ)
            mu_param: Death rate (μ)
            age_t: Age of the manuscript
            T: Total observation time

        Returns:
            Log age density f(t)
        """
        try:
            lambda_minus_mu = lambda_param - mu_param

            # Calculate log f(t) in parts to avoid overflow
            # Numerator: λe^(-λt) - μe^(-(λ-μ)T)*e^(-μt)
            term1 = lambda_param * np.exp(-lambda_param * age_t)
            term2 = (
                mu_param * np.exp(-(lambda_minus_mu) * T) * np.exp(-mu_param * age_t)
            )

            numerator = term1 - term2

            # Denominator: 1 - e^(-(λ-μ)T)
            exp_neg_diff_T = np.exp(-(lambda_minus_mu) * T)
            denominator = 1 - exp_neg_diff_T

            if denominator * numerator <= 0:
                return np.float64(-np.inf)

            log_numerator = np.log(np.abs(numerator))
            log_denominator = np.log(np.abs(denominator))

            # log f(t) = log(numerator) - log(denominator)
            log_density = log_numerator - log_denominator

            return np.float64(log_density)

        except (ValueError, FloatingPointError, OverflowError):
            return np.float64(-np.inf)

    def log_likelihood(
        self,
        birth_rate: Optional[float] = None,
        death_rate: Optional[float] = None,
    ) -> np.float64:
        """
        Calculate log-likelihood for observed manuscript data using birth-death process.

        This implements the proper birth-death process likelihood with:
        1. Probability of n individuals at time T: P_n(t)
        2. Age density for each manuscript: f(t)
        3. Total log-likelihood as sum of log densities

        Uses self.ages for manuscript ages and infers counts (assumes 1 per manuscript).

        Args:
            birth_rate: Birth rate (λ). If None, uses self.birth_rate
            death_rate: Death rate (μ). If None, uses self.death_rate

        Returns:
            Log-likelihood value
        """
        # Use provided parameters or fall back to self attributes
        lambda_param = birth_rate if birth_rate is not None else float(self.birth_rate)
        mu_param = death_rate if death_rate is not None else float(self.death_rate)

        if lambda_param <= 0 or mu_param < 0:
            return np.float64(-np.inf)

        # Use self.ages and infer counts (1 per manuscript)
        ages = self.ages
        if len(ages) == 0:
            return np.float64(-np.inf)

        counts = np.ones_like(ages, dtype=np.float64)

        # Handle the case where λ ≈ μ (critical case) - use small offset
        if np.abs(lambda_param - mu_param) < 1e-10:
            # Add small perturbation to avoid singularity
            if lambda_param >= mu_param:
                mu_param = lambda_param - 1e-10
            else:
                lambda_param = mu_param + 1e-10

        # Current time T (maximum age observed, representing present time)
        T = np.float64(np.max(ages))

        # Total number of observed manuscripts
        n_total: int = int(np.sum(counts))

        if n_total == 0:
            return np.float64(-np.inf)

        # 1. Calculate P_n(T) - probability of n individuals at time T

        log_P_n_T = self._calculate_population_probability(
            lambda_param, mu_param, T, n_total
        )
        if not np.isfinite(log_P_n_T):
            return np.float64(-np.inf)

        # 2. Calculate age density for each manuscript
        # f(t) = λe^(-λt) * (μe^((λ-μ)t) - μ) / (λe^((λ-μ)T) - μ)

        # For each age t, calculate the log density
        log_densities = np.zeros_like(ages, dtype=np.float64)

        for i, age_t in enumerate(ages):
            if counts[i] == 0:
                continue

            # Use helper function to calculate log age density
            log_density = self._calculate_age_density(lambda_param, mu_param, age_t, T)

            if not np.isfinite(log_density):
                return np.float64(-np.inf)

            # Add log density weighted by count
            log_densities[i] = counts[i] * log_density

        # 3. Total log-likelihood = log(P_n(T)) + sum of log densities
        total_log_likelihood = log_P_n_T + np.sum(log_densities)

        # Final numerical check
        if not np.isfinite(total_log_likelihood):
            return np.float64(-np.inf)

        return np.float64(total_log_likelihood)

    def _log_likelihood_external(
        self,
        birth_rate: float,
        death_rate: float,
        ages: NDArray[np.float64],
        counts: NDArray[np.float64],
    ) -> np.float64:
        """
        Calculate log-likelihood for external data (used in fitting).

        This is a helper method for calculating log-likelihood with provided
        parameters and data arrays, used during optimization.

        Args:
            birth_rate: Birth rate (λ)
            death_rate: Death rate (μ)
            ages: Array of manuscript ages
            counts: Array of manuscript counts for each age

        Returns:
            Log-likelihood value
        """
        lambda_param, mu_param = float(birth_rate), float(death_rate)

        if lambda_param <= 0 or mu_param < 0:
            return np.float64(-np.inf)

        # Handle the case where λ ≈ μ (critical case) - use small offset
        if np.abs(lambda_param - mu_param) < 1e-10:
            # Add small perturbation to avoid singularity
            if lambda_param >= mu_param:
                mu_param = lambda_param - 1e-10
            else:
                lambda_param = mu_param + 1e-10

        # Current time T (maximum age observed, representing present time)
        T = np.float64(np.max(ages))

        # Total number of observed manuscripts
        n_total: int = int(np.sum(counts))
        assert isinstance(n_total, int)  # Type hint for mypy

        if n_total == 0:
            return np.float64(-np.inf)

        # Calculate P_n(T) using helper method
        log_P_n_T = self._calculate_population_probability(
            lambda_param, mu_param, T, n_total
        )
        if not np.isfinite(log_P_n_T):
            return np.float64(-np.inf)

        # 2. Calculate age density for each manuscript
        # For each age t, calculate the log density
        log_densities = np.zeros_like(ages, dtype=np.float64)

        for i, age_t in enumerate(ages):
            if counts[i] == 0:
                continue

            # Use helper function to calculate log age density
            log_density = self._calculate_age_density(lambda_param, mu_param, age_t, T)

            if not np.isfinite(log_density):
                return np.float64(-np.inf)

            # Add log density weighted by count
            log_densities[i] = counts[i] * log_density

        # 3. Total log-likelihood = log(P_n(T)) + sum of log densities
        total_log_likelihood = log_P_n_T + np.sum(log_densities)

        # Final numerical check
        if not np.isfinite(total_log_likelihood):
            return np.float64(-np.inf)

        return np.float64(total_log_likelihood)

    def negative_log_likelihood(self, params: NDArray[np.float64]) -> float:
        """
        Negative log-likelihood for optimization (minimization).

        Uses the proper birth-death process likelihood with:
        1. P_n(T) probability of n individuals at time T
        2. Age density f(t) for each observed manuscript

        Uses the internal ages from self.ages and infers counts (1 for each manuscript).

        Args:
            params: Parameters [birth_rate (λ), death_rate (μ)]

        Returns:
            Negative log-likelihood value (float for scipy optimization)
        """
        if len(self.ages) == 0:
            return float(np.inf)  # No data available

        # Extract parameters
        birth_rate, death_rate = params

        result = -self.log_likelihood(
            birth_rate=float(birth_rate), death_rate=float(death_rate)
        )
        return float(result)  # Convert to Python float for scipy

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

        Uses self.ages and self.creation_times to fit the model parameters.

        Args:
            method: Optimization method for scipy.optimize.minimize
            initial_params: Initial parameter values (birth_rate, death_rate)

        Returns:
            Dictionary with fit results including parameters and statistics
        """
        if len(self.ages) == 0:
            raise ValueError(
                "No ages data available. Set creation_times and final_time first."
            )

        # Set initial parameters with better estimates based on data
        if initial_params is None:
            # Estimate initial parameters from data characteristics
            max_age = float(self.ages.max())
            n_manuscripts = len(self.ages)

            # Simple heuristic: death rate inversely related to max survival age
            # birth rate related to total observed manuscripts
            est_death_rate = 1.0 / max_age if max_age > 0 else 0.001
            est_birth_rate = (
                max(0.001, n_manuscripts / (max_age * 100)) if max_age > 0 else 0.01
            )

            initial_params = (est_birth_rate, est_death_rate)

        # Parameter bounds: birth_rate > 0, death_rate >= 0
        # Add reasonable upper bounds to prevent extreme values
        bounds: List[Tuple[Optional[float], Optional[float]]] = [
            (1e-6, 10.0),  # birth_rate: small positive to reasonable maximum
            (1e-8, 1.0),  # death_rate: very small to 1 (100% death rate)
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
                self.birth_rate, self.death_rate = result.x
                self.fitted = True

                # Calculate additional statistics using internal data
                log_like = (
                    self.log_likelihood()
                )  # Use current self.birth_rate and self.death_rate
                n_params = len(result.x)
                n_data = len(self.ages)
                aic = 2 * n_params - 2 * log_like
                bic = n_params * np.log(n_data) - 2 * log_like

                self.fit_results = {
                    "success": True,
                    "birth_rate": self.birth_rate,
                    "death_rate": self.death_rate,
                    "log_likelihood": log_like,
                    "aic": aic,
                    "bic": bic,
                    "n_data_points": n_data,
                    "optimization_result": result,
                    "ages_used": self.ages,
                    "counts_used": np.ones_like(self.ages, dtype=np.float64),
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
        count_column: Optional[str] = None,
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

        Args:
            manuscript_data: DataFrame with manuscript information (if None, uses internal ages)
            age_column: Column name containing manuscript ages
            count_column: Column name for counts (if None, assumes one manuscript per row)
            method: Optimization method for scipy.optimize.minimize
            initial_params: Initial parameter values (birth_rate, death_rate)

        Returns:
            Dictionary with fit results including parameters and statistics
        """
        # If no manuscript_data provided, use internal ages
        if manuscript_data is None:
            return self.fit_from_self_data(method=method, initial_params=initial_params)

        # Prepare data from provided DataFrame
        if count_column is None:
            # Check if we have creation_times and final_time available
            if len(self.ages) > 0:
                # Use the stored process ages directly
                ages_array = self.ages
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
                age_bins = np.arange(
                    0, float(ages_array.max()) + 50, 50, dtype=np.float64
                )  # 50-year bins
                counts, bin_edges = np.histogram(ages_array, bins=age_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            # Use provided counts
            bin_centers = manuscript_data[age_column].values
            counts_values = manuscript_data[count_column].values
            counts = np.asarray(counts_values, dtype=np.float64)

        # Remove empty bins - convert to numpy arrays for type safety
        counts_array = np.asarray(counts, dtype=np.float64)
        bin_centers_array = np.asarray(bin_centers, dtype=np.float64)
        non_zero_mask = counts_array > 0
        ages_fit: NDArray[np.float64] = bin_centers_array[non_zero_mask]
        counts_fit: NDArray[np.float64] = counts_array[non_zero_mask]

        if len(ages_fit) == 0:
            raise ValueError("No valid data points for fitting")

        # Set initial parameters with better estimates based on data
        if initial_params is None:
            # Estimate initial parameters from data characteristics
            max_age = float(ages_fit.max())
            total_manuscripts = float(counts_fit.sum())

            # Simple heuristic: death rate inversely related to max survival age
            # birth rate related to total observed manuscripts
            est_death_rate = 1.0 / max_age if max_age > 0 else 0.001
            est_birth_rate = (
                max(0.001, total_manuscripts / (max_age * 100)) if max_age > 0 else 0.01
            )

            initial_params = (est_birth_rate, est_death_rate)

        # Parameter bounds: birth_rate > 0, death_rate >= 0
        # Add reasonable upper bounds to prevent extreme values
        bounds: List[Tuple[Optional[float], Optional[float]]] = [
            (1e-6, 10.0),  # birth_rate: small positive to reasonable maximum
            (1e-8, 1.0),  # death_rate: very small to 1 (100% death rate)
        ]

        # Create objective function for external data fitting
        def objective_func(params: NDArray[np.float64]) -> float:
            # Store original ages for restoration
            original_ages = self.ages.copy() if hasattr(self, "ages") else None
            try:
                # Temporarily set ages for external data
                self.ages = ages_fit
                return self.negative_log_likelihood(params)
            finally:
                # Restore original ages
                if original_ages is not None:
                    self.ages = original_ages

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
                self.birth_rate, self.death_rate = result.x
                self.fitted = True

                # Calculate additional statistics using fitted parameters
                birth_rate, death_rate = result.x
                log_like = self._log_likelihood_external(
                    float(birth_rate), float(death_rate), ages_fit, counts_fit
                )
                n_params = len(result.x)
                n_data = len(ages_fit)
                aic = 2 * n_params - 2 * log_like
                bic = n_params * np.log(n_data) - 2 * log_like

                self.fit_results = {
                    "success": True,
                    "birth_rate": self.birth_rate,
                    "death_rate": self.death_rate,
                    "log_likelihood": log_like,
                    "aic": aic,
                    "bic": bic,
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
        if random_seed is not None:
            np.random.seed(random_seed)

        # Reset the process
        self.creation_times = []
        self.final_time = max_time
        self._update_ages()

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

        # Update ages after simulation is complete
        self._update_ages()

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


def prepare_manuscript_data(
    df: pd.DataFrame, current_year: int = 1500, date_column: str = "midYear"
) -> pd.DataFrame:
    """
    Prepare manuscript data for birth-death process analysis.

    Args:
        df: DataFrame with manuscript data
        current_year: Current year for age calculation (default 1500 for medieval manuscript analysis)
        date_column: Column containing manuscript dates

    Returns:
        DataFrame with age calculations and binned data, filtered to manuscripts before current_year
    """
    # Filter manuscripts to only include those with dates before current_year
    manuscript_data = df.copy()
    manuscript_data = manuscript_data[manuscript_data[date_column] < current_year]

    # Calculate ages (manuscripts created before current_year will have positive ages)
    manuscript_data["age"] = current_year - manuscript_data[date_column]

    # Remove manuscripts with negative ages (future dates) - should be minimal after filtering
    manuscript_data = manuscript_data[manuscript_data["age"] >= 0]

    # Add century and period groupings for analysis
    manuscript_data["century"] = np.floor(manuscript_data[date_column] / 100) + 1

    # Define historical periods
    def get_period(year: float) -> str:
        if year < 500:
            return "Early Christian"
        elif year < 1000:
            return "Early Medieval"
        elif year < 1500:
            return "Late Medieval"
        else:
            return "Post-Medieval"

    manuscript_data["period"] = manuscript_data[date_column].apply(get_period)

    return manuscript_data


def compare_models(
    models: List[BirthDeathProcess], manuscript_data: pd.DataFrame, **fit_params: Any
) -> pd.DataFrame:
    """
    Compare multiple birth-death process models.

    Args:
        models: List of BirthDeathProcess instances to compare
        manuscript_data: Manuscript data for fitting
        **fit_params: Additional parameters for fitting

    Returns:
        DataFrame comparing model performance
    """
    results = []

    for i, model in enumerate(models):
        try:
            fit_result = model.fit(manuscript_data, **fit_params)

            if fit_result["success"]:
                results.append(
                    {
                        "model_id": i,
                        "birth_rate": fit_result["birth_rate"],
                        "death_rate": fit_result["death_rate"],
                        "log_likelihood": fit_result["log_likelihood"],
                        "aic": fit_result["aic"],
                        "bic": fit_result["bic"],
                        "n_params": 2,
                        "success": True,
                    }
                )
            else:
                results.append(
                    {
                        "model_id": i,
                        "birth_rate": np.nan,
                        "death_rate": np.nan,
                        "log_likelihood": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "n_params": 2,
                        "success": False,
                    }
                )

        except Exception as e:
            results.append(
                {
                    "model_id": i,
                    "birth_rate": np.nan,
                    "death_rate": np.nan,
                    "log_likelihood": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "n_params": 2,
                    "success": False,
                    "error": str(e),
                }
            )

    return pd.DataFrame(results)


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

    # Fit model
    model = BirthDeathProcess(birth_rate=1.0, death_rate=0.01)

    fit_results = model.fit(synthetic_data, age_column="age", count_column="count")

    if fit_results["success"]:
        print("\nFit Results:")
        print(f"  Estimated birth rate (λ): {model.birth_rate:.6f}")
        print(f"  Estimated death rate (μ): {model.death_rate:.6f}")
        print(f"  Log-likelihood: {fit_results['log_likelihood']:.2f}")
        print(f"  AIC: {fit_results['aic']:.2f}")
        print(f"  BIC: {fit_results['bic']:.2f}")

        # Calculate parameter errors
        birth_error = abs(model.birth_rate - true_birth_rate) / true_birth_rate * 100
        death_error = abs(model.death_rate - true_death_rate) / true_death_rate * 100

        print("\nParameter Estimation Errors:")
        print(f"  Birth rate error: {birth_error:.2f}%")
        print(f"  Death rate error: {death_error:.2f}%")

    else:
        print(f"\nFit failed: {fit_results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    # Run example analysis
    example_analysis()
