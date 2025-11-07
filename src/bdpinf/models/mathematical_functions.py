"""
Mathematical functions for birth-death process calculations.

This module contains pure mathematical functions for calculating birth-death
process probabilities and densities. These functions are isolated from the
main BirthDeathProcess class for better testability and modularity.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def calculate_population_log_likelihood(
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

    Formula:
    log P_0(T) = log(α_t)
    log P_n(T) = log(1 - α_t) + log(1 - β_t) + (n-1) * log(β_t)
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
        if n < 0:
            return np.float64(-np.inf)

        if time < 0:
            return np.float64(-np.inf)

        # Handle parameter scaling
        if log_scale_params:
            log_lambda = lambda_param
            log_mu = mu_param
            lambda_val = np.exp(lambda_param)
            mu_val = np.exp(mu_param)
        else:
            # Validate positive parameters
            if lambda_param <= 0 or mu_param <= 0:
                return np.float64(-np.inf)
            log_lambda = np.log(lambda_param)
            log_mu = np.log(mu_param)
            lambda_val = lambda_param
            mu_val = mu_param

        if abs(log_lambda - log_mu) < 1e-12:
            # Limiting case: λ ≈ μ
            rate_time_product = lambda_val * time
            alpha_t = beta_t = rate_time_product / (1 + rate_time_product)
        else:
            # General case: λ ≠ μ
            lambda_minus_mu = lambda_val - mu_val
            exponent = lambda_minus_mu * time
            exp_term = np.exp(exponent)
            if not np.isfinite(exp_term):
                return np.float64(-np.inf)
            denominator = lambda_val * exp_term - mu_val
            alpha_t = mu_val * (exp_term - 1) / denominator

            # Calculate β_t = (λ/μ) * α_t
            beta_t = (lambda_val / mu_val) * alpha_t

        # Validate α_t and β_t are in valid range (0, 1)
        if not ((0 < alpha_t < 1) and (0 < beta_t < 1)):
            return np.float64(-np.inf)

        if n == 0:
            return np.float64(np.log(alpha_t))

        # Calculate final log probability
        log_term1 = np.log(1 - alpha_t)
        log_term2 = np.log(1 - beta_t)
        log_term3 = (n - 1) * np.log(beta_t)

        # Check all terms are finite
        if not (
            np.isfinite(log_term1) and np.isfinite(log_term2) and np.isfinite(log_term3)
        ):
            return np.float64(-np.inf)

        log_probability = log_term1 + log_term2 + log_term3

        # Final numerical check
        if not np.isfinite(log_probability):
            return np.float64(-np.inf)

        return np.float64(log_probability)

    except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
        return np.float64(-np.inf)


def calculate_population_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    time: float,
    n: int,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of log age density f(t) w.r.t. λ and μ.

    Args:
        lambda_param: Birth rate (λ) > 0
        mu_param: Death rate (μ) > 0
        age_t: Age of the manuscript
        T: Total observation time >= age_t
        log_scale_params: If True, parameters are in log scale

    Returns:
        Tuple of derivatives (df/dλ, df/dμ)
    """

    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        lambda_val = lambda_param
        mu_val = mu_param
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)

    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ

        if n == 0:
            df_dlambda = 1 / (lambda_val + lambda_val**2 * time)
        else:
            numerator = n - 1 - 2 * time * lambda_val
            denominator = lambda_val * (1 + time * lambda_val)
            df_dlambda = numerator / denominator

        return np.array([df_dlambda, df_dlambda], dtype=np.float64)

    # General case: λ ≠ μ

    if n == 0:
        lambda_numerator = np.exp(time * (lambda_val - mu_val)) * (
            np.exp(time * (lambda_val - mu_val)) - lambda_val * time + mu_val * time - 1
        )
        lambda_denominator = (np.exp(time * (lambda_val - mu_val)) - 1) * (
            lambda_val * np.exp(time * (lambda_val - mu_val)) - mu_val
        )
        mu_numerator = np.exp(time * (lambda_val - mu_val)) * (
            lambda_val * (np.exp(time * (lambda_val - mu_val)) - mu_val * time - 1)
            + time * mu_val**2
        )
        mu_denominator = (
            mu_val
            * (np.exp(time * (lambda_val - mu_val)) - 1)
            * (lambda_val * np.exp(time * (lambda_val - mu_val)) - mu_val)
        )
    else:
        lambda_numerator_1 = mu_val * (
            mu_val
            - mu_val * n
            + lambda_val * (n - mu_val * time + 1)
            + lambda_val**2 * time
        )
        lambda_numerator_2 = np.exp(time * (lambda_val - mu_val)) * (
            n
            * (lambda_val - mu_val)
            * (lambda_val * time * (lambda_val - mu_val) - mu_val)
            - mu_val * (3 * lambda_val + mu_val)
        )
        lambda_numerator_3 = (
            lambda_val
            * np.exp(2 * time * (lambda_val - mu_val))
            * (2 * mu_val + lambda_val * time * (mu_val - lambda_val))
        )
        lambda_numerator = lambda_numerator_1 + lambda_numerator_2 + lambda_numerator_3
        lambda_denominator = (
            lambda_val
            * (lambda_val - mu_val)
            * (np.exp(time * (lambda_val - mu_val)) - 1)
            * (lambda_val * np.exp(time * (lambda_val - mu_val)) - mu_val)
        )

        mu_numerator_1 = (
            lambda_val**2
            * time
            * np.exp(time * (lambda_val - mu_val))
            * (np.exp(time * (lambda_val - mu_val)) - n)
        )
        mu_numerator_2 = -lambda_val * (
            -(2 * mu_val * n * time + n + 3) * np.exp(time * (lambda_val - mu_val))
            + n
            + (mu_val * time + 2) * np.exp(2 * time * (lambda_val - mu_val))
            + mu_val * time
            + 1
        )
        mu_numerator_3 = -mu_val * (
            (mu_val * n * time + n - 1) * np.exp(time * (lambda_val - mu_val))
            - n
            - mu_val * time
            + 1
        )
        mu_numerator = mu_numerator_1 + mu_numerator_2 + mu_numerator_3
        mu_denominator = (
            (lambda_val - mu_val)
            * (np.exp(time * (lambda_val - mu_val)) - 1)
            * (lambda_val * np.exp(time * (lambda_val - mu_val)) - mu_val)
        )

    df_dlambda = lambda_numerator / lambda_denominator
    df_dmu = mu_numerator / mu_denominator

    return np.array([df_dlambda, df_dmu], dtype=np.float64)


def calculate_spine_age_log_likelihood(
    lambda_param: float,
    mu_param: float,
    age_t: float,
    T: float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log age density f(t) for the special case of the spine individual.
    Mixture of continuous and discrete components.

    Args:
        lambda_param: Birth rate (λ) > 0
        mu_param: Death rate (μ) > 0
        age_t: Age of the manuscript
        T: Total observation time >= age_t
        log_scale_params: If True, parameters are in log scale
    Returns:
        Log age density f(t)
    """
    try:
        # Input validation
        if (age_t < 0) or (T < age_t):
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

        if abs(log_lambda - log_mu) < 1e-12:
            # Limiting case: λ ≈ μ
            if age_t == T:
                log_density = np.log(1 + lambda_val * T) - lambda_val * T
            else:
                log_density = (
                    2 * np.log(lambda_val) + np.log(age_t) - lambda_val * age_t
                )
            return np.float64(log_density)

        # General case: λ ≠ μ
        if age_t == T:
            numerator = lambda_val * np.exp(-mu_val * T) - mu_val * np.exp(
                -lambda_val * T
            )
            denominator = lambda_val - mu_val

            if not np.sign(numerator) == np.sign(denominator):
                return np.float64(-np.inf)

            log_numerator = np.log(np.abs(numerator))
            log_denominator = np.log(np.abs(denominator))
        else:
            numerator_1 = np.exp(-mu_val * T) - np.exp(-lambda_val * T)
            denominator = lambda_val - mu_val

            if not np.sign(numerator_1) == np.sign(denominator):
                return np.float64(-np.inf)
            log_numerator = (
                np.log(lambda_val) + np.log(mu_val) + np.log(np.abs(numerator_1))
            )
            log_denominator = np.log(np.abs(denominator))

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


def calculate_non_spine_age_log_likelihood(
    lambda_param: float,
    mu_param: float,
    age_t: float,
    T: float,
    log_scale_params: bool = False,
) -> np.float64:
    """
    Calculate log-likelihood of observing a manuscript with age t in a birth-death process.

    For non-spine individuals (manuscripts descended from the initial individual),
    the age density represents the probability density of observing a manuscript
    of age t conditional on survival to the observation time T.

    The age density is:
        f(t) = [λe^(-λt) - μe^(-(λ-μ)T)e^(-μt)] / [1 - e^(-(λ-μ)T)]

    When λ ≈ μ (limiting case), the density simplifies to:
        f(t) = [1 + λ(T-t)] / T * e^(-λt)

    All calculations are performed in log space for numerical stability.

    Args:
        lambda_param: Birth rate (λ > 0), or log(λ) if log_scale_params=True
        mu_param: Death rate (μ > 0), or log(μ) if log_scale_params=True
        age_t: Age of the manuscript (t ≥ 0)
        T: Total observation time from process start to present (T ≥ t)
        log_scale_params: If True, lambda_param and mu_param are log-transformed

    Returns:
        Log-likelihood value log(f(t)), or -inf if parameters are invalid

    Raises:
        Returns -inf for: negative ages, T < t, non-positive rates, or numerical overflow
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

            log_numerator = -lambda_val * age_t + np.log(1 + lambda_val * (T - age_t))
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
            numerator_term2 = mu_val * np.exp(-rate_diff_time) * np.exp(-mu_val * age_t)

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


def calculate_age_log_likelihood(
    lambda_param: float,
    mu_param: float,
    age_t: float,
    T: float,
    log_scale_params: bool = False,
    is_spine: bool = False,
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
        age_t: Age of the manuscript
        T: Total observation time >= age_t
        log_scale_params: If True, parameters are in log scale

    Returns:
        Log age density f(t)
    """
    if is_spine:
        return calculate_spine_age_log_likelihood(
            lambda_param, mu_param, age_t, T, log_scale_params
        )
    else:
        return calculate_non_spine_age_log_likelihood(
            lambda_param, mu_param, age_t, T, log_scale_params
        )


def calculate_non_spine_age_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    age_t: float,
    T: float,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of log age density f(t) w.r.t. λ and μ.

    Args:
        lambda_param: Birth rate (λ) > 0
        mu_param: Death rate (μ) > 0
        age_t: Age of the manuscript
        T: Total observation time >= age_t
        log_scale_params: If True, parameters are in log scale

    Returns:
        Tuple of derivatives (df/dλ, df/dμ)
    """

    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        lambda_val = lambda_param
        mu_val = mu_param
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)

    # Check for numerical stability in parameter space
    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ
        numerator = T + age_t**2 * lambda_val - age_t * (2 + lambda_val * T)
        denominator = 1 - age_t * lambda_val + lambda_val * T
        df_dlambda = numerator / denominator
        return np.array([df_dlambda, df_dlambda], dtype=np.float64)

    else:
        # General case: λ ≠ μ
        lambda_numerator_1 = (lambda_val * age_t - 1) * np.exp(
            mu_val * age_t + T * (lambda_val - mu_val)
        )
        lambda_numerator_2 = -(lambda_val * age_t + lambda_val * T - 1) * np.exp(
            mu_val * age_t + 2 * T * (lambda_val - mu_val)
        )
        lambda_numerator_3 = -mu_val * T * np.exp(lambda_val * age_t)
        lambda_numerator_4 = (
            2 * mu_val * T * np.exp(lambda_val * age_t + T * (lambda_val - mu_val))
        )
        lambda_numerator = (
            lambda_numerator_1
            + lambda_numerator_2
            + lambda_numerator_3
            + lambda_numerator_4
        )
        lambda_denominator = (np.exp(T * (lambda_val - mu_val)) - 1) * (
            lambda_val * np.exp(mu_val * age_t + T * (lambda_val - mu_val))
            - mu_val * np.exp(lambda_val * age_t)
        )
        mu_numerator_1 = (
            lambda_val * T * np.exp(mu_val * age_t + 2 * T * (lambda_val - mu_val))
        )
        mu_numerator_2 = np.exp(lambda_val * age_t) * (-mu_val * age_t + mu_val * T + 1)
        mu_numerator_3 = (mu_val * age_t - 2 * mu_val * T - 1) * np.exp(
            lambda_val * age_t + T * (lambda_val - mu_val)
        )
        mu_numerator = mu_numerator_1 + mu_numerator_2 + mu_numerator_3
        mu_denominator = lambda_denominator

    df_dlambda = lambda_numerator / lambda_denominator
    df_dmu = mu_numerator / mu_denominator

    return np.array([df_dlambda, df_dmu], dtype=np.float64)


def calculate_spine_age_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    age_t: float,
    T: float,
    log_scale_params: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of log age density f(t) w.r.t. λ and μ for spine individual.

    Args:
        lambda_param: Birth rate (λ) > 0
        mu_param: Death rate (μ) > 0
        age_t: Age of the manuscript
        T: Total observation time >= age_t
        log_scale_params: If True, parameters are in log scale
    Returns:
        Tuple of derivatives (df/dλ, df/dμ)
    """
    # Handle parameter scaling
    if log_scale_params:
        log_lambda = lambda_param
        log_mu = mu_param
        # Convert log parameters back to original scale
        lambda_val = np.exp(lambda_param)
        mu_val = np.exp(mu_param)
    else:
        lambda_val = lambda_param
        mu_val = mu_param
        log_lambda = np.log(lambda_param)
        log_mu = np.log(mu_param)

    pass

    # Check for numerical stability in parameter space
    if abs(log_lambda - log_mu) < 1e-12:
        # Limiting case: λ ≈ μ
        if age_t == T:
            # -((T^2 λ)/(1 + T λ))
            df_dlambda = -(lambda_val * T**2) / (1 + lambda_val * T)
        else:
            df_dlambda = 2 / lambda_val - age_t
        return np.array([df_dlambda, df_dlambda], dtype=np.float64)

    else:
        # General case: λ ≠ μ
        if age_t == T:
            # -(e^(λ T) (λ T + 1))/(μ e^(μ T) - λ e^(λ T)) - (λ T - μ T + 1)/(λ - μ)
            lambda_1 = (
                -np.exp(lambda_val * T)
                * (lambda_val * T + 1)
                / (mu_val * np.exp(mu_val * T) - lambda_val * np.exp(lambda_val * T))
            )
            lambda_2 = -(lambda_val * T - mu_val * T + 1) / (lambda_val - mu_val)
            df_dlambda = lambda_1 + lambda_2
            # -(λ (e^(λ T) (T (λ - μ) - 1) + e^(μ T)))/((λ - μ) (λ e^(λ T) - μ e^(μ T)))
            mu_numerator = -lambda_val * (
                np.exp(lambda_val * T) * (T * (lambda_val - mu_val) - 1)
                + np.exp(mu_val * T)
            )
            mu_denominator = (lambda_val - mu_val) * (
                lambda_val * np.exp(lambda_val * T) - mu_val * np.exp(mu_val * T)
            )
            df_dmu = mu_numerator / mu_denominator
        else:
            # 1/(μ - λ) + 1/λ + (T e^(μ T))/(e^(λ T) - e^(μ T))
            lambda_1 = 1 / (mu_val - lambda_val) + 1 / lambda_val
            lambda_2 = (T * np.exp(mu_val * T)) / (
                np.exp(lambda_val * T) - np.exp(mu_val * T)
            )
            df_dlambda = lambda_1 + lambda_2
            # λ/(μ (λ - μ)) + (T e^(λ T))/(e^(μ T) - e^(λ T))
            mu_1 = lambda_val / (mu_val * (lambda_val - mu_val))
            mu_2 = (T * np.exp(lambda_val * T)) / (
                np.exp(mu_val * T) - np.exp(lambda_val * T)
            )
            df_dmu = mu_1 + mu_2

    return np.array([df_dlambda, df_dmu], dtype=np.float64)


def calculate_age_log_likelihood_derivatives(
    lambda_param: float,
    mu_param: float,
    age_t: float,
    T: float,
    log_scale_params: bool = False,
    is_spine: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate derivatives of log age density f(t) w.r.t. λ and μ.

    Args:
        lambda_param: Birth rate (λ) > 0
        mu_param: Death rate (μ) > 0
        age_t: Age of the manuscript
        T: Total observation time >= age_t
        log_scale_params: If True, parameters are in log scale
        is_spine: If True, calculate for spine individual
    Returns:
        Tuple of derivatives (df/dλ, df/dμ)
    """
    if is_spine:
        return calculate_spine_age_log_likelihood_derivatives(
            lambda_param, mu_param, age_t, T, log_scale_params
        )
    else:
        return calculate_non_spine_age_log_likelihood_derivatives(
            lambda_param, mu_param, age_t, T, log_scale_params
        )


def predict_manuscript_survival(
    birth_rate: float,
    death_rate: float,
    ages: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Predict expected manuscript survival probability for given ages.

    This is a simple exponential survival function based on the death rate.

    Args:
        birth_rate: Birth rate parameter (used for scaling)
        death_rate: Death rate parameter
        ages: Age or array of ages to predict for

    Returns:
        Predicted survival probability or array of probabilities
    """
    if isinstance(ages, (int, float)):
        ages = np.array([ages], dtype=np.float64)

    # Convert to numpy array if not already
    ages_array = np.asarray(ages, dtype=np.float64)

    # Simple exponential survival model
    survival_probs = birth_rate * np.exp(-death_rate * ages_array)

    # Return scalar if input was scalar
    if len(survival_probs) == 1 and isinstance(ages, (int, float)):
        return float(survival_probs[0])

    return survival_probs
