"""
Inference module for manuscript data analysis using birth-death processes.

This module provides functionality to fit birth-death process models to manuscript
datasets, estimating birth and death rates based on manuscript counts and ages.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize


class BirthDeathProcess:
    """
    Birth-death process model for manuscript survival and creation.

    This class models manuscript data as a birth-death process where:
    - Birth rate (λ): Rate of manuscript creation
    - Death rate (μ): Rate of manuscript loss/destruction
    - Observed data: Number of surviving manuscripts and their ages
    """

    def __init__(self, birth_rate: float = 1.0, death_rate: float = 0.1):
        """
        Initialize birth-death process with initial parameter estimates.

        Args:
            birth_rate: Initial estimate of manuscript creation rate (λ)
            death_rate: Initial estimate of manuscript loss rate (μ)
        """
        self.birth_rate: float = birth_rate
        self.death_rate: float = death_rate
        self.fitted = False
        self.fit_results: Optional[Dict[str, Any]] = None

    def survival_probability(self, age: float) -> Any:
        """
        Calculate survival probability for a manuscript of given age.

        For a birth-death process, the survival probability is:
        P(survival | age) = exp(-μ * age)

        Args:
            age: Age of manuscript in years

        Returns:
            Probability of survival
        """
        return np.exp(-self.death_rate * age)

    def expected_manuscripts(self, age: float, time_window: float = 1.0) -> Any:
        """
        Calculate expected number of manuscripts created in a time window
        that survive to a given age.

        Args:
            age: Age of manuscripts
            time_window: Time window for manuscript creation

        Returns:
            Expected number of surviving manuscripts
        """
        return self.birth_rate * time_window * self.survival_probability(age)

    def log_likelihood(
        self, params: np.ndarray, ages: np.ndarray, counts: np.ndarray
    ) -> Any:
        """
        Calculate log-likelihood for observed manuscript data.

        This is a placeholder for the likelihood function. Users can override
        this method with their specific likelihood formulation.

        Args:
            params: Parameters [birth_rate, death_rate]
            ages: Array of manuscript ages
            counts: Array of manuscript counts for each age

        Returns:
            Log-likelihood value
        """
        birth_rate, death_rate = params
        if birth_rate <= 0 or death_rate < 0:
            return -np.inf

        # Simple Poisson likelihood as placeholder
        # This assumes each age bin follows Poisson distribution
        expected_counts = birth_rate * np.exp(-death_rate * ages)

        # Handle zero expected counts
        expected_counts = np.maximum(expected_counts, 1e-10)

        # Calculate Poisson log-likelihood
        log_like = np.sum(counts * np.log(expected_counts) - expected_counts)

        return log_like

    def negative_log_likelihood(
        self, params: np.ndarray, ages: np.ndarray, counts: np.ndarray
    ) -> float:
        """
        Negative log-likelihood for optimization (minimization).

        Args:
            params: Parameters [birth_rate, death_rate]
            ages: Array of manuscript ages
            counts: Array of manuscript counts for each age

        Returns:
            Negative log-likelihood value
        """
        result = -self.log_likelihood(params, ages, counts)
        return float(result)  # Explicit conversion to float

    def fit(
        self,
        manuscript_data: pd.DataFrame,
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
            manuscript_data: DataFrame with manuscript information
            age_column: Column name containing manuscript ages
            count_column: Column name for counts (if None, assumes one manuscript per row)
            method: Optimization method for scipy.optimize.minimize
            initial_params: Initial parameter values (birth_rate, death_rate)

        Returns:
            Dictionary with fit results including parameters and statistics
        """
        # Prepare data
        if count_column is None:
            # Assume one manuscript per row, bin by age
            ages = manuscript_data[age_column].values
            # Convert to numpy array to ensure proper typing
            ages_array = np.asarray(ages, dtype=float)
            age_bins = np.arange(0, float(ages_array.max()) + 50, 50)  # 50-year bins
            counts, bin_edges = np.histogram(ages_array, bins=age_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            # Use provided counts
            bin_centers = manuscript_data[age_column].values
            counts_values = manuscript_data[count_column].values
            counts = np.asarray(counts_values)

        # Remove empty bins - convert to numpy arrays for type safety
        counts_array = np.asarray(counts)
        bin_centers_array = np.asarray(bin_centers)
        non_zero_mask = counts_array > 0
        ages_fit = bin_centers_array[non_zero_mask]
        counts_fit = counts_array[non_zero_mask]

        if len(ages_fit) == 0:
            raise ValueError("No valid data points for fitting")

        # Set initial parameters
        if initial_params is None:
            initial_params = (self.birth_rate, self.death_rate)

        # Parameter bounds: birth_rate > 0, death_rate >= 0
        bounds: List[Tuple[Optional[float], Optional[float]]] = [
            (1e-6, None),
            (0, None),
        ]

        # Optimize
        try:
            # Ensure parameters are numpy arrays for scipy compatibility
            x0 = np.array(initial_params, dtype=np.float64)

            result = optimize.minimize(
                self.negative_log_likelihood,
                x0,
                args=(ages_fit, counts_fit),
                method=method,
                bounds=bounds,
            )

            if result.success:
                self.birth_rate, self.death_rate = result.x
                self.fitted = True

                # Calculate additional statistics
                log_like = self.log_likelihood(result.x, ages_fit, counts_fit)
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

        except Exception as e:
            self.fit_results = {"success": False, "error": str(e)}

        return self.fit_results

    def predict(self, ages: np.ndarray) -> np.ndarray:
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
        max_age: float = 2000,
        time_step: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Simulate manuscript data from the birth-death process.

        Args:
            max_age: Maximum age to simulate
            time_step: Time step for simulation
            random_seed: Random seed for reproducibility

        Returns:
            DataFrame with simulated manuscript data
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        ages = np.arange(0, max_age, time_step)
        expected_counts = self.predict(ages)
        simulated_counts = np.random.poisson(expected_counts)

        return pd.DataFrame(
            {
                "age": ages,
                "expected_count": expected_counts,
                "simulated_count": simulated_counts,
            }
        )


def prepare_manuscript_data(
    df: pd.DataFrame, current_year: int = 2025, date_column: str = "midYear"
) -> pd.DataFrame:
    """
    Prepare manuscript data for birth-death process analysis.

    Args:
        df: DataFrame with manuscript data
        current_year: Current year for age calculation
        date_column: Column containing manuscript dates

    Returns:
        DataFrame with age calculations and binned data
    """
    # Calculate ages
    manuscript_data = df.copy()
    manuscript_data["age"] = current_year - manuscript_data[date_column]

    # Remove manuscripts with negative ages (future dates)
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
    age_range: np.ndarray = np.linspace(0, ages_fit.max(), 100)
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

    # Create synthetic data for demonstration
    np.random.seed(42)
    true_birth_rate = 0.5
    true_death_rate = 0.001

    ages = np.linspace(10, 1500, 30)
    true_counts = true_birth_rate * np.exp(-true_death_rate * ages)
    observed_counts = np.random.poisson(true_counts)

    synthetic_data = pd.DataFrame(
        {"age": ages, "count": observed_counts, "midYear": 2025 - ages}
    )

    print("Synthetic data created with true parameters:")
    print(f"  Birth rate (λ): {true_birth_rate}")
    print(f"  Death rate (μ): {true_death_rate}")

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
