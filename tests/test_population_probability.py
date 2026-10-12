"""Test cases for calculate_populations_log_lik based on the Arley & Borchsenius formula."""

import math
from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from bdpinf.inference.likelihood import log_likelihood
from bdpinf.models import (
    calculate_populations_log_lik,
    calculate_populations_log_pmf,
)
from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.process import BirthDeathProcess


def _exact_log_pmf(lam: float, mu: float, t: float, n: int) -> float:
    """Log P(Z_T = n | Z_0 = 1) = log[(1 - alpha) pi (1 - pi)^(n - 1)], 80 digits."""
    getcontext().prec = 80
    lam_d, mu_d, t_d = Decimal(lam), Decimal(mu), Decimal(t)
    gamma = lam_d - mu_d
    w = (lam_d * (gamma * t_d).exp() - mu_d) / gamma
    pi = 1 / w
    n_t = (-gamma * t_d).exp() * w - 1
    alpha = n_t / (1 + n_t)
    return float(((1 - alpha) * pi * (1 - pi) ** (n - 1)).ln())


class TestPopulationProbability:
    """Test suite for population probability calculations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

    def test_input_validation(self) -> None:
        """Test that invalid inputs return -inf."""
        # Negative birth rate
        with pytest.raises(ValueError):
            calculate_populations_log_lik(-0.1, 0.05, times=100.0, ns=10)

        # Negative death rate
        with pytest.raises(ValueError):
            calculate_populations_log_lik(0.1, -0.05, times=100.0, ns=10)

        # Zero death rate
        result = calculate_populations_log_lik(0.1, 0.0, times=100.0, ns=10)
        assert math.isfinite(float(result))

        # Zero birth rate
        result = calculate_populations_log_lik(0.0, 0.1, times=100.0, ns=10)
        assert float(result) == float("-inf")

        # Negative time
        with pytest.raises(ValueError):
            calculate_populations_log_lik(0.1, 0.05, times=-100.0, ns=10)

        with pytest.raises(ValueError):
            calculate_populations_log_lik(0.1, 0.05, times=0.0, ns=10)

    def test_zero_birth_rate_n_zero(self) -> None:
        """Zero birth rate with n=0 stays finite."""
        result = calculate_populations_log_lik(0.0, 0.05, times=100.0, ns=0)
        assert math.isfinite(float(result))
        assert_allclose(
            float(result),
            float(calculate_extinction_pr(0.0, 0.05, 100.0, return_log_prob=True)),
        )

    def test_zero_birth_rate_n_one(self) -> None:
        """Zero birth rate with n=1 stays finite."""
        result = calculate_populations_log_lik(0.0, 0.05, times=100.0, ns=1)
        assert math.isfinite(float(result))
        ext_pr = float(calculate_extinction_pr(0.0, 0.05, 100.0))
        assert_allclose(float(result), math.log1p(-ext_pr))

    def test_zero_birth_rate_high_n(self) -> None:
        """Zero birth rate with n>1 returns -inf."""
        result = calculate_populations_log_lik(0.0, 0.05, times=100.0, ns=10)
        assert float(result) == float("-inf")

    def test_unbias_positive_count_uses_geometric_mass(self) -> None:
        """Unbiased positive counts follow the conditioned geometric mass."""
        birth_rate, death_rate, time, n = 0.1, 0.05, 100.0, 5
        alpha_t = float(
            calculate_extinction_pr(
                birth_rate,
                death_rate,
                time,
                return_log_prob=False,
            )
        )
        beta_t = (birth_rate / death_rate) * alpha_t

        result = calculate_populations_log_lik(
            birth_rate,
            death_rate,
            times=time,
            ns=n,
            unbias=True,
        )

        expected = math.log1p(-beta_t) + (n - 1) * math.log(beta_t)
        assert_allclose(float(result), expected)

    def test_finite_results(self) -> None:
        """Test that valid inputs produce finite results."""
        test_cases = [
            (0.01, 0.001, 1000.0, 50),
            (0.1, 0.05, 100.0, 25),
        ]

        for birth_rate, death_rate, time, n in test_cases:
            result = calculate_populations_log_lik(
                birth_rate,
                death_rate,
                times=time,
                ns=n,
            )
            assert math.isfinite(float(result)), (
                f"Non-finite result for λ={birth_rate}, μ={death_rate}, T={time}, n={n}"
            )

    def test_nearly_equal_rates(self) -> None:
        """Test the special case where λ ≈ μ."""
        # Test case where rates are very close
        birth_rate = 1.0
        death_rate = 0.9999999999  # Very close to λ
        time = 100.0
        n = 10

        result = calculate_populations_log_lik(birth_rate, death_rate, times=time, ns=n)

        # Should not return -inf due to the perturbation handling
        assert math.isfinite(float(result)), (
            "Nearly equal rates should be handled with perturbation"
        )

    def test_population_size_effect(self) -> None:
        """Test that larger populations generally have lower probabilities."""
        birth_rate, death_rate, time = 0.1, 0.05, 100.0

        results = []
        for n in [1, 5, 10, 20, 50]:
            result = calculate_populations_log_lik(
                birth_rate,
                death_rate,
                times=time,
                ns=n,
            )
            results.append((n, float(result)))
            assert math.isfinite(float(result)), f"Non-finite result for n={n}"

        # Check that probabilities generally decrease with larger n
        # (allowing for some numerical noise)
        for i in range(len(results) - 1):
            n1, prob1 = results[i]
            n2, prob2 = results[i + 1]
            assert prob2 <= prob1 + 1e-10, (
                f"Probability should decrease: P_{n1}={prob1}, P_{n2}={prob2}"
            )

    def test_time_effect(self) -> None:
        """Test the effect of different observation times."""
        birth_rate, death_rate, n = 0.1, 0.05, 10

        results = []
        # Use shorter times to avoid beta_t = 1 boundary issues
        for time in [50.0, 100.0, 200.0, 500.0]:  # Removed 1000 which causes beta_t = 1
            result = calculate_populations_log_lik(
                birth_rate,
                death_rate,
                times=time,
                ns=n,
            )
            results.append((time, float(result)))
            assert math.isfinite(float(result)), f"Non-finite result for T={time}"

        # Results should be finite for all times
        assert all(math.isfinite(prob) for _, prob in results)

    def test_alpha_beta_bounds(self) -> None:
        """Test that α_t and β_t stay within valid bounds [0, 1]."""
        # We'll test this indirectly by ensuring the function doesn't return -inf
        # when it should produce valid results

        test_cases = [
            (0.1, 0.05, 100.0, 5),  # λ > μ
            (0.05, 0.1, 100.0, 5),  # λ < μ
            (1.0, 0.9, 50.0, 3),  # Close rates
            (0.01, 0.005, 1000.0, 20),  # Small rates, long time
        ]

        for birth_rate, death_rate, time, n in test_cases:
            result = calculate_populations_log_lik(
                birth_rate,
                death_rate,
                times=time,
                ns=n,
            )
            # If the function returns -inf,
            # it means alpha_t or beta_t went out of bounds
            # For these reasonable parameters, we expect finite results
            assert math.isfinite(float(result)), (
                f"Expected finite result for λ={birth_rate}, μ={death_rate}, "
                f"T={time}, n={n}, got {result}"
            )

    def test_formula_consistency(self) -> None:
        """Test that our implementation matches manual calculation."""
        birth_rate, death_rate, time, n = 0.1, 0.05, 100.0, 5

        # Manual calculation following the same steps
        alpha_t = float(
            calculate_extinction_pr(
                birth_rate,
                death_rate,
                time,
                return_log_prob=False,
            )
        )
        beta_t = (birth_rate / death_rate) * alpha_t

        manual_result = (
            math.log(1 - alpha_t) + math.log(1 - beta_t) + (n - 1) * math.log(beta_t)
        )

        # Function result
        function_result = float(
            calculate_populations_log_lik(
                birth_rate,
                death_rate,
                times=time,
                ns=n,
            )
        )

        assert abs(manual_result - function_result) < 1e-12, (
            f"Manual and function results should match: "
            f"manual={manual_result}, function={function_result}"
        )

    def test_initial_population_gt_one(self) -> None:
        """Validate the n0 > 1 branch against the closed-form summation."""

        def davison_formula(a: int, k: int, alpha_t: float, beta_t: float) -> float:
            """Calculate the probability using Davison's formula.

            Parameters
            ----------
            a: int
                initial population size (n0)
            k: int
                observed population size (n)
            alpha_t: float
                extinction probability at time t
            beta_t: float
                (λ/μ) * alpha_t

            """
            terms: list[float] = []
            for j in range(max(0, a - k), a):
                term = (
                    math.comb(a, j)
                    * math.comb(k - 1, a - j - 1)
                    * (alpha_t**j)
                    * (beta_t ** (k - a + j))
                    * ((1 - alpha_t - beta_t + alpha_t * beta_t) ** (a - j))
                )
                terms.append(term)
            return sum(terms)

        def tavare_formula(n: int, m: int, alpha_t: float, beta_t: float) -> float:
            """Calculate the probability using Tavaré's formula.

            Parameters
            ----------
            n: int
                initial population size (n0)
            m: int
                observed population size (n)
            alpha_t: float
                extinction probability at time t
            beta_t: float
                (λ/μ) * alpha_t

            """
            terms: list[float] = []
            for j in range(1, min(n, m) + 1):
                term = (
                    math.comb(n, j)
                    * math.comb(m - 1, j - 1)
                    * (alpha_t ** (n - j))
                    * (beta_t ** (m - j))
                    * ((1 - alpha_t - beta_t + alpha_t * beta_t) ** j)
                )
                terms.append(term)
            return sum(terms)

        birth_rate, death_rate, time = 0.2, 0.1, 1.0
        n0 = 5
        n = 4

        alpha_t = float(
            calculate_extinction_pr(
                birth_rate,
                death_rate,
                time,
                return_log_prob=False,
            )
        )

        beta_t = (birth_rate / death_rate) * alpha_t

        davison_result = davison_formula(n0, n, alpha_t, beta_t)
        tavare_result = tavare_formula(n0, n, alpha_t, beta_t)

        my_result = math.exp(
            float(
                calculate_populations_log_lik(
                    birth_rate,
                    death_rate,
                    times=time,
                    ns=n,
                    initial_population=n0,
                )
            ),
        )

        assert_allclose(my_result, davison_result)
        assert_allclose(my_result, tavare_result)
        assert_allclose(tavare_result, davison_result)

    def test_extreme_parameters(self) -> None:
        """Test behavior with extreme but valid parameters."""
        # Very small rates
        result = calculate_populations_log_lik(1e-6, 1e-7, times=1000.0, ns=5)
        assert math.isfinite(float(result)), "Should handle very small rates"

        # Large time
        result = calculate_populations_log_lik(0.1, 0.05, times=10000.0, ns=5)
        # This might return -inf due to numerical overflow, which is acceptable
        v = float(result)
        assert v == float("-inf") or math.isfinite(v), (
            "Large time should either overflow or be finite"
        )

        # Large population
        result = calculate_populations_log_lik(0.1, 0.05, times=100.0, ns=1000)
        v = float(result)
        assert v == float("-inf") or math.isfinite(v), (
            "Large n should either be very small or finite"
        )

    def test_n_equals_one_special_case(self) -> None:
        """Test the special case where n=1."""
        # When n=1, the (n-1)*log(beta_t) term becomes 0
        birth_rate, death_rate, time = 0.1, 0.05, 100.0

        result = calculate_populations_log_lik(birth_rate, death_rate, times=time, ns=1)

        assert math.isfinite(float(result)), "n=1 case should produce finite result"

        # Manual verification for n=1
        alpha_t = float(
            calculate_extinction_pr(
                birth_rate,
                death_rate,
                time,
                return_log_prob=False,
            )
        )
        beta_t = (birth_rate / death_rate) * alpha_t

        expected = math.log(1 - alpha_t) + math.log(
            1 - beta_t,
        )  # (n-1)*log(beta_t) = 0 for n=1

        assert abs(float(result) - expected) < 1e-12, (
            f"n=1 case mismatch: expected={expected}, got={float(result)}"
        )

    @pytest.mark.parametrize(
        "birth_rate,death_rate,time,n",
        [
            (0.1, 0.05, 100.0, 5),
            # (0.5, 0.3, 200.0, 10),  # This leads to beta_t = 1, excluded
            (1.0, 0.8, 50.0, 3),
            (0.01, 0.008, 1000.0, 15),
            (2.0, 1.5, 30.0, 7),
        ],
    )
    def test_parametrized_valid_cases(self, birth_rate, death_rate, time, n):
        """Test multiple valid parameter combinations."""
        result = calculate_populations_log_lik(birth_rate, death_rate, times=time, ns=n)

        assert math.isfinite(float(result)), (
            f"Expected finite result for λ={birth_rate}, μ={death_rate}, "
            f"T={time}, n={n}"
        )

        # For birth-death processes, log probabilities should be negative
        assert float(result) <= 0, (
            f"Log probability should be non-positive, got {result}"
        )

    def test_numerical_stability(self) -> None:
        """Test numerical stability edge cases."""
        # Case where exp_term might be very large
        birth_rate, death_rate = 1.0, 0.1  # λ >> μ
        time = 10.0  # Moderate time
        n = 5

        result = calculate_populations_log_lik(birth_rate, death_rate, times=time, ns=n)

        # Should either be finite or -inf (due to overflow), but not NaN
        v = float(result)
        assert v == float("-inf") or math.isfinite(v), (
            "Result should be finite or -inf, not NaN"
        )
        assert not math.isnan(v), "Result should not be NaN"

    def test_boundary_conditions(self) -> None:
        """Test conditions that might cause boundary issues."""
        # Test case where alpha_t or beta_t might approach 1
        # This is a challenging case numerically

        # When μ is very small compared to λ, beta_t approaches λ/λ = 1
        birth_rate = 1.0
        death_rate = 1e-10  # Very small
        time = 1.0  # Small time to avoid other numerical issues
        n = 2

        result = calculate_populations_log_lik(birth_rate, death_rate, times=time, ns=n)

        # This case might legitimately return -inf due to boundary conditions
        v = float(result)
        assert v == float("-inf") or math.isfinite(v), (
            "Boundary case should be handled gracefully"
        )

    def test_extreme_survival_cases_are_finite_and_exact(self) -> None:
        """pi(T) underflows in linear space here, yet the log pmf is finite and exact.

        The earlier implementation formed ``1 - (1 - pi)`` and returned ``-inf``
        for these cases; working with ``log pi = -log W(T)`` directly gives the
        correct value, compared against a 80-digit decimal evaluation.
        """
        boundary_cases = [
            (1.0, 0.5, 100.0, 10, "λ = 2μ case"),
            (0.1, 0.05, 1000.0, 10, "Long time case"),
            (0.5, 0.3, 200.0, 10, "Moderate ratio case"),
        ]

        for birth_rate, death_rate, time, n, description in boundary_cases:
            result = calculate_populations_log_lik(
                birth_rate,
                death_rate,
                times=time,
                ns=n,
            )
            assert float(result) == pytest.approx(
                _exact_log_pmf(birth_rate, death_rate, time, n), rel=1e-12
            ), description


class TestPopulationsLogPmf:
    """Test calculate_populations_log_pmf vectorised PMF."""

    def test_pmf_matches_lik_elementwise(self) -> None:
        """Vectorised PMF must equal element-wise calculate_populations_log_lik."""
        birth_rate, death_rate, time = 0.1, 0.05, 100.0
        ns = np.array([1, 3, 5, 10, 20], dtype=np.float64)

        pmf_vec = np.asarray(
            calculate_populations_log_pmf(
                birth_rates=jnp.array([birth_rate]),
                death_rates=jnp.array([death_rate]),
                ns=jnp.array(ns),
                times=jnp.array([time]),
            )
        )

        for i, n in enumerate(ns):
            expected = float(
                calculate_populations_log_lik(
                    birth_rate, death_rate, times=time, ns=int(n)
                )
            )
            assert_allclose(
                pmf_vec[i], expected, rtol=1e-10, err_msg=f"PMF mismatch at n={n}"
            )

    def test_pmf_sums_to_one(self) -> None:
        """Probabilities over a wide support should sum close to 1."""
        birth_rate, death_rate, time = (
            0.05,
            0.1,
            50.0,
        )  # subcritical so mass concentrates
        ns = np.arange(0, 200, dtype=np.float64)

        log_pmf = np.asarray(
            calculate_populations_log_pmf(
                birth_rates=jnp.array([birth_rate]),
                death_rates=jnp.array([death_rate]),
                ns=jnp.array(ns),
                times=jnp.array([time]),
            )
        )
        total = float(np.sum(np.exp(log_pmf)))
        assert abs(total - 1.0) < 0.01, f"PMF should sum ≈ 1, got {total}"

    def test_pmf_n_zero_equals_extinction(self) -> None:
        """P(Z_T = 0) must equal the extinction probability."""
        birth_rate, death_rate, time = 0.05, 0.1, 50.0
        log_pmf = calculate_populations_log_pmf(
            birth_rates=jnp.array([birth_rate]),
            death_rates=jnp.array([death_rate]),
            ns=jnp.array([0.0]),
            times=jnp.array([time]),
        )
        expected_log_alpha = float(
            calculate_extinction_pr(birth_rate, death_rate, time, return_log_prob=True)
        )
        assert_allclose(float(log_pmf[0]), expected_log_alpha, rtol=1e-10)

    def test_pmf_skip_normalization_matches_normal(self) -> None:
        """_skip_normalization=True must return the same values as the default path."""
        birth_rates = jnp.array([0.1])
        death_rates = jnp.array([0.05])
        ns = jnp.array([1.0, 5.0, 10.0])
        times = jnp.array([100.0])

        normal = np.asarray(
            calculate_populations_log_pmf(birth_rates, death_rates, ns, times)
        )
        skipped = np.asarray(
            calculate_populations_log_pmf(
                birth_rates, death_rates, ns, times, _skip_normalization=True
            )
        )
        assert_allclose(normal, skipped, rtol=1e-12)

    def test_pmf_initial_population_gt_one_matches_log_lik(self) -> None:
        """Vectorised n0>1 path must match the per-element formula, incl. n=0."""
        birth_rate, death_rate, time = 0.2, 0.1, 1.0
        n0 = 5
        ns = np.array([0.0, 1.0, 4.0, 5.0], dtype=np.float64)

        pmf_vec = np.asarray(
            calculate_populations_log_pmf(
                birth_rates=jnp.array([birth_rate]),
                death_rates=jnp.array([death_rate]),
                ns=jnp.array(ns),
                times=jnp.array([time]),
                initial_population=n0,
            )
        )

        for i, n in enumerate(ns):
            expected = float(
                calculate_populations_log_lik(
                    birth_rate, death_rate, times=time, ns=int(n), initial_population=n0
                )
            )
            assert_allclose(
                pmf_vec[i], expected, rtol=1e-10, err_msg=f"PMF mismatch at n={n}"
            )

    def test_pmf_initial_population_gt_one_is_jittable(self) -> None:
        """Regression: the n0>1 path must trace under jax.jit/make_jaxpr.

        Previously broke tracing via per-element int()/bool() materialization
        in a Python loop.
        """
        birth_rates = jnp.array([0.2])
        death_rates = jnp.array([0.1])
        ns = jnp.array([0.0, 1.0, 4.0, 5.0])
        times = jnp.array([1.0])

        def fn(
            b: jnp.ndarray, d: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray
        ) -> jnp.ndarray:
            return calculate_populations_log_pmf(
                b, d, n, t, initial_population=5, _skip_normalization=True
            )

        jaxpr = jax.make_jaxpr(fn)(birth_rates, death_rates, ns, times)
        assert jaxpr is not None

        jitted = np.asarray(jax.jit(fn)(birth_rates, death_rates, ns, times))
        eager = np.asarray(fn(birth_rates, death_rates, ns, times))
        assert_allclose(jitted, eager, rtol=1e-12)


class TestPopulationProbabilityIntegration:
    """Integration tests with the full BirthDeathProcess class."""

    def test_integration_with_log_likelihood(self) -> None:
        """Test that population probability works correctly in log-likelihood calculation."""
        model = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            epoch_end_times=1000.0,
        )

        # Set up a simple individual scenario
        creation_times = [1000.0, 1200.0, 1400.0]  # Three records
        model.creation_times = creation_times
        model.set_epoch_end_times(1500.0)

        # Test that log-likelihood calculation doesn't fail using modular function
        ages = model.ages
        log_likelihood_value = log_likelihood(
            birth_rates=0.1,
            death_rates=0.05,
            ages=ages,
            times=1500.0,
        )

        # Should be finite or -inf, but not NaN
        assert not math.isnan(float(log_likelihood_value)), (
            "Log-likelihood should not be NaN"
        )

    def test_population_probability_in_context(self) -> None:
        """Test population probability calculation in realistic individual context."""
        # Realistic medieval individual parameters
        birth_rate = 0.01  # 1 individual per 100 years
        death_rate = 0.001  # 0.1% death rate per year
        time = 1000.0  # 1000 years of observation
        n = 50  # 50 surviving records

        result = calculate_populations_log_lik(birth_rate, death_rate, times=time, ns=n)

        assert math.isfinite(float(result)), (
            "Realistic individual parameters should give finite result"
        )
        assert float(result) <= 0, "Log probability should be non-positive"
