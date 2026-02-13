from typing import Dict

import pytest
import torch
from numpy.testing import assert_allclose

from inf_bp_texts.models.mathematical_derivatives import (
    calculate_ages_log_lik_derivatives,
    calculate_non_spine_ages_log_lik_derivatives,
    calculate_populations_log_lik_derivatives,
    calculate_spine_ages_log_lik_derivatives,
)
from inf_bp_texts.models.mathematical_functions import (
    calculate_ages_log_lik,
    calculate_non_spine_ages_log_lik,
    calculate_populations_log_lik,
    calculate_spine_ages_log_lik,
)


@pytest.fixture
def standard_scalar_params() -> Dict[str, torch.Tensor]:
    """Standard scalar parameters for testing."""
    return {
        "lambda_param": torch.tensor([0.1], dtype=torch.float64),
        "mu_param": torch.tensor([0.05], dtype=torch.float64),
        "time": torch.tensor([100.0], dtype=torch.float64),
    }


@pytest.fixture
def nearly_equal_params() -> Dict[str, torch.Tensor]:
    """Parameters where lambda is approximately mu (numerical stability test)."""
    return {
        "lambda_param": torch.tensor([0.1], dtype=torch.float64),
        "mu_param": torch.tensor([0.1 + 1e-13], dtype=torch.float64),
        "time": torch.tensor([100.0], dtype=torch.float64),
    }


@pytest.fixture
def sample_ages_scalar() -> float:
    """Single age value for testing."""
    return 50.0


@pytest.fixture
def sample_ages_array() -> torch.Tensor:
    """Array of ages for vectorized testing."""
    return torch.tensor([10.0, 30.0, 50.0, 70.0, 90.0], dtype=torch.float64)


@pytest.fixture
def sample_population_counts() -> torch.Tensor:
    """Sample population counts (n values)."""
    return torch.tensor([0, 1, 5, 10, 20], dtype=torch.int64)


class TestPopulationsLogLikelihoodDerivatives:
    """Test derivatives of population log-likelihood."""

    def test_basic_derivatives(
        self, standard_scalar_params: Dict[str, torch.Tensor]
    ) -> None:
        result = calculate_populations_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_derivatives_with_n_zero(
        self, standard_scalar_params: Dict[str, torch.Tensor]
    ) -> None:
        result = calculate_populations_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=torch.tensor([0], dtype=torch.int64),
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_derivatives_array_ns(
        self,
        standard_scalar_params: Dict[str, torch.Tensor],
        sample_population_counts: torch.Tensor,
    ) -> None:
        result = calculate_populations_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            standard_scalar_params["time"],
            ns=sample_population_counts,
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_derivatives_critical_case(
        self, nearly_equal_params: Dict[str, torch.Tensor]
    ) -> None:
        result = calculate_populations_log_lik_derivatives(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            nearly_equal_params["time"],
            ns=torch.tensor([1], dtype=torch.int64),
        )

        assert torch.all(torch.isfinite(result))


class TestSpineAgesLogLikelihoodDerivatives:
    """Test derivatives for spine age log-likelihood."""

    def test_scalar_age_derivatives(
        self, standard_scalar_params: Dict[str, torch.Tensor], sample_ages_scalar: float
    ) -> None:
        result = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_array_ages_derivatives(
        self,
        standard_scalar_params: Dict[str, torch.Tensor],
        sample_ages_array: torch.Tensor,
    ) -> None:
        result = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_derivatives_at_boundary(
        self, standard_scalar_params: Dict[str, torch.Tensor]
    ) -> None:
        T = standard_scalar_params["time"]
        result = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            T,
            T,
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_derivatives_critical_case(
        self, nearly_equal_params: Dict[str, torch.Tensor], sample_ages_scalar: float
    ) -> None:
        result = calculate_spine_ages_log_lik_derivatives(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            sample_ages_scalar,
            nearly_equal_params["time"],
        )

        assert torch.all(torch.isfinite(result))


class TestNonSpineAgesLogLikelihoodDerivatives:
    """Test derivatives for non-spine age log-likelihood."""

    def test_scalar_age_derivatives(
        self, standard_scalar_params: Dict[str, torch.Tensor], sample_ages_scalar: float
    ) -> None:
        result = calculate_non_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_array_ages_derivatives(
        self,
        standard_scalar_params: Dict[str, torch.Tensor],
        sample_ages_array: torch.Tensor,
    ) -> None:
        result = calculate_non_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_array,
            standard_scalar_params["time"],
        )

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))

    def test_derivatives_critical_case(
        self, nearly_equal_params: Dict[str, torch.Tensor], sample_ages_scalar: float
    ) -> None:
        result = calculate_non_spine_ages_log_lik_derivatives(
            nearly_equal_params["lambda_param"],
            nearly_equal_params["mu_param"],
            sample_ages_scalar,
            nearly_equal_params["time"],
        )

        assert torch.all(torch.isfinite(result))


class TestAgesLogLikelihoodDerivatives:
    """Test unified age log-likelihood derivatives interface."""

    def test_spine_derivatives_routing(
        self, standard_scalar_params: Dict[str, torch.Tensor], sample_ages_scalar: float
    ) -> None:
        result = calculate_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=True,
        )

        expected = calculate_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert_allclose(result, expected)

    def test_non_spine_derivatives_routing(
        self, standard_scalar_params: Dict[str, torch.Tensor], sample_ages_scalar: float
    ) -> None:
        result = calculate_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
            are_spines=False,
        )

        expected = calculate_non_spine_ages_log_lik_derivatives(
            standard_scalar_params["lambda_param"],
            standard_scalar_params["mu_param"],
            sample_ages_scalar,
            standard_scalar_params["time"],
        )

        assert_allclose(result, expected)


class TestDerivativeIntegration:
    def test_derivatives_sum_over_array(self) -> None:
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = 0.08
        T = torch.tensor([50.0], dtype=torch.float64)
        ns = torch.tensor([1, 5, 10], dtype=torch.int64)

        result_together = calculate_populations_log_lik_derivatives(
            lambda_val, mu_val, T, ns
        )

        derivs_separate = [
            calculate_populations_log_lik_derivatives(lambda_val, mu_val, T, n)
            for n in ns
        ]
        result_separate = torch.stack(derivs_separate, dim=0).sum(dim=0)

        assert_allclose(result_together, result_separate, rtol=1e-8)

    def test_derivatives_match_numerical_gradient(
        self, standard_scalar_params: Dict[str, torch.Tensor]
    ) -> None:
        lambda_param = (
            standard_scalar_params["lambda_param"].detach().clone().requires_grad_(True)
        )
        mu_param = (
            standard_scalar_params["mu_param"].detach().clone().requires_grad_(True)
        )

        objective_val = calculate_populations_log_lik(
            lambda_param, mu_param, standard_scalar_params["time"], ns=5
        )

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_populations_log_lik_derivatives(
                lambda_param.detach(),
                mu_param.detach(),
                standard_scalar_params["time"],
                ns=5,
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            f"Gradient error {relative_error.max().item():.2e} exceeds tolerance"
        )


class TestGradientValidation:
    """
    Comprehensive tests for analytical gradient functions
    using PyTorch's native autograd.
    """

    def test_populations_log_lik_derivatives_scalar_n(self) -> None:
        time = 100.0
        n = 5

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_populations_log_lik(
            lambda_param, mu_param, time, ns=n
        )

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_populations_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), time, ns=n
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for scalar n"
        )

    def test_populations_log_lik_derivatives_array_n(self) -> None:
        time = 100.0
        ns = torch.tensor([0, 1, 2, 5, 10], dtype=torch.int64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_populations_log_lik(
            lambda_param, mu_param, time, ns=ns
        )

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_populations_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), time, ns=ns
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 3e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for array n"
        )

    def test_populations_log_lik_derivatives_near_equal_rates(self) -> None:
        time = 100.0
        n = 5

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.1 + 1e-13, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_populations_log_lik(
            lambda_param, mu_param, time, ns=n
        )

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_populations_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), time, ns=n
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for lambda ~= mu"
        )

    def test_non_spine_ages_derivatives_scalar_age(self) -> None:
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_non_spine_ages_log_lik(lambda_param, mu_param, age, T)

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_non_spine_ages_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), age, T
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for scalar age"
        )

    def test_non_spine_ages_derivatives_array_ages(self) -> None:
        ages = torch.tensor([10.0, 30.0, 50.0, 70.0, 90.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_non_spine_ages_log_lik(
            lambda_param, mu_param, ages, T
        )

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_non_spine_ages_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), ages, T
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for array ages"
        )

    def test_non_spine_ages_derivatives_near_equal_rates(self) -> None:
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.1 + 1e-13, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_non_spine_ages_log_lik(lambda_param, mu_param, age, T)

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_non_spine_ages_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), age, T
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for lambda ~= mu"
        )

    def test_spine_ages_derivatives_interior_ages(self) -> None:
        ages = torch.tensor([10.0, 30.0, 50.0, 70.0, 90.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_spine_ages_log_lik(
            lambda_param,
            mu_param,
            ages,
            T,
        )

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_spine_ages_log_lik_derivatives(
                lambda_param.detach(),
                mu_param.detach(),
                ages,
                T,
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for interior ages"
        )

    def test_spine_ages_derivatives_boundary_age(self) -> None:
        age = 100.0
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_spine_ages_log_lik(lambda_param, mu_param, age, T)

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_spine_ages_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), age, T
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance at boundary"
        )

    def test_spine_ages_derivatives_mixed_ages(self) -> None:
        ages = torch.tensor([10.0, 50.0, 90.0, 100.0, 100.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_spine_ages_log_lik(lambda_param, mu_param, ages, T)

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_spine_ages_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), ages, T
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for mixed ages"
        )

    def test_spine_ages_derivatives_near_equal_rates(self) -> None:
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.1 + 1e-13, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_spine_ages_log_lik(lambda_param, mu_param, age, T)

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_spine_ages_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), age, T
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            "Gradient error "
            f"{relative_error.max().item():.2e} exceeds tolerance for lambda ~= mu"
        )

    def test_derivatives_with_log_scale_params(self) -> None:
        log_lambda = torch.log(torch.tensor(0.1, dtype=torch.float64))
        log_mu = torch.log(torch.tensor(0.05, dtype=torch.float64))
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        log_params = torch.tensor(
            [log_lambda, log_mu], dtype=torch.float64, requires_grad=True
        )

        lambda_val = torch.exp(log_params[0])
        mu_val = torch.exp(log_params[1])

        objective_val = calculate_non_spine_ages_log_lik(lambda_val, mu_val, age, T)

        objective_val.backward()
        autograd_grads = log_params.grad

        with torch.no_grad():
            analytical_grads = calculate_non_spine_ages_log_lik_derivatives(
                lambda_val.detach(), mu_val.detach(), age, T
            )
            analytical_grads_log = analytical_grads * torch.exp(log_params.detach())

        relative_error = torch.abs(autograd_grads - analytical_grads_log) / (
            torch.abs(analytical_grads_log) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            f"Gradient error {relative_error} exceeds tolerance for log-scale params"
        )

    def test_finite_difference_manual_check(self) -> None:
        lambda_val = torch.tensor([0.1], dtype=torch.float64)
        mu_val = torch.tensor([0.05], dtype=torch.float64)
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)
        epsilon = 1e-7

        analytical = calculate_non_spine_ages_log_lik_derivatives(
            lambda_val,
            mu_val,
            age,
            T,
        )

        f_lambda_plus = calculate_non_spine_ages_log_lik(
            lambda_val + epsilon,
            mu_val,
            age,
            T,
        )
        f_lambda_minus = calculate_non_spine_ages_log_lik(
            lambda_val - epsilon,
            mu_val,
            age,
            T,
        )
        d_lambda_numerical = (f_lambda_plus - f_lambda_minus) / (2 * epsilon)

        f_mu_plus = calculate_non_spine_ages_log_lik(
            lambda_val,
            mu_val + epsilon,
            age,
            T,
        )
        f_mu_minus = calculate_non_spine_ages_log_lik(
            lambda_val,
            mu_val - epsilon,
            age,
            T,
        )
        d_mu_numerical = (f_mu_plus - f_mu_minus) / (2 * epsilon)

        numerical = torch.tensor(
            [d_lambda_numerical, d_mu_numerical], dtype=torch.float64
        )

        relative_error = torch.abs(analytical - numerical) / (
            torch.abs(analytical) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            f"Relative error {relative_error} exceeds tolerance"
        )

    def test_gradient_consistency_across_parameter_ranges(self) -> None:
        ages = torch.tensor([20.0, 50.0, 80.0], dtype=torch.float64)
        T = torch.tensor([100.0], dtype=torch.float64)

        test_cases = [
            (0.01, 0.005, "low rates"),
            (0.1, 0.05, "medium rates"),
            (0.5, 0.3, "high rates"),
            (0.1, 0.09, "close rates"),
            (0.2, 0.05, "large difference"),
        ]

        for lambda_val, mu_val, description in test_cases:
            lambda_param = torch.tensor(
                lambda_val, dtype=torch.float64, requires_grad=True
            )
            mu_param = torch.tensor(mu_val, dtype=torch.float64, requires_grad=True)

            objective_val = (
                calculate_non_spine_ages_log_lik(lambda_param, mu_param, ages, T)
                if lambda_param.grad is not None and mu_param.grad is not None
                else None
            )

            objective_val.backward()
            autograd_grads = (
                torch.tensor(
                    [lambda_param.grad.item(), mu_param.grad.item()],
                    dtype=torch.float64,
                )
                if lambda_param.grad is not None and mu_param.grad is not None
                else None
            )

            with torch.no_grad():
                analytical_grads = calculate_non_spine_ages_log_lik_derivatives(
                    lambda_param.detach(), mu_param.detach(), ages, T
                )

            relative_error = torch.abs(autograd_grads - analytical_grads) / (
                torch.abs(analytical_grads) + 1e-10
            )
            assert torch.all(relative_error < 1e-4), (
                "Gradient error "
                f"{relative_error.max().item():.2e} exceeds tolerance for {description}"
            )

    def test_wrapper_function_derivatives(self) -> None:
        age = 50.0
        T = torch.tensor([100.0], dtype=torch.float64)

        lambda_param = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val = calculate_ages_log_lik(
            lambda_param, mu_param, age, T, are_spines=False
        )

        objective_val.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param.grad.item(), mu_param.grad.item()], dtype=torch.float64
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_ages_log_lik_derivatives(
                lambda_param.detach(), mu_param.detach(), age, T, are_spines=False
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            f"Wrapper gradient error {relative_error.max().item():.2e} exceeds tolerance"
        )

        lambda_param_spine = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        mu_param_spine = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

        objective_val_spine = calculate_ages_log_lik(
            lambda_param_spine, mu_param_spine, age, T, are_spines=True
        )

        objective_val_spine.backward()
        autograd_grads = (
            torch.tensor(
                [lambda_param_spine.grad.item(), mu_param_spine.grad.item()],
                dtype=torch.float64,
            )
            if lambda_param.grad is not None and mu_param.grad is not None
            else None
        )

        with torch.no_grad():
            analytical_grads = calculate_ages_log_lik_derivatives(
                lambda_param_spine.detach(),
                mu_param_spine.detach(),
                age,
                T,
                are_spines=True,
            )

        relative_error = torch.abs(autograd_grads - analytical_grads) / (
            torch.abs(analytical_grads) + 1e-10
        )
        assert torch.all(relative_error < 1e-5), (
            f"Wrapper gradient error {relative_error.max().item():.2e} exceeds tolerance"
        )
