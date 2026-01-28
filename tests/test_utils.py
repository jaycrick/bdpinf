"""
Test suite for BirthDeathProcess utility functions and helper methods.
"""

import numpy as np
import torch

from inf_bp_texts.inference.likelihood import log_likelihood
from inf_bp_texts.models import BirthDeathProcess
from inf_bp_texts.utils import ensure_1d_tensor


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_creation_times(self) -> None:
        """Test behavior with empty creation times."""
        process = BirthDeathProcess([], 1000.0)

        assert len(process.creation_times) == 0
        assert len(process.get_manuscript_ages()) == 0

        # Should handle empty state gracefully
        summary = process.get_process_summary()
        assert summary["n_manuscripts"] == 0

    def test_zero_rates(self) -> None:
        """Test behavior with zero rates."""
        # Zero birth rate should be invalid for log-likelihood
        process = BirthDeathProcess([], 1000.0, birth_rate=0.0, death_rate=0.01)
        process.creation_times = [100.0]
        process.set_final_time(200.0)

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()
        ll = log_likelihood(0.0, 0.01, ages, process.final_time)
        assert ll == -np.inf

    def test_negative_rates(self) -> None:
        """Test behavior with negative rates."""
        process = BirthDeathProcess([], 1000.0)
        process.creation_times = [100.0]
        process.set_final_time(200.0)

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()

        # Negative rates should give -inf log-likelihood
        ll = log_likelihood(-0.01, 0.01, ages, process.final_time)
        assert ll == -np.inf

        ll = log_likelihood(0.01, -0.01, ages, process.final_time)
        assert ll == -np.inf

    def test_very_large_ages(self) -> None:
        """Test behavior with very large ages."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.001, death_rate=0.0001)
        process.creation_times = [1.0, 2.0, 3.0]
        process.set_final_time(100000.0)  # Very large final time

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()

        # Should handle large ages without crashing
        ll = log_likelihood(0.001, 0.0001, ages, process.final_time)
        # Result might be -inf due to numerical limits, but shouldn't crash
        assert np.isfinite(ll) or ll == -np.inf

    def test_single_manuscript(self) -> None:
        """Test behavior with single manuscript."""
        process = BirthDeathProcess([], 1000.0, birth_rate=0.01, death_rate=0.005)
        process.add_manuscript(100.0)
        process.set_final_time(500.0)

        assert len(process.get_manuscript_ages()) == 1
        assert process.get_manuscript_ages()[0] == 400.0

        # Use modular log_likelihood function
        ages = process.get_manuscript_ages()

        # Should be able to calculate log-likelihood
        ll = log_likelihood(0.01, 0.005, ages, process.final_time)
        assert np.isfinite(ll)

    def test_final_time_before_creation_time(self) -> None:
        """Test behavior when final_time is before creation_time."""
        process = BirthDeathProcess([], 1000.0)
        process.add_manuscript(1000.0)
        process.set_final_time(500.0)  # Before creation time

        # The manuscript should be filtered out (since creation time > final time)
        # So there should be no manuscripts left
        ages = process.get_manuscript_ages()
        assert len(ages) == 0


class TestEnsure1dTensor:
    """Test the ensure_1d_tensor utility function."""

    def test_float_scalar(self) -> None:
        """Test conversion of float scalar."""
        result = ensure_1d_tensor(5.0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)
        assert result.dtype == torch.float64
        assert result.item() == 5.0

    def test_int_scalar(self) -> None:
        """Test conversion of int scalar."""
        result = ensure_1d_tensor(42)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)
        assert result.dtype == torch.float64
        assert result.item() == 42.0

    def test_zero_scalar(self) -> None:
        """Test conversion of zero scalar."""
        result = ensure_1d_tensor(0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)
        assert result.item() == 0.0

    def test_negative_scalar(self) -> None:
        """Test conversion of negative scalar."""
        result = ensure_1d_tensor(-3.14)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)
        assert result.item() == -3.14

    def test_1d_tensor(self) -> None:
        """Test that 1D tensor is preserved."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = ensure_1d_tensor(input_tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert result.dtype == torch.float64
        assert torch.allclose(result, input_tensor)

    def test_1d_tensor_wrong_dtype(self) -> None:
        """Test that 1D tensor with wrong dtype is converted."""
        input_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = ensure_1d_tensor(input_tensor)
        assert result.dtype == torch.float64
        assert result.shape == (3,)
        assert torch.allclose(
            result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )

    def test_2d_tensor_flattened(self) -> None:
        """Test that 2D tensor is flattened."""
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        result = ensure_1d_tensor(input_tensor)
        assert result.shape == (4,)
        assert result.dtype == torch.float64
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_3d_tensor_flattened(self) -> None:
        """Test that 3D tensor is flattened."""
        input_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = ensure_1d_tensor(input_tensor)
        assert result.shape == (8,)
        assert result.dtype == torch.float64
        expected = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        assert torch.allclose(result, expected)

    def test_0d_tensor_scalar(self) -> None:
        """Test conversion of 0D (scalar) tensor."""
        input_tensor = torch.tensor(7.5, dtype=torch.float32)
        result = ensure_1d_tensor(input_tensor)
        assert result.shape == (1,)
        assert result.dtype == torch.float64
        assert result.item() == 7.5

    def test_empty_tensor(self) -> None:
        """Test conversion of empty tensor."""
        input_tensor = torch.tensor([], dtype=torch.float64)
        result = ensure_1d_tensor(input_tensor)
        assert result.shape == (0,)
        assert result.dtype == torch.float64

    def test_numpy_array(self) -> None:
        """Test conversion from numpy array."""
        input_array = np.array([1.0, 2.0, 3.0])
        result = ensure_1d_tensor(input_array)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert result.dtype == torch.float64
        assert torch.allclose(
            result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )

    def test_numpy_2d_array_flattened(self) -> None:
        """Test conversion from 2D numpy array."""
        input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ensure_1d_tensor(input_array)
        assert result.shape == (4,)
        assert result.dtype == torch.float64
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_list_conversion(self) -> None:
        """Test conversion from Python list."""
        input_list = [1.0, 2.0, 3.0]
        result = ensure_1d_tensor(input_list)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert result.dtype == torch.float64
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_preserves_gradient(self) -> None:
        """Test that gradient tracking is preserved for tensors with requires_grad."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = ensure_1d_tensor(input_tensor)
        assert result.requires_grad
        # Test backward pass
        loss = result.sum()
        loss.backward()
        assert input_tensor.grad is not None
