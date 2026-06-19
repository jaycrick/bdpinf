"""Test suite for BirthDeathProcess utility functions and helper methods."""

import jax
import jax.numpy as jnp
import numpy as np

from inf_bp_texts.inference.likelihood import log_likelihood
from inf_bp_texts.models import BirthDeathProcess
from inf_bp_texts.utils import ensure_1d_tensor, format_ordinal


class TestFormatOrdinal:
    """Test format_ordinal suffix handling, including the 11-13 special case."""

    def test_first_second_third(self) -> None:
        assert format_ordinal(1) == "1st"
        assert format_ordinal(2) == "2nd"
        assert format_ordinal(3) == "3rd"
        assert format_ordinal(4) == "4th"

    def test_eleven_to_thirteen_are_th(self) -> None:
        assert format_ordinal(11) == "11th"
        assert format_ordinal(12) == "12th"
        assert format_ordinal(13) == "13th"

    def test_century_values(self) -> None:
        assert format_ordinal(21) == "21st"
        assert format_ordinal(22) == "22nd"
        assert format_ordinal(23) == "23rd"
        assert format_ordinal(111) == "111th"
        assert format_ordinal(113) == "113th"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_creation_times(self) -> None:
        """Test behavior with empty creation times."""
        process = BirthDeathProcess(creation_times=[], epoch_end_times=1000.0)

        assert len(process.creation_times) == 0
        assert len(process.manuscript_ages) == 0
        assert process.n_manuscripts == 0

    def test_zero_rates(self) -> None:
        """Test behavior with zero rates."""
        # Zero birth rate should be invalid for log-likelihood
        process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=1000.0,
            birth_rates=0.0,
            death_rates=0.01,
        )
        process.creation_times = [100.0]
        process.set_epoch_end_times(200.0)

        # Use modular log_likelihood function
        ages = process.manuscript_ages
        ll = log_likelihood(0.0, 0.01, ages, process.epoch_end_times[-1])
        assert ll == -np.inf

    def test_negative_rates(self) -> None:
        """Test behavior with negative rates."""
        process = BirthDeathProcess(creation_times=[], epoch_end_times=1000.0)
        process.creation_times = [100.0]
        process.set_epoch_end_times(200.0)

        # Use modular log_likelihood function
        ages = process.manuscript_ages

        # Negative rates should give -inf log-likelihood
        ll = log_likelihood(-0.01, 0.01, ages, process.epoch_end_times[-1])
        assert ll == -np.inf

        ll = log_likelihood(0.01, -0.01, ages, process.epoch_end_times[-1])
        assert ll == -np.inf

    def test_very_large_ages(self) -> None:
        """Test behavior with very large ages."""
        process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=1000.0,
            birth_rates=0.001,
            death_rates=0.0001,
        )
        process.creation_times = [1.0, 2.0, 3.0]
        process.set_epoch_end_times(100000.0)  # Very large final time

        # Use modular log_likelihood function
        ages = process.manuscript_ages

        # Should handle large ages without crashing
        ll = log_likelihood(0.001, 0.0001, ages, process.epoch_end_times[-1])
        # Result might be -inf due to numerical limits, but shouldn't crash
        assert np.isfinite(ll) or ll == -np.inf

    def test_single_manuscript(self) -> None:
        """Test behavior with single manuscript."""
        process = BirthDeathProcess(
            creation_times=[],
            epoch_end_times=1000.0,
            birth_rates=0.01,
            death_rates=0.005,
        )
        process.add_manuscript(100.0)
        process.set_epoch_end_times(500.0)

        assert len(process.manuscript_ages) == 1
        assert process.manuscript_ages[0] == 400.0

        # Use modular log_likelihood function
        ages = process.manuscript_ages

        # Should be able to calculate log-likelihood
        ll = log_likelihood(0.01, 0.005, ages, process.epoch_end_times[-1])
        assert np.isfinite(ll)

    def test_epoch_end_time_before_creation_time(self) -> None:
        """Test behavior when epoch end time is before creation_time."""
        process = BirthDeathProcess(creation_times=[], epoch_end_times=1000.0)
        process.add_manuscript(1000.0)
        process.set_epoch_end_times(500.0)  # Before creation time

        # The manuscript should be filtered out (since creation time > final time)
        # So there should be no manuscripts left
        ages = process.manuscript_ages
        assert len(ages) == 0


class TestEnsure1dTensor:
    """Test the ensure_1d_tensor utility function."""

    def test_float_scalar(self) -> None:
        """Test conversion of float scalar."""
        result = ensure_1d_tensor(5.0)
        assert isinstance(result, jax.Array)
        assert result.shape == (1,)
        assert result.dtype == jnp.float64
        assert float(result[0]) == 5.0

    def test_int_scalar(self) -> None:
        """Test conversion of int scalar."""
        result = ensure_1d_tensor(42)
        assert isinstance(result, jax.Array)
        assert result.shape == (1,)
        assert result.dtype == jnp.float64
        assert float(result[0]) == 42.0

    def test_zero_scalar(self) -> None:
        """Test conversion of zero scalar."""
        result = ensure_1d_tensor(0)
        assert isinstance(result, jax.Array)
        assert result.shape == (1,)
        assert float(result[0]) == 0.0

    def test_negative_scalar(self) -> None:
        """Test conversion of negative scalar."""
        result = ensure_1d_tensor(-3.14)
        assert isinstance(result, jax.Array)
        assert result.shape == (1,)
        assert abs(float(result[0]) - (-3.14)) < 1e-14

    def test_1d_jax_array(self) -> None:
        """Test that 1D JAX array is preserved."""
        input_arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        result = ensure_1d_tensor(input_arr)
        assert isinstance(result, jax.Array)
        assert result.shape == (3,)
        assert result.dtype == jnp.float64
        assert np.allclose(np.array(result), np.array([1.0, 2.0, 3.0]))

    def test_1d_numpy_wrong_dtype(self) -> None:
        """Test that integer array is cast to float64."""
        input_arr = np.array([1, 2, 3], dtype=np.int32)
        result = ensure_1d_tensor(input_arr)
        assert result.dtype == jnp.float64
        assert result.shape == (3,)
        assert np.allclose(np.array(result), np.array([1.0, 2.0, 3.0]))

    def test_2d_array_flattened(self) -> None:
        """Test that 2D array is flattened."""
        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ensure_1d_tensor(input_arr)
        assert result.shape == (4,)
        assert result.dtype == jnp.float64
        assert np.allclose(np.array(result), np.array([1.0, 2.0, 3.0, 4.0]))

    def test_3d_array_flattened(self) -> None:
        """Test that 3D array is flattened."""
        input_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = ensure_1d_tensor(input_arr)
        assert result.shape == (8,)
        assert result.dtype == jnp.float64
        assert np.allclose(
            np.array(result),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )

    def test_0d_array_scalar(self) -> None:
        """Test conversion of 0D (scalar) array."""
        input_arr = np.float32(7.5)
        result = ensure_1d_tensor(input_arr)
        assert result.shape == (1,)
        assert result.dtype == jnp.float64
        assert float(result[0]) == 7.5

    def test_empty_array(self) -> None:
        """Test conversion of empty array."""
        input_arr = np.array([], dtype=np.float64)
        result = ensure_1d_tensor(input_arr)
        assert result.shape == (0,)
        assert result.dtype == jnp.float64

    def test_numpy_array(self) -> None:
        """Test conversion from numpy array."""
        input_array = np.array([1.0, 2.0, 3.0])
        result = ensure_1d_tensor(input_array)
        assert isinstance(result, jax.Array)
        assert result.shape == (3,)
        assert result.dtype == jnp.float64
        assert np.allclose(np.array(result), np.array([1.0, 2.0, 3.0]))

    def test_numpy_2d_array_flattened(self) -> None:
        """Test conversion from 2D numpy array."""
        input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ensure_1d_tensor(input_array)
        assert result.shape == (4,)
        assert result.dtype == jnp.float64
        assert np.allclose(np.array(result), np.array([1.0, 2.0, 3.0, 4.0]))

    def test_list_conversion(self) -> None:
        """Test conversion from Python list."""
        input_list = [1.0, 2.0, 3.0]
        result = ensure_1d_tensor(input_list)
        assert isinstance(result, jax.Array)
        assert result.shape == (3,)
        assert result.dtype == jnp.float64
        assert np.allclose(np.array(result), np.array([1.0, 2.0, 3.0]))

    def test_jax_differentiable(self) -> None:
        """Test that the result supports JAX differentiation."""
        import jax

        result = ensure_1d_tensor(np.array([1.0, 2.0, 3.0]))
        grad_fn = jax.grad(lambda x: x.sum())
        grads = grad_fn(result)
        assert np.allclose(np.array(grads), np.ones(3))
