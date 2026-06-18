"""Tests for models.validation input normalization and validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inf_bp_texts.models.validation import (
    ValidationConfig,
    normalize_and_validate_inputs,
)


class TestValidation:
    """Test normalize_and_validate_inputs and ValidationConfig."""

    def test_normalize_and_validate_inputs_returns_tensors_for_valid_input(
        self,
    ) -> None:
        br, dr, t = normalize_and_validate_inputs(
            birth_rates=0.01, death_rates=0.003, times=500.0
        )
        assert isinstance(br, jax.Array)
        assert isinstance(dr, jax.Array)
        assert isinstance(t, jax.Array)
        assert br.dtype == jnp.float64
        assert dr.dtype == jnp.float64
        assert t.dtype == jnp.float64
        assert float(br[0]) == pytest.approx(0.01)
        assert float(dr[0]) == pytest.approx(0.003)
        assert float(t[0]) == pytest.approx(500.0)

    def test_validate_raises_for_negative_times(self) -> None:
        with pytest.raises(ValueError, match="times must be"):
            normalize_and_validate_inputs(
                birth_rates=0.01, death_rates=0.003, times=-1.0
            )

    def test_validate_raises_for_non_monotone_times(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            normalize_and_validate_inputs(
                birth_rates=np.array([0.01, 0.01]),
                death_rates=np.array([0.003, 0.003]),
                times=np.array([500.0, 300.0]),
            )

    def test_validate_raises_for_ages_outside_range(self) -> None:
        with pytest.raises(ValueError, match="ages"):
            normalize_and_validate_inputs(
                birth_rates=0.01,
                death_rates=0.003,
                times=100.0,
                ages=np.array([200.0]),
            )

    def test_validation_config_disables_selected_checks(self) -> None:
        config = ValidationConfig(
            validate_lengths=False,
            validate_shapes=False,
            require_times_positive=False,
            require_times_increasing=False,
            require_ages_in_range=False,
            require_rates_non_negative=False,
        )
        # Should not raise despite mismatched lengths and non-monotone times
        br, t = normalize_and_validate_inputs(
            birth_rates=0.01,
            times=np.array([500.0, 300.0]),
            config=config,
        )
        assert isinstance(br, jax.Array)
        assert isinstance(t, jax.Array)

    def test_normalize_multi_epoch_returns_correct_lengths(self) -> None:
        n = 3
        br = np.full(n, 0.01)
        dr = np.full(n, 0.003)
        t = np.array([200.0, 400.0, 600.0])
        out_br, out_dr, out_t = normalize_and_validate_inputs(
            birth_rates=br, death_rates=dr, times=t
        )
        assert len(out_br) == n
        assert len(out_dr) == n
        assert len(out_t) == n
