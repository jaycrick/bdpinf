"""Tests for models.validation input normalization and validation."""

import pytest
import torch

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
        assert isinstance(br, torch.Tensor)
        assert isinstance(dr, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        assert br.dtype == torch.float64
        assert dr.dtype == torch.float64
        assert t.dtype == torch.float64
        assert float(br.item()) == pytest.approx(0.01)
        assert float(dr.item()) == pytest.approx(0.003)
        assert float(t.item()) == pytest.approx(500.0)

    def test_validate_raises_for_negative_times(self) -> None:
        with pytest.raises(ValueError, match="times must be"):
            normalize_and_validate_inputs(
                birth_rates=0.01, death_rates=0.003, times=-1.0
            )

    def test_validate_raises_for_non_monotone_times(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            normalize_and_validate_inputs(
                birth_rates=torch.tensor([0.01, 0.01], dtype=torch.float64),
                death_rates=torch.tensor([0.003, 0.003], dtype=torch.float64),
                times=torch.tensor([500.0, 300.0], dtype=torch.float64),
            )

    def test_validate_raises_for_ages_outside_range(self) -> None:
        with pytest.raises(ValueError, match="ages"):
            normalize_and_validate_inputs(
                birth_rates=0.01,
                death_rates=0.003,
                times=100.0,
                ages=torch.tensor([200.0], dtype=torch.float64),
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
            times=torch.tensor([500.0, 300.0], dtype=torch.float64),
            config=config,
        )
        assert isinstance(br, torch.Tensor)
        assert isinstance(t, torch.Tensor)

    def test_normalize_multi_epoch_returns_correct_lengths(self) -> None:
        n = 3
        br = torch.full((n,), 0.01, dtype=torch.float64)
        dr = torch.full((n,), 0.003, dtype=torch.float64)
        t = torch.tensor([200.0, 400.0, 600.0], dtype=torch.float64)
        out_br, out_dr, out_t = normalize_and_validate_inputs(
            birth_rates=br, death_rates=dr, times=t
        )
        assert len(out_br) == n
        assert len(out_dr) == n
        assert len(out_t) == n
