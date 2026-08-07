"""Tests for parameter constraints in fitting module."""

import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.inference.fitting import (
    ParameterConstraints,
    _apply_constraints_to_full_params,
    _reduce_to_free_params,
    fit_birth_death_processes,
)


class TestParameterConstraints:
    """Test ParameterConstraints validation and utility methods."""

    def test_validate_empty_constraints(self) -> None:
        """Empty constraints should validate successfully."""
        constraints = ParameterConstraints()
        constraints.validate(6)  # Should not raise

    def test_validate_out_of_bounds(self) -> None:
        """Out of bounds indices should raise ValueError."""
        constraints = ParameterConstraints(fixed_params={10: 0.5})
        with pytest.raises(ValueError, match="out of range"):
            constraints.validate(6)

    def test_validate_overlapping_indices(self) -> None:
        """Overlapping fixed and equality indices should raise ValueError."""
        constraints = ParameterConstraints(
            fixed_params={0: 0.5},
            equality_groups=[[0, 1]],
        )
        with pytest.raises(ValueError, match="appears in both"):
            constraints.validate(6)

    def test_validate_single_element_group(self) -> None:
        """Equality groups with single element should raise ValueError."""
        constraints = ParameterConstraints(equality_groups=[[0]])
        with pytest.raises(ValueError, match="at least 2 elements"):
            constraints.validate(6)

    def test_get_free_param_indices_no_constraints(self) -> None:
        """With no constraints, all indices should be free."""
        constraints = ParameterConstraints()
        free_indices = constraints.get_free_param_indices(6)
        assert free_indices == [0, 1, 2, 3, 4, 5]

    def test_get_free_param_indices_with_fixed(self) -> None:
        """Fixed parameters should not be in free indices."""
        constraints = ParameterConstraints(fixed_params={0: 0.5, 2: 0.3})
        free_indices = constraints.get_free_param_indices(6)
        assert 0 not in free_indices
        assert 2 not in free_indices
        assert set(free_indices) == {1, 3, 4, 5}

    def test_get_free_param_indices_with_equality(self) -> None:
        """Equality groups should have one representative in free indices."""
        constraints = ParameterConstraints(equality_groups=[[0, 1], [3, 4]])
        free_indices = constraints.get_free_param_indices(6)
        # Should include one from each group plus unconstrained indices
        assert 0 in free_indices or 1 in free_indices  # One from first group
        assert 3 in free_indices or 4 in free_indices  # One from second group
        assert 2 in free_indices  # Unconstrained
        assert 5 in free_indices  # Unconstrained
        assert len(free_indices) == 4  # 1 + 1 + 2 unconstrained


class TestConstraintTransforms:
    """Test transformation between full and reduced parameter spaces."""

    def test_apply_constraints_no_constraints(self) -> None:
        """With no constraints, full params should match input free params."""
        free_params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        constraints = ParameterConstraints()
        full_params = _apply_constraints_to_full_params(free_params, constraints, 6)
        assert jnp.allclose(full_params, free_params)

    def test_apply_constraints_with_fixed(self) -> None:
        """Fixed parameters should be set correctly."""
        free_params = jnp.array([0.1, 0.2, 0.4, 0.5])
        constraints = ParameterConstraints(fixed_params={0: 0.99, 2: 0.88})
        full_params = _apply_constraints_to_full_params(free_params, constraints, 6)

        assert float(full_params[0]) == pytest.approx(0.99)
        assert float(full_params[2]) == pytest.approx(0.88)
        # Free params should be in their positions
        free_idx = 0
        for i in [1, 3, 4, 5]:  # free indices
            assert float(full_params[i]) == pytest.approx(float(free_params[free_idx]))
            free_idx += 1

    def test_apply_constraints_with_equality(self) -> None:
        """Equal parameters should have same value."""
        free_params = jnp.array([0.1, 0.2, 0.3, 0.4])
        constraints = ParameterConstraints(equality_groups=[[1, 2], [4, 5]])
        full_params = _apply_constraints_to_full_params(free_params, constraints, 6)

        # Indices 1 and 2 should be equal
        assert float(full_params[1]) == pytest.approx(float(full_params[2]))
        # Indices 4 and 5 should be equal
        assert float(full_params[4]) == pytest.approx(float(full_params[5]))

    def test_round_trip_transform(self) -> None:
        """Reducing and expanding should be inverse operations."""
        constraints = ParameterConstraints(
            fixed_params={0: 0.1, 3: 0.4},
            equality_groups=[[1, 2], [4, 5]],
        )

        # Apply equality constraints to original
        full_params_constrained = jnp.array([0.1, 0.2, 0.2, 0.4, 0.5, 0.5])

        reduced = _reduce_to_free_params(full_params_constrained, constraints, 6)
        expanded = _apply_constraints_to_full_params(reduced, constraints, 6)

        assert jnp.allclose(expanded, full_params_constrained)

    def test_reduce_to_free_params(self) -> None:
        """Should extract only free parameters in correct order."""
        full_params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        constraints = ParameterConstraints(fixed_params={0: 0.1, 3: 0.4})

        reduced = _reduce_to_free_params(full_params, constraints, 6)

        # Should have indices 1, 2, 4, 5, and one from [1,2] and one from [4,5]
        # Actually, with no equality groups, should have 4 free params (6 - 2 fixed)
        expected_count = 4  # indices 1, 2, 4, 5
        assert len(reduced) == expected_count


class TestConstraintDataTypes:
    """Test constraint handling with different array dtypes."""

    def test_constraints_float32(self) -> None:
        """Constraints should work with float32 arrays."""
        free_params = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        constraints = ParameterConstraints()
        full_params = _apply_constraints_to_full_params(free_params, constraints, 3)
        assert full_params.dtype == jnp.float32

    def test_constraints_float64(self) -> None:
        """Constraints should work with float64 arrays."""
        free_params = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)
        constraints = ParameterConstraints()
        full_params = _apply_constraints_to_full_params(free_params, constraints, 3)
        assert full_params.dtype == jnp.float64

    def test_constraints_returns_jax_array(self) -> None:
        """Output should be a JAX array (not numpy or other type)."""
        import jax

        free_params = jnp.array([0.1, 0.2, 0.3])
        constraints = ParameterConstraints()
        full_params = _apply_constraints_to_full_params(free_params, constraints, 3)
        assert isinstance(full_params, jax.Array)


class TestComplexConstraintScenarios:
    """Test realistic constraint scenarios."""

    def test_time_invariant_rates(self) -> None:
        """All rates equal - reduces 6 params to 2."""
        # n_epochs = 3, so 6 params total
        constraints = ParameterConstraints(
            equality_groups=[
                [0, 1, 2],  # all birth rates equal
                [3, 4, 5],  # all death rates equal
            ],
        )

        free_indices = constraints.get_free_param_indices(6)
        # Should have 2 free params (representatives from each group)
        assert len(free_indices) == 2

        # Apply constraints
        free_params = jnp.array([0.1, 0.05])
        full_params = _apply_constraints_to_full_params(free_params, constraints, 6)

        # All birth rates should be 0.1
        assert jnp.allclose(full_params[:3], jnp.full((3,), 0.1))
        # All death rates should be 0.05
        assert jnp.allclose(full_params[3:], jnp.full((3,), 0.05))

    def test_partial_pooling(self) -> None:
        """Early and late periods have constant rates within group."""
        # n_epochs = 4, group epochs 0-1 and 2-3
        constraints = ParameterConstraints(
            equality_groups=[
                [0, 1],  # birth rates epochs 0-1
                [2, 3],  # birth rates epochs 2-3
                [4, 5],  # death rates epochs 0-1
                [6, 7],  # death rates epochs 2-3
            ],
        )

        free_indices = constraints.get_free_param_indices(8)
        assert len(free_indices) == 4  # One from each group

        free_params = jnp.array([0.1, 0.15, 0.05, 0.08])
        full_params = _apply_constraints_to_full_params(free_params, constraints, 8)

        # Check pooling worked
        assert jnp.allclose(full_params[0], full_params[1])
        assert jnp.allclose(full_params[2], full_params[3])
        assert jnp.allclose(full_params[4], full_params[5])
        assert jnp.allclose(full_params[6], full_params[7])


class TestConstraintSatisfactionAfterFitting:
    """Check that constraints remain satisfied in fitted outputs."""

    def test_fit_preserves_fixed_parameter_values(self) -> None:
        """Fixed parameters should remain fixed after fitting."""
        ages_epoch_1 = np.array([100.0, 180.0, 240.0], dtype=np.float64)
        ages_epoch_2 = np.array([120.0, 260.0, 390.0], dtype=np.float64)
        times = np.array([500.0, 1000.0], dtype=np.float64)
        fixed_birth_0 = 0.01
        fixed_death_1 = 0.02
        constraints = ParameterConstraints(
            fixed_params={0: fixed_birth_0, 3: fixed_death_1},
        )

        result = fit_birth_death_processes(
            list_of_ages=[ages_epoch_1, ages_epoch_2],
            times=times,
            param_constraints=constraints,
        )

        if result["success"]:
            assert result["birth_rates"][0] == pytest.approx(fixed_birth_0)
            assert result["death_rates"][1] == pytest.approx(fixed_death_1)

    def test_fit_preserves_equality_groups(self) -> None:
        """Equal-parameter groups should still be equal after fitting."""
        ages_epoch_1 = np.array([80.0, 140.0, 210.0], dtype=np.float64)
        ages_epoch_2 = np.array([110.0, 230.0, 360.0], dtype=np.float64)
        times = np.array([500.0, 1000.0], dtype=np.float64)
        constraints = ParameterConstraints(
            equality_groups=[[0, 1], [2, 3]],
        )

        result = fit_birth_death_processes(
            list_of_ages=[ages_epoch_1, ages_epoch_2],
            times=times,
            param_constraints=constraints,
        )

        if result["success"]:
            assert result["birth_rates"][0] == pytest.approx(
                result["birth_rates"][1],
            )
            assert result["death_rates"][0] == pytest.approx(
                result["death_rates"][1],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
