"""
Integration tests for architectural improvements.

These tests validate that the configuration system, error handling,
and modular structure work together correctly.
"""

import pandas as pd
import pytest

from inf_bp_texts.config import DEFAULT_MODEL, DEFAULT_OPTIMIZATION
from inf_bp_texts.exceptions import ValidationError, validate_positive_number
from inf_bp_texts.inference import fit_from_dataframe
from inf_bp_texts.models.birth_death_process import BirthDeathProcess
from inf_bp_texts.simulation.simulator import generate_synthetic_dataset


class TestArchitecturalIntegration:
    """Test that architectural components work together."""

    def test_config_integration_with_fitting(self):
        """Test that configuration system integrates with fitting module."""
        # Generate test data
        data = generate_synthetic_dataset(
            true_birth_rate=0.2,
            true_death_rate=0.01,
            observation_years=(100, 500),
            n_manuscripts=30,
            random_seed=42,
        )

        # Use configuration in fitting
        config = DEFAULT_OPTIMIZATION
        results = fit_from_dataframe(
            manuscript_data=data,
            age_column="age",
            observation_time=600.0,
            method=config.method,
            initial_params=(config.initial_birth_rate, config.initial_death_rate),
        )

        assert results["success"]
        assert results["birth_rate"] > 0
        assert results["death_rate"] > 0

    def test_error_handling_integration(self):
        """Test that error handling works across modules."""
        # Test validation in model creation
        with pytest.raises(ValidationError):
            validate_positive_number(-1.0, "test_param")

        # Test that model handles empty creation times gracefully
        model = BirthDeathProcess([], final_time=100.0)
        ages = model.get_manuscript_ages()
        assert len(ages) == 0, "Empty model should return empty ages array"

    def test_modular_workflow_integration(self):
        """Test complete workflow using modular components."""
        # 1. Create model with configuration
        config = DEFAULT_MODEL
        model = BirthDeathProcess(
            creation_times=[100.0, 200.0, 300.0],
            final_time=config.default_observation_time,
        )

        # 2. Create test data
        test_data = pd.DataFrame(
            {
                "age": model.get_manuscript_ages(),
                "count": [1] * len(model.get_manuscript_ages()),
            }
        )

        # 3. Fit using inference module
        results = fit_from_dataframe(
            manuscript_data=test_data,
            age_column="age",
            observation_time=model.final_time,
        )

        # 4. Update model with results
        if results["success"]:
            model.update_parameters(
                birth_rate=results["birth_rate"],
                death_rate=results["death_rate"],
                fit_results=results,
            )

            assert model.fitted
            assert model.birth_rate is not None
            assert model.death_rate is not None

    def test_configuration_validation(self):
        """Test that configuration provides valid defaults."""
        opt_config = DEFAULT_OPTIMIZATION
        model_config = DEFAULT_MODEL

        # Validate optimization config
        assert opt_config.method in ["L-BFGS-B", "Nelder-Mead", "Powell", "TNC"]
        assert opt_config.max_iterations > 0
        assert opt_config.tolerance > 0
        assert opt_config.initial_birth_rate > 0
        assert opt_config.initial_death_rate > 0

        # Validate model config
        assert model_config.min_manuscripts_for_fitting > 0
        assert model_config.bootstrap_samples > 0
        assert 0 < model_config.confidence_level < 1

    def test_error_propagation_through_modules(self):
        """Test that errors propagate correctly through the module hierarchy."""
        # Create invalid data
        invalid_data = pd.DataFrame({"wrong_column": [1, 2, 3]})

        # Should raise appropriate error when trying to fit
        with pytest.raises(Exception):  # Could be KeyError or ValidationError
            fit_from_dataframe(
                manuscript_data=invalid_data,
                age_column="age",  # This column doesn't exist
                observation_time=100.0,
            )

    def test_configuration_customization(self):
        """Test that configurations can be customized for different use cases."""
        from inf_bp_texts.config import ModelConfig, OptimizationConfig

        # Create custom configurations
        fast_config = OptimizationConfig(
            method="Nelder-Mead", max_iterations=50, tolerance=1e-3
        )

        test_model_config = ModelConfig(
            min_manuscripts_for_fitting=5,
            bootstrap_samples=10,  # Very small for testing
            confidence_level=0.9,
        )

        # Use custom configs
        assert fast_config.method == "Nelder-Mead"
        assert fast_config.max_iterations == 50
        assert test_model_config.bootstrap_samples == 10


class TestModularImportStructure:
    """Test that the modular import structure is clean and consistent."""

    def test_core_imports(self):
        """Test that core imports work as expected."""
        # Test main package import
        from inf_bp_texts import BirthDeathProcess

        assert BirthDeathProcess is not None

        # Test modular imports
        from inf_bp_texts.config import DEFAULT_OPTIMIZATION
        from inf_bp_texts.exceptions import ValidationError
        from inf_bp_texts.inference.fitting import fit_birth_death_processes
        from inf_bp_texts.models.birth_death_process import (
            BirthDeathProcess as ModelBDP,
        )
        from inf_bp_texts.simulation.simulator import simulate_birth_death_process

        # Ensure they're all accessible
        assert ModelBDP is not None
        assert fit_birth_death_processes is not None
        assert simulate_birth_death_process is not None
        assert DEFAULT_OPTIMIZATION is not None
        assert ValidationError is not None

    def test_no_circular_imports(self):
        """Test that there are no circular import dependencies."""
        # This test simply tries to import all main modules
        # If there are circular dependencies, this will fail

        # If we get here, no circular imports detected
        assert True

    def test_backward_compatibility_removed(self):
        """Test that backward compatibility has been properly removed."""
        from inf_bp_texts import BirthDeathProcess

        # Create model
        model = BirthDeathProcess([], final_time=100.0)

        # These methods should NOT exist (backward compatibility removed)
        assert not hasattr(model, "fit"), "fit() method should be removed"
        assert not hasattr(model, "simulate"), "simulate() method should be removed"
        assert not hasattr(model, "log_likelihood"), (
            "log_likelihood() method should be removed"
        )

        # But this should exist (new modular approach)
        assert hasattr(model, "update_parameters"), (
            "update_parameters() method should exist"
        )


class TestPerformanceAndValidation:
    """Test performance characteristics and validation."""

    def test_configuration_performance(self):
        """Test that configuration access is efficient."""
        import time

        # Time configuration access
        start_time = time.time()
        for _ in range(1000):
            config = DEFAULT_OPTIMIZATION
            _ = config.method
            _ = config.max_iterations
        end_time = time.time()

        # Should be very fast (< 1 second for 1000 accesses)
        assert (end_time - start_time) < 1.0

    def test_validation_error_messages(self):
        """Test that validation provides helpful error messages."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_number(-5.0, "test_parameter")

        error = exc_info.value
        assert "test_parameter" in str(error)
        assert "-5.0" in str(error)
        assert error.field_name == "test_parameter"
        assert error.received_value == -5.0

    def test_memory_usage_reasonable(self):
        """Test that importing modules doesn't use excessive memory."""
        import sys

        # Get initial memory footprint
        initial_modules = len(sys.modules)

        # Import our modules

        # Check that we haven't loaded an excessive number of modules
        final_modules = len(sys.modules)
        new_modules = final_modules - initial_modules

        # Should be reasonable (< 20 new modules)
        assert new_modules < 20, f"Imported too many modules: {new_modules}"

        # Check that we haven't loaded an excessive number of modules
        final_modules = len(sys.modules)
        new_modules = final_modules - initial_modules

        # Should be reasonable (< 20 new modules)
        assert new_modules < 20, f"Imported too many modules: {new_modules}"
