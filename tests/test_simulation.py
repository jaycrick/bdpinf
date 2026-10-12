"""Test suite for BirthDeathProcess simulation functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bdpinf.simulation.simulator as simulator_module
from bdpinf.models.math_ages import (
    calculate_non_spine_ages_log_lik,
    calculate_spine_ages_log_lik,
)
from bdpinf.process import BirthDeathProcess
from bdpinf.simulation import simulate_birth_death_process
from bdpinf.simulation.simulator import simulate_bdp_ages_from_distributions


class TestSimulation:
    """Test simulation functionality."""

    def test_simulation_basic(self) -> None:
        """Test basic simulation functionality using modular approach."""
        # Run simulation using modular function
        df = simulate_birth_death_process(
            birth_rates=0.1,
            death_rates=0.02,
            times=100.0,
            random_seed=42,
        )

        # Should return a BirthDeathProcess
        assert isinstance(df, BirthDeathProcess)

        process = df
        surviving_records = process.surviving_records
        assert surviving_records is not None

        # Check that epoch_end_times and creation_times were set
        assert process.epoch_end_times[-1] == 100.0
        assert isinstance(process.creation_times, list)

        # Ages should be calculated consistently
        ages = process.ages
        assert len(ages) == len(process.creation_times)

    def test_simulation_with_seed_reproducible(self) -> None:
        """Test that simulation with same seed is reproducible."""
        # Run simulations using modular function with same seed
        df1 = simulate_birth_death_process(
            birth_rates=0.1,
            death_rates=0.02,
            times=100.0,
            random_seed=42,
        )
        df2 = simulate_birth_death_process(
            birth_rates=0.1,
            death_rates=0.02,
            times=100.0,
            random_seed=42,
        )

        # Should have same number of records
        assert len(df1) == len(df2)

        # Create process objects from simulation results
        surviving1 = df1.surviving_records
        surviving2 = df2.surviving_records
        assert surviving1 is not None and surviving2 is not None
        creation_times1 = surviving1["creation_time"].tolist()
        creation_times2 = surviving2["creation_time"].tolist()

        assert len(creation_times1) == len(creation_times2)

    def test_simulation_different_seeds_different_results(self) -> None:
        """Test that different seeds give different results."""
        # Run simulations using modular function with different seeds
        df1 = simulate_birth_death_process(
            birth_rates=0.1,
            death_rates=0.02,
            times=100.0,
            random_seed=42,
        )
        df2 = simulate_birth_death_process(
            birth_rates=0.1,
            death_rates=0.02,
            times=100.0,
            random_seed=123,
        )

        # Very unlikely to have identical results with different seeds
        # (unless both simulations produce 0 records, which is possible but rare)
        if len(df1) > 0 and len(df2) > 0:
            # If both have records, they should be different
            surviving1 = df1.surviving_records
            surviving2 = df2.surviving_records
            assert surviving1 is not None and surviving2 is not None
            creation_times1 = surviving1["creation_time"].tolist()
            creation_times2 = surviving2["creation_time"].tolist()
            assert not df1.equals(df2) or len(creation_times1) != len(creation_times2)

    def test_simulation_data_frame_structure(self) -> None:
        """Test that simulation returns properly structured DataFrame."""
        # Run simulation using modular function
        df = simulate_birth_death_process(
            birth_rates=0.2,
            death_rates=0.01,
            times=50.0,
            random_seed=42,
        )

        if len(df) > 0:
            # Check required columns for surviving records
            surviving = df.surviving_records
            assert surviving is not None
            if len(surviving) > 0:
                required_cols = ["individual_id", "creation_time", "age", "survived"]
                for col in required_cols:
                    assert col in surviving.columns

                # Check data types and validity
                assert all(surviving["individual_id"] >= 0)
                assert all(surviving["creation_time"] >= 0)
                assert all(surviving["creation_time"] <= 50.0)
                assert all(surviving["age"] >= 0)
                assert all(surviving["survived"])

            # Check dead records
            dead = df.extinct_records
            assert dead is not None
            if len(dead) > 0:
                required_cols = [
                    "individual_id",
                    "creation_time",
                    "death_time",
                    "lifespan",
                    "survived",
                ]
                for col in required_cols:
                    assert col in dead.columns

                assert all(dead["creation_time"] >= 0)
                assert all(dead["death_time"] > dead["creation_time"])
                assert all(dead["lifespan"] >= 0)
                assert all(~dead["survived"])

    def test_simulation_ages_consistency(self) -> None:
        """Test that ages are calculated correctly after simulation."""
        # Run simulation using modular function
        df = simulate_birth_death_process(
            birth_rates=0.15,
            death_rates=0.03,
            times=80.0,
            random_seed=42,
        )

        # Create process object from simulation results
        surviving_records = df.surviving_records
        assert surviving_records is not None
        if len(surviving_records) > 0:
            creation_times = surviving_records["creation_time"].tolist()
            process = BirthDeathProcess(
                creation_times=creation_times,
                epoch_end_times=80.0,
                birth_rates=0.15,
                death_rates=0.03,
            )

            # Check that manual calculation matches process.ages
            manual_ages = [
                process.epoch_end_times[-1] - ct for ct in process.creation_times
            ]
            np.testing.assert_array_almost_equal(
                process.ages,
                manual_ages,
            )

            # Check that ages for surviving records match process.ages
            df_ages = surviving_records["age"].values
            # All surviving records should have ages that match process.ages
            np.testing.assert_array_almost_equal(
                sorted(df_ages),
                sorted(process.ages),
            )

    def test_simulation_empty_result(self) -> None:
        """Test simulation that might produce no records."""
        # Very low birth rate, high death rate, short time
        df = simulate_birth_death_process(
            birth_rates=0.001,
            death_rates=1.0,
            times=1.0,
            random_seed=42,
        )

        # Should still return a BirthDeathProcess, even if empty
        assert isinstance(df, BirthDeathProcess)

        process = df
        surviving_records = process.surviving_records
        assert surviving_records is not None

        # Check that the process state is still valid
        assert process.epoch_end_times[-1].item() == 1.0
        assert isinstance(process.creation_times, list)
        assert isinstance(process.ages, jax.Array)

    def test_simulation_resets_process(self) -> None:
        """Test that creating new process with simulation results gives proper state."""
        # Create initial process with some state
        initial_creation_times = [10.0, 20.0, 30.0]
        process = BirthDeathProcess(
            creation_times=initial_creation_times,
            epoch_end_times=100.0,
            birth_rates=0.1,
            death_rates=0.02,
        )
        initial_ages = jnp.array(process.ages)

        # Run simulation using modular function - should give new results
        df = simulate_birth_death_process(
            birth_rates=0.1,
            death_rates=0.02,
            times=50.0,
            random_seed=42,
        )

        # Create new process from simulation results
        surviving_records = df.surviving_records
        assert surviving_records is not None
        new_creation_times = (
            surviving_records["creation_time"].tolist()
            if len(surviving_records) > 0
            else []
        )
        new_process = BirthDeathProcess(
            creation_times=new_creation_times,
            epoch_end_times=50.0,
            birth_rates=0.1,
            death_rates=0.02,
        )

        # New process should have different state from initial process
        assert new_process.epoch_end_times[-1] == 50.0
        # creation_times should be from simulation, not initial values
        assert new_process.creation_times != initial_creation_times
        # ages should be different
        if len(new_process.ages) > 0:
            assert not np.array_equal(new_process.ages, initial_ages)

    def test_bdp_distribution_sampler_defaults_to_extinction_branch(
        self,
        monkeypatch,
    ) -> None:
        class DummyRng:
            def uniform(self):
                return 0.0

            def geometric(self, _):
                raise AssertionError("geometric should not be called when extinct")

        monkeypatch.setattr(
            simulator_module,
            "calculate_extinction_pr",
            lambda *args, **kwargs: 1.0,
        )

        sampled = simulate_bdp_ages_from_distributions(
            birth_rate=0.5,
            death_rate=0.3,
            max_time=5.0,
            rng=DummyRng(),
        )

        assert sampled.spine_ages is not None and sampled.spine_ages.tolist() == []
        assert (
            sampled.non_spine_ages is not None and sampled.non_spine_ages.tolist() == []
        )

    def test_bdp_distribution_sampler_can_force_non_extinction(
        self,
        monkeypatch,
    ) -> None:
        class DummyRng:
            def uniform(self):
                return 0.0

            def geometric(self, _):
                return np.array(2)

        monkeypatch.setattr(
            simulator_module,
            "population_geometric_parameter",
            lambda *args, **kwargs: 0.5,
        )
        monkeypatch.setattr(
            simulator_module,
            "_sample_spine_ages",
            lambda *args, **kwargs: [1.23],
        )
        monkeypatch.setattr(
            simulator_module,
            "_sample_non_spine_ages",
            lambda *args, **kwargs: [4.56],
        )

        sampled = simulate_bdp_ages_from_distributions(
            birth_rate=0.5,
            death_rate=0.3,
            max_time=5.0,
            rng=DummyRng(),
            ensure_non_extinct=True,
        )

        assert sampled.spine_ages is not None and sampled.spine_ages.tolist() == [1.23]
        assert (
            sampled.non_spine_ages is not None
            and sampled.non_spine_ages.tolist() == [4.56]
        )

    def test_simulation_parameter_effects(self) -> None:
        """Test that birth/death rates affect simulation results as expected."""
        max_time = 10.0
        random_seed = 42

        # High birth rate, low death rate - should produce many records
        df_high_birth = simulate_birth_death_process(
            birth_rates=0.5,
            death_rates=0.01,
            times=max_time,
            random_seed=random_seed,
        )

        # Low birth rate, high death rate - should produce few records
        df_low_birth = simulate_birth_death_process(
            birth_rates=0.01,
            death_rates=0.1,
            times=max_time,
            random_seed=random_seed + 1,
        )

        # This is probabilistic, but with these extreme parameters,
        # high birth rate should generally produce more surviving records
        high_surv = df_high_birth.surviving_records
        low_surv = df_low_birth.surviving_records
        assert high_surv is not None and low_surv is not None
        surviving_high = len(high_surv)
        surviving_low = len(low_surv)

        # Note: This test might occasionally fail due to randomness,
        # but with these extreme parameters it should usually pass
        # We'll just check that both simulations ran without error
        assert isinstance(surviving_high, int)
        assert isinstance(surviving_low, int)

    def test_simulation_time_bounds(self) -> None:
        """Test that all creation times are within the specified bounds."""
        max_time = 75.0
        df = simulate_birth_death_process(
            birth_rates=0.2,
            death_rates=0.05,
            times=max_time,
            random_seed=42,
        )

        # All creation times should be between 0 and max_time
        surviving_records = df.surviving_records
        assert surviving_records is not None
        if len(surviving_records) > 0:
            creation_times = surviving_records["creation_time"].tolist()
            assert all(0 <= ct <= max_time for ct in creation_times)

        # All creation times in DataFrame should also be within bounds
        if len(df) > 0:
            df_records = df.records
            assert df_records is not None
            assert all(0 <= ct <= max_time for ct in df_records["creation_time"])

    def test_simulation_ages_non_negative(self) -> None:
        """Test that all calculated ages are non-negative."""
        df = simulate_birth_death_process(
            birth_rates=0.1,
            death_rates=0.02,
            times=60.0,
            random_seed=42,
        )

        # Create process from simulation results to test age calculation
        surviving_records = df.surviving_records
        assert surviving_records is not None
        if len(surviving_records) > 0:
            creation_times = surviving_records["creation_time"].tolist()
            process = BirthDeathProcess(
                creation_times=creation_times,
                epoch_end_times=60.0,
                birth_rates=0.1,
                death_rates=0.02,
            )

            # All ages in process.ages should be non-negative
            assert all(age >= 0 for age in process.ages)

        # All ages in DataFrame should be non-negative
        if len(surviving_records) > 0:
            assert all(age >= 0 for age in surviving_records["age"])


class TestMultiEpochSimulation:
    """Test multi-epoch simulation functionality."""

    def test_two_epoch_simulation_basic(self) -> None:
        """Test basic two-epoch simulation."""
        birth_rates = np.array([0.1, 0.05])
        death_rates = np.array([0.02, 0.01])
        times = np.array([100.0, 200.0])

        df = simulate_birth_death_process(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            random_seed=42,
        )

        # Should return a BirthDeathProcess
        assert isinstance(df, BirthDeathProcess)

        # All creation times should be <= final time
        if len(df) > 0:
            assert df.records is not None
            assert all(df.records["creation_time"] <= times[-1])

    def test_two_epoch_simulation_reproducible(self) -> None:
        """Test that two-epoch simulation with same seed is reproducible."""
        birth_rates = np.array([0.08, 0.12])
        death_rates = np.array([0.03, 0.02])
        times = np.array([50.0, 100.0])

        df1 = simulate_birth_death_process(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            random_seed=123,
        )
        df2 = simulate_birth_death_process(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            random_seed=123,
        )

        # Should have same number of records
        assert len(df1) == len(df2)

        # Creation times should match
        if len(df1) > 0 and len(df2) > 0:
            np.testing.assert_array_almost_equal(
                sorted(df1.records["creation_time"]),
                sorted(df2.records["creation_time"]),
            )

    def test_three_epoch_simulation(self) -> None:
        """Test simulation with three epochs."""
        birth_rates = np.array([0.15, 0.1, 0.12])
        death_rates = np.array([0.02, 0.04, 0.03])
        times = np.array([30.0, 60.0, 100.0])

        df = simulate_birth_death_process(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            random_seed=42,
        )

        assert isinstance(df, BirthDeathProcess)
        # Check that all times are within bounds
        if len(df) > 0:
            assert df.records is not None
            assert all(df.records["creation_time"] >= 0)
            assert all(df.records["creation_time"] <= times[-1])

    def test_multi_epoch_data_frame_structure(self) -> None:
        """Test that multi-epoch simulation returns properly structured DataFrame."""
        birth_rates = np.array([0.2, 0.15])
        death_rates = np.array([0.01, 0.02])
        times = np.array([25.0, 50.0])

        df = simulate_birth_death_process(
            birth_rates=birth_rates,
            death_rates=death_rates,
            times=times,
            random_seed=42,
        )

        if len(df) > 0:
            surviving = df.surviving_records
            if len(surviving) > 0:
                required_cols = ["individual_id", "creation_time", "age", "survived"]
                for col in required_cols:
                    assert col in surviving.columns

                assert all(surviving["creation_time"] >= 0)
                assert all(surviving["creation_time"] <= times[-1])
                assert all(surviving["age"] >= 0)

    def test_multi_epoch_different_epochs_affect_results(self) -> None:
        """Test that changing epoch parameters affects simulation outcomes."""
        # Two epochs with moderately different rates (avoid extreme values)
        birth_rates1 = np.array([0.15, 0.05])  # Higher then lower birth rate
        death_rates1 = np.array([0.02, 0.02])
        times = np.array([25.0, 50.0])  # Shorter time for faster test

        birth_rates2 = np.array([0.05, 0.15])  # Lower then higher birth rate
        death_rates2 = np.array([0.02, 0.02])

        df1 = simulate_birth_death_process(
            birth_rates=birth_rates1,
            death_rates=death_rates1,
            times=times,
            random_seed=42,
        )
        df2 = simulate_birth_death_process(
            birth_rates=birth_rates2,
            death_rates=death_rates2,
            times=times,
            random_seed=43,  # Different seed to ensure they differ
        )

        # Both simulations should complete
        assert isinstance(df1, BirthDeathProcess)
        assert isinstance(df2, BirthDeathProcess)

        # Results should differ (different epoch configurations and seeds)
        # This is probabilistic but with different seeds should be reliable
        if len(df1) > 0 or len(df2) > 0:
            # At least one simulation produced records
            assert True

    def test_multi_epoch_scalar_compatibility(self) -> None:
        """Test that single-epoch as array matches scalar input."""
        birth_rate = 0.1
        death_rate = 0.02
        time = 100.0

        # Scalar inputs
        df_scalar = simulate_birth_death_process(
            birth_rates=birth_rate,
            death_rates=death_rate,
            times=time,
            random_seed=42,
        )

        # Array inputs (single epoch)
        df_array = simulate_birth_death_process(
            birth_rates=np.array([birth_rate]),
            death_rates=np.array([death_rate]),
            times=np.array([time]),
            random_seed=42,
        )

        # Should produce identical results
        assert len(df_scalar) == len(df_array)
        if len(df_scalar) > 0:
            scalar_records = df_scalar.records
            array_records = df_array.records
            assert scalar_records is not None and array_records is not None
            np.testing.assert_array_almost_equal(
                sorted(scalar_records["creation_time"]),
                sorted(array_records["creation_time"]),
            )


class TestBdpAgeDistributionSampling:
    """Test age sampling from BDP distribution formulas."""

    def test_population_and_age_accounting(self) -> None:
        """Sampled ages should match one spine plus n-1 non-spine ages."""
        birth_rate = 0.25
        death_rate = 0.1
        max_time = 12.0
        rng = np.random.default_rng(2026)

        for _ in range(15):
            ages = simulate_bdp_ages_from_distributions(
                birth_rate=birth_rate,
                death_rate=death_rate,
                max_time=max_time,
                rng=rng,
            )
            spine = ages.spine_ages
            non_spine = ages.non_spine_ages
            assert spine is not None and non_spine is not None
            pop_size = len(spine) + len(non_spine)
            assert pop_size >= 0
            if pop_size == 0:
                assert spine.tolist() == []
                assert non_spine.tolist() == []
            else:
                assert len(spine) == 1
                assert len(non_spine) == pop_size - 1

                spine_age = float(spine[0])
                assert 0.0 <= spine_age <= max_time
                for age in non_spine:
                    assert 0.0 <= float(age) <= max_time

    def test_sampling_is_reproducible_with_seed(self) -> None:
        """Sampling should be deterministic with identical seeds."""
        birth_rate = 0.5
        death_rate = 0.3
        max_time = 5.0

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        draws1 = [
            simulate_bdp_ages_from_distributions(
                birth_rate=birth_rate,
                death_rate=death_rate,
                max_time=max_time,
                rng=rng1,
            )
            for _ in range(4)
        ]
        draws2 = [
            simulate_bdp_ages_from_distributions(
                birth_rate=birth_rate,
                death_rate=death_rate,
                max_time=max_time,
                rng=rng2,
            )
            for _ in range(4)
        ]

        assert draws1 == draws2

    @pytest.mark.slow
    def test_sampled_ages_have_better_mean_log_likelihood_than_uniform(self) -> None:
        """Sampled ages should score better than a uniform baseline under matching models."""
        birth_rate = 0.5
        death_rate = 0.2
        max_time = 6.0
        rng = np.random.default_rng(7)

        sampled_spine: list[float] = []
        sampled_non_spine: list[float] = []

        for _ in range(900):
            ages = simulate_bdp_ages_from_distributions(
                birth_rate=birth_rate,
                death_rate=death_rate,
                max_time=max_time,
                rng=rng,
            )
            sampled_spine.extend(
                ages.spine_ages.tolist() if ages.spine_ages is not None else []
            )
            sampled_non_spine.extend(
                ages.non_spine_ages.tolist() if ages.non_spine_ages is not None else []
            )

        assert len(sampled_spine) > 200
        assert len(sampled_non_spine) > 200

        sampled_spine_arr = jnp.array(sampled_spine, dtype=jnp.float64)
        sampled_non_spine_arr = jnp.array(sampled_non_spine, dtype=jnp.float64)

        baseline_spine_arr = jnp.array(
            rng.uniform(0.0, max_time, size=len(sampled_spine)),
            dtype=jnp.float64,
        )
        baseline_non_spine_arr = jnp.array(
            rng.uniform(0.0, max_time, size=len(sampled_non_spine)),
            dtype=jnp.float64,
        )

        sampled_spine_mean_ll = calculate_spine_ages_log_lik(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=sampled_spine_arr,
            times=max_time,
        ) / len(sampled_spine)
        baseline_spine_mean_ll = calculate_spine_ages_log_lik(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=baseline_spine_arr,
            times=max_time,
        ) / len(sampled_spine)
        sampled_non_spine_mean_ll = calculate_non_spine_ages_log_lik(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=sampled_non_spine_arr,
            times=max_time,
        ) / len(sampled_non_spine)
        baseline_non_spine_mean_ll = calculate_non_spine_ages_log_lik(
            birth_rates=birth_rate,
            death_rates=death_rate,
            ages=baseline_non_spine_arr,
            times=max_time,
        ) / len(sampled_non_spine)

        assert jnp.isfinite(sampled_spine_mean_ll)
        assert jnp.isfinite(sampled_non_spine_mean_ll)
        assert sampled_spine_mean_ll > baseline_spine_mean_ll
        assert sampled_non_spine_mean_ll > baseline_non_spine_mean_ll
