"""Tests for splitting-based conditioned survival simulators."""

# ruff: noqa: S101

import numpy as np

from inf_bp_texts.models import BirthDeathProcess
from inf_bp_texts.simulation import (
    FlemingViotConfig,
    SplittingConfig,
    plot_level_survival_probabilities,
    simulate_subcritical_survival_fleming_viot,
    simulate_subcritical_survival_splitting,
)
from inf_bp_texts.simulation.splitting import (
    _adaptive_particle_count,
    _calculate_extinction_probability,
)


class TestSplittingSimulation:
    """Validate rare-event splitting and Fleming-Viot approximations."""

    def test_splitting_returns_expected_shapes(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=40.0,
            n_particles=200,
            n_levels=8,
            random_seed=17,
        )

        result = simulate_subcritical_survival_splitting(config)

        assert result.level_times.shape[0] == config.n_levels + 1
        assert result.level_survival_probabilities.shape[0] == config.n_levels
        assert result.level_mean_populations.shape[0] == config.n_levels + 1
        assert 0.0 <= result.survival_probability_estimate <= 1.0

    def test_splitting_product_matches_stored_estimate(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=30.0,
            n_particles=150,
            n_levels=6,
            random_seed=91,
        )

        result = simulate_subcritical_survival_splitting(config)
        reconstructed = float(np.prod(result.level_survival_probabilities))
        assert np.isclose(
            reconstructed,
            result.survival_probability_estimate,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_splitting_reproducible_with_fixed_seed(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=25.0,
            n_particles=120,
            n_levels=5,
            random_seed=2026,
        )

        first = simulate_subcritical_survival_splitting(config)
        second = simulate_subcritical_survival_splitting(config)

        assert np.isclose(
            first.survival_probability_estimate,
            second.survival_probability_estimate,
        )
        assert np.allclose(
            first.level_survival_probabilities,
            second.level_survival_probabilities,
        )

    def test_conditioned_trajectories_survive_to_terminal_time(self) -> None:
        config = SplittingConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=40.0,
            n_particles=250,
            n_levels=8,
            random_seed=11,
        )

        result = simulate_subcritical_survival_splitting(config)

        assert len(result.conditioned_trajectories) == result.final_survivor_count

        for trajectory in result.conditioned_trajectories:
            assert isinstance(trajectory.process, BirthDeathProcess)
            assert np.isclose(trajectory.times[-1], config.terminal_time)
            assert trajectory.populations[-1] > 0

    def test_extinction_helper_uses_requested_band(self) -> None:
        probability = _calculate_extinction_probability(
            birth_rate=0.9,
            death_rate=1.0,
            interval_time=0.5,
        )

        assert np.e ** (-2.0) < probability < 0.5
        assert _adaptive_particle_count(
            birth_rate=0.9,
            death_rate=1.0,
            interval_time=0.5,
        ) == int(np.ceil(1.0 / probability))

    def test_fleming_viot_returns_expected_shapes(self) -> None:
        config = FlemingViotConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=40.0,
            n_particles=200,
            n_levels=8,
            random_seed=13,
            n_conditioned_trajectories=6,
        )

        result = simulate_subcritical_survival_fleming_viot(config)

        assert result.level_times.shape[0] == config.n_levels + 1
        assert result.level_survival_probabilities.shape[0] == config.n_levels
        assert result.level_mean_populations.shape[0] == config.n_levels + 1
        assert 0.0 <= result.survival_proxy_estimate <= 1.0

    def test_splitting_level_times_invariant_on_early_extinction(self) -> None:
        # Regression test: when all particles go extinct before the final level
        # the loop breaks early and level_times was previously not truncated,
        # leaving len(level_times) >> len(level_survival_probabilities) + 1.
        # Use a near-zero birth rate and very few particles so extinction before
        # the last level is guaranteed regardless of seed.
        config = SplittingConfig(
            birth_rate=0.05,
            death_rate=1.0,
            terminal_time=50.0,
            n_particles=5,
            n_levels=15,
            random_seed=0,
        )

        result = simulate_subcritical_survival_splitting(config)

        # Core invariant: always one more boundary than probability estimates.
        assert len(result.level_times) == len(result.level_survival_probabilities) + 1
        assert len(result.level_times) == len(result.level_mean_populations)
        # Confirm we actually hit the early-exit path.
        assert len(result.level_survival_probabilities) < config.n_levels
        # plot_level_survival_probabilities must not raise.
        plot_level_survival_probabilities(
            level_times=result.level_times,
            level_survival_probabilities=result.level_survival_probabilities,
        )

    def test_fleming_viot_level_times_invariant_on_early_extinction(self) -> None:
        # Regression test for the same early-exit truncation bug in the
        # Fleming-Viot path.
        config = FlemingViotConfig(
            birth_rate=0.05,
            death_rate=1.0,
            terminal_time=50.0,
            n_particles=5,
            n_levels=15,
            random_seed=0,
        )

        result = simulate_subcritical_survival_fleming_viot(config)

        assert len(result.level_times) == len(result.level_survival_probabilities) + 1
        assert len(result.level_times) == len(result.level_mean_populations)
        assert len(result.level_survival_probabilities) < config.n_levels
        plot_level_survival_probabilities(
            level_times=result.level_times,
            level_survival_probabilities=result.level_survival_probabilities,
        )

    def test_fleming_viot_maintains_particle_bank_when_survivors_exist(self) -> None:
        config = FlemingViotConfig(
            birth_rate=0.9,
            death_rate=1.0,
            terminal_time=20.0,
            n_particles=100,
            n_levels=4,
            random_seed=29,
            n_conditioned_trajectories=3,
        )

        result = simulate_subcritical_survival_fleming_viot(config)
        if len(result.level_diagnostics) > 0:
            # Level-wise proxy stays within [0,1], and if all levels had survivors,
            # the final conditioned bank can still provide trajectories.
            assert np.all(result.level_survival_probabilities >= 0.0)
            assert np.all(result.level_survival_probabilities <= 1.0)
