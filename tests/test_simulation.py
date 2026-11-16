"""Test the age samplers in `bdpinf.simulation.simulator`."""

import jax.numpy as jnp
import numpy as np
import pytest

import bdpinf.simulation.simulator as simulator_module
from bdpinf.models.math_nested import (
    nested_terms,
    non_spine_log_density,
    spine_log_density,
)
from bdpinf.simulation.simulator import simulate_bdp_ages_from_distributions


class TestBdpAgeDistributionSampler:
    """Test the distribution sampler with a stubbed RNG."""

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


class TestBdpAgeDistributionSampling:
    """Test age sampling from BDP distribution formulas."""

    @pytest.mark.slow
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

        sampled_spine_mean_ll = spine_log_density(
            nested_terms(birth_rate, death_rate, max_time), sampled_spine_arr
        ).sum() / len(sampled_spine)
        baseline_spine_mean_ll = spine_log_density(
            nested_terms(birth_rate, death_rate, max_time), baseline_spine_arr
        ).sum() / len(sampled_spine)
        sampled_non_spine_mean_ll = non_spine_log_density(
            nested_terms(birth_rate, death_rate, max_time), sampled_non_spine_arr
        ).sum() / len(sampled_non_spine)
        baseline_non_spine_mean_ll = non_spine_log_density(
            nested_terms(birth_rate, death_rate, max_time), baseline_non_spine_arr
        ).sum() / len(sampled_non_spine)

        assert jnp.isfinite(sampled_spine_mean_ll)
        assert jnp.isfinite(sampled_non_spine_mean_ll)
        assert sampled_spine_mean_ll > baseline_spine_mean_ll
        assert sampled_non_spine_mean_ll > baseline_non_spine_mean_ll
