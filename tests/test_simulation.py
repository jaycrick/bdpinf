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
            lambda *_args, **_kwargs: 1.0,
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
            lambda *_args, **_kwargs: 0.5,
        )
        monkeypatch.setattr(
            simulator_module,
            "_sample_spine_ages",
            lambda *_args, **_kwargs: [1.23],
        )
        monkeypatch.setattr(
            simulator_module,
            "_sample_non_spine_ages",
            lambda *_args, **_kwargs: [4.56],
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


# Recorded from the per-sample-envelope implementation (before the envelope was
# computed once per batch); `_sample_spine_ages(..., np.random.default_rng(123),
# n_samples=8)` must keep reproducing them exactly.
SEEDED_SPINE_AGES = {
    (0.5, 0.2, 5.0): [
        0.9218590534933485,
        4.616724990135282,
        4.4494634655559295,
        5.0,
        3.149701022948404,
        5.0,
        2.5908251842635712,
        5.0,
    ],
    (0.3, 0.3, 4.0): [
        0.7374872427946788,
        3.6933799921082255,
        3.5595707724447436,
        4.0,
        2.072660147410857,
        4.0,
        4.0,
        4.0,
    ],
    (1.0, 1.0 + 1e-13, 3.0): [
        0.16146305640666803,
        3.0,
        3.0,
        2.7700349940811693,
        0.7348938032063894,
        2.782221775575501,
        1.5544951105581426,
        3.0,
    ],
    (0.1, 0.4, 8.0): [
        0.4305681504177814,
        8.0,
        8.0,
        7.386759984216451,
        7.119141544889487,
        8.0,
        5.0395216367174465,
        8.0,
    ],
}


def _scalar_spine_envelope(terms, max_time, continuous_mass, grid_size=2048):
    """Per-point envelope exactly as the sampler computed it before batching."""
    grid = np.linspace(0.0, max_time, num=grid_size, endpoint=False)

    def density(age):
        log_lik = spine_log_density(terms, age).sum()
        if not bool(jnp.isfinite(log_lik)):
            return 0.0
        value = float(jnp.exp(log_lik))
        if value < 0.0 or not np.isfinite(value):
            return 0.0
        return value / continuous_mass

    return 1.05 * max(density(float(a)) for a in grid)


class TestSpineEnvelope:
    """Test the batched rejection-sampling envelope."""

    @pytest.mark.parametrize(
        ("birth", "death", "max_time"),
        [(0.5, 0.2, 5.0), (0.3, 0.3, 4.0), (1.0, 1.0 + 1e-13, 3.0), (0.1, 0.4, 8.0)],
    )
    def test_vectorised_envelope_matches_per_point(
        self,
        birth,
        death,
        max_time,
    ) -> None:
        terms = nested_terms(birth, death, max_time)
        continuous_mass = 0.7
        expected = _scalar_spine_envelope(terms, max_time, continuous_mass)
        actual = simulator_module._spine_envelope(terms, max_time, continuous_mass)
        assert actual == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(("birth", "death", "max_time"), list(SEEDED_SPINE_AGES))
    def test_seeded_spine_ages_unchanged(self, birth, death, max_time) -> None:
        ages = simulator_module._sample_spine_ages(
            birth,
            death,
            max_time,
            np.random.default_rng(123),
            n_samples=8,
        )
        assert ages == SEEDED_SPINE_AGES[(birth, death, max_time)]
