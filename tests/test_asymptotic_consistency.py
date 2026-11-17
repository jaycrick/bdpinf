"""Tests for the asymptotic consistency analysis helpers."""

import numpy as np
import pytest

import bdpinf.analysis.asymptotic_consistency as ac


class TestAsymptoticConsistency:
    """Test asymptotic consistency sampling and fitting helpers."""

    @pytest.mark.parametrize("sampling", ["distribution", "process"])
    def test_sample_age_vector_is_reproducible_and_in_range(self, sampling):
        regime = {
            "name": "R",
            "sampling": sampling,
            "birth_rate": 0.5,
            "death_rate": 0.2,
        }
        kwargs = {"observation_time": 4.0, "simulation_index": 0, "seed": 42}

        ages = ac.sample_age_vector(regime, **kwargs)

        assert ages.size > 0  # ensure_non_extinct
        assert np.all((ages >= 0.0) & (ages <= 4.0))
        np.testing.assert_array_equal(ages, ac.sample_age_vector(regime, **kwargs))
        other = ac.sample_age_vector(regime, **{**kwargs, "simulation_index": 1})
        assert other.size != ages.size or not np.array_equal(other, ages)

    def test_reproduce_replicate_monkeypatched_fit(self, monkeypatch, make_fit_result):
        # stub fit function to return predictable result
        captured = {}

        def fake_fit(*_args, **kwargs):
            captured.update(kwargs)
            return make_fit_result([1.23], [0.45], times=[1.0])

        monkeypatch.setattr(ac, "fit", fake_fit)
        # also patch sampling to simple vector
        monkeypatch.setattr(
            ac, "sample_age_vector", lambda *_a, **_k: np.array([0.1, 0.2])
        )

        row = {
            "regime": "R",
            "sampling": "process",
            "T": 1.0,
            "simulation_index": 0,
            "replicate_seed": 123,
        }
        regime_lookup = {
            "R": {
                "name": "R",
                "sampling": "process",
                "birth_rate": 1.0,
                "death_rate": 0.5,
            },
        }
        fit_config = ac.FittingRunConfig(seed=42, method="L-BFGS-B", tol=1e-7)
        payload = ac.reproduce_replicate(
            row,
            regime_lookup,
            fit_config=fit_config,
        )
        assert payload["fit_result"].success
        assert payload["fit_result"].birth_values[0] == pytest.approx(1.23)
        assert captured["method"] == "L-BFGS-B"
        assert captured["tol"] == 1e-7
        assert captured["init"] == {"birth.v0": 1.0, "death.v0": 0.5}

    def test_fixed_time_size_grid(self):
        grid = ac.fixed_time_size_grid(1000, num_points=50)
        assert grid[0] == 1
        assert grid[-1] == 1000
        assert len(grid) == 50
        assert np.all(np.diff(grid) > 0)
        assert len(ac.fixed_time_size_grid(10, num_points=50)) == 10

    def test_population_size_quantile_matches_geometric(self):
        # Subcritical with lam=1, mu=2, T large: pi -> 1 - lam/mu = 0.5.
        n = ac.population_size_quantile(1.0, 2.0, 30.0, q=0.99)
        assert n == int(np.ceil(np.log(0.01) / np.log(0.5)))

    def test_sample_nested_ages_reproducible(self):
        regime = {"name": "R", "birth_rate": 1.0, "death_rate": 0.5}
        a = ac.sample_nested_ages(regime, 3.0, 40, 0, seed=1)
        b = ac.sample_nested_ages(regime, 3.0, 40, 0, seed=1)
        assert a.size == 40
        assert np.all((a >= 0) & (a <= 3.0))
        np.testing.assert_array_equal(a, b)

    def test_run_fixed_time_replicate_warm_starts(self, monkeypatch, make_fit_result):
        inits = []

        def fake_fit(_data, _model, **kwargs):
            inits.append(kwargs["init"])
            return make_fit_result([1.1], [0.4], times=[3.0])

        monkeypatch.setattr(ac, "fit", fake_fit)
        regime = {"name": "R", "birth_rate": 1.0, "death_rate": 0.5}
        rows = ac.run_fixed_time_replicate(
            regime,
            3.0,
            np.array([1, 5, 9]),
            0,
            fit_config=ac.FittingRunConfig(seed=1),
        )
        assert [r["population_size"] for r in rows] == [1, 5, 9]
        assert inits[0] == {"birth.v0": 1.0, "death.v0": 0.5}
        assert not isinstance(inits[1], dict)
        assert rows[0]["sampling"] == "fixed_time"

    def test_simulate_bdp_ages_given_size(self):
        from bdpinf.simulation import simulate_bdp_ages_given_size

        process = simulate_bdp_ages_given_size(
            1.0, 0.5, 3.0, 12, np.random.default_rng(0)
        )
        assert process.spine_ages is not None
        assert process.non_spine_ages is not None
        assert len(process.spine_ages) == 1
        assert len(process.non_spine_ages) == 11
        with pytest.raises(ValueError, match="population_size"):
            simulate_bdp_ages_given_size(1.0, 0.5, 3.0, 0, np.random.default_rng(0))
