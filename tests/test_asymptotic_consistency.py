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

        def fake_fit(*args, **kwargs):
            captured.update(kwargs)
            return make_fit_result([1.23], [0.45], times=[1.0])

        monkeypatch.setattr(ac, "fit", fake_fit)
        # also patch sampling to simple vector
        monkeypatch.setattr(
            ac, "sample_age_vector", lambda *a, **k: np.array([0.1, 0.2])
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
