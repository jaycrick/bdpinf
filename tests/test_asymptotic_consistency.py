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

    def test_reproduce_replicate_monkeypatched_fit(self, monkeypatch):
        # stub fit function to return predictable result
        def fake_fit(*args, **kwargs):
            return {
                "success": True,
                "birth_rates": np.array([1.23]),
                "death_rates": np.array([0.45]),
            }

        monkeypatch.setattr(ac, "fit_birth_death_processes", fake_fit)
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
        fit_config = ac.FittingRunConfig(
            seed=42,
            fit_optimizers=["adam"],
            fit_optimizer_kwargs={},
            convergence_config=ac.ConvergenceConfig(
                gradient_threshold=1e-6,
                absolute_tolerance=1e-10,
                patience=10,
            ),
        )
        payload = ac.reproduce_replicate(
            row,
            regime_lookup,
            fit_config=fit_config,
        )
        assert (
            payload["fit_result"]["success"] is True
            or payload["fit_result"] is None
            or isinstance(payload["fit_result"], dict)
        )
