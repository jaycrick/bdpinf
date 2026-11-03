"""Tests for the generic per-subset fit helpers."""

import pickle

import numpy as np
import pandas as pd
import pytest

from bdpinf.analysis.subset_fitting import (
    BayesianSubsetConfig,
    DataSubset,
    SubsetFitResult,
    fit_subset,
    fit_subset_bayesian,
    fit_subsets_bayesian,
)


def _make_subset(
    key: str = "non_lectionary", seed: int = 42, n: int = 40
) -> DataSubset:
    rng = np.random.default_rng(seed)
    year = rng.integers(100, 2000, size=n).astype(float)
    frame = pd.DataFrame({"year": year})
    return DataSubset(key=key, label=key.replace("_", " ").title(), frame=frame)


def _make_range_subset(n: int = 40) -> DataSubset:
    rng = np.random.default_rng(7)
    mid = rng.integers(200, 1900, size=n).astype(float)
    frame = pd.DataFrame({"year": mid, "early": mid - 40, "late": mid + 40})
    return DataSubset(key="ranged", label="Ranged", frame=frame)


class TestFitSubset:
    """Test fit_subset."""

    def test_fits_over_date_ranges(self) -> None:
        fit = fit_subset(
            _make_range_subset(),
            epochs=[(100, 2000, "100-2000 CE")],
            start_time=100,
            age_column="year",
            early_column="early",
            late_column="late",
            random_seed=42,
        )
        assert fit.success
        assert np.isfinite(fit.log_likelihood)

    def test_range_log_likelihood_differs_from_point_fit(self) -> None:
        kwargs = {
            "epochs": [(100, 2000, "100-2000 CE")],
            "start_time": 100,
            "age_column": "year",
            "random_seed": 42,
        }
        ranged = fit_subset(
            _make_range_subset(),
            early_column="early",
            late_column="late",
            **kwargs,
        )
        point = fit_subset(_make_range_subset(), **kwargs)
        assert ranged.log_likelihood != point.log_likelihood

    def test_one_range_column_raises(self) -> None:
        with pytest.raises(ValueError, match="together"):
            fit_subset(
                _make_range_subset(),
                epochs=[(100, 2000, "100-2000 CE")],
                start_time=100,
                age_column="year",
                early_column="early",
                random_seed=42,
            )

    def test_missing_range_column_returns_failure(self) -> None:
        fit = fit_subset(
            _make_range_subset(),
            epochs=[(100, 2000, "100-2000 CE")],
            start_time=100,
            age_column="year",
            early_column="nope",
            late_column="late",
            random_seed=42,
        )
        assert not fit.success

    def test_returns_subset_fit_result(self) -> None:
        subset = _make_subset()
        fit = fit_subset(
            subset,
            epochs=[(100, 2000, "100-2000 CE")],
            start_time=100,
            age_column="year",
            optimizers=["lbfgs"],
            random_seed=42,
        )
        assert isinstance(fit, SubsetFitResult)
        assert fit.key == "non_lectionary"

    def test_epoch_results_length_matches_epochs_on_success(self) -> None:
        subset = _make_subset()
        epochs = [(100, 2000, "100-2000 CE")]
        fit = fit_subset(
            subset,
            epochs=epochs,
            start_time=100,
            age_column="year",
            optimizers=["lbfgs"],
            random_seed=42,
        )
        if fit.success:
            assert len(fit.epoch_results) == len(epochs)
        else:
            assert fit.epoch_results == []

    def test_epoch_result_keys_present_on_success(self) -> None:
        subset = _make_subset()
        fit = fit_subset(
            subset,
            epochs=[(100, 2000, "100-2000 CE")],
            start_time=100,
            age_column="year",
            optimizers=["lbfgs"],
            random_seed=42,
        )
        if fit.success and fit.epoch_results:
            ep = fit.epoch_results[0]
            for key in (
                "period",
                "start",
                "end",
                "n_individuals",
                "birth_rates",
                "death_rates",
                "half_life",
            ):
                assert key in ep, f"epoch_result missing key '{key}'"

    def test_empty_subset_returns_failure_without_raising(self) -> None:
        empty = DataSubset(
            key="empty",
            label="Empty",
            frame=pd.DataFrame({"year": pd.Series([], dtype=float)}),
        )
        fit = fit_subset(
            empty,
            epochs=[(100, 2000, "100-2000 CE")],
            start_time=100,
            age_column="year",
            random_seed=42,
        )
        assert fit.success is False
        assert fit.epoch_results == []

    def test_missing_age_column_returns_failure(self) -> None:
        subset = _make_subset()
        fit = fit_subset(
            subset,
            epochs=[(100, 2000, "100-2000 CE")],
            start_time=100,
            age_column="nonexistent_col",
            random_seed=42,
        )
        assert fit.success is False

    def test_n_individuals_matches_frame_length(self) -> None:
        subset = _make_subset()
        fit = fit_subset(
            subset,
            epochs=[(100, 2000, "100-2000 CE")],
            start_time=100,
            age_column="year",
            optimizers=["lbfgs"],
            random_seed=42,
        )
        assert fit.n_individuals == len(subset.frame)


# ---------------------------------------------------------------------------
# Helpers for Bayesian driver tests
# ---------------------------------------------------------------------------


def _make_minimal_config(subset: DataSubset) -> BayesianSubsetConfig:
    """Return a BayesianSubsetConfig using a single epoch (fast)."""
    from bdpinf.inference.bayesian import BayesianPriorConfig

    return BayesianSubsetConfig(
        subset=subset,
        epochs=((100, 2000, "100-2000 CE"),),
        start_time=100,
        prior_config=BayesianPriorConfig(
            birth_loc=-6.0,
            birth_scale=2.0,
            death_loc=-6.0,
            death_scale=2.0,
        ),
        age_column="year",
    )


def _minimal_mcmc_config():  # type: ignore[return]
    from bdpinf.inference.bayesian import BayesianMCMCConfig

    return BayesianMCMCConfig(
        num_samples=10,
        warmup_steps=10,
        num_chains=1,
        disable_progressbar=True,
    )


class TestBayesianSubsetConfig:
    """Test BayesianSubsetConfig picklability and structure."""

    def test_is_picklable(self) -> None:
        subset = _make_subset()
        cfg = _make_minimal_config(subset)
        blob = pickle.dumps(cfg)
        restored = pickle.loads(blob)
        assert restored.subset.key == "non_lectionary"
        assert restored.epochs == ((100, 2000, "100-2000 CE"),)

    def test_frozen_raises_on_assign(self) -> None:
        subset = _make_subset()
        cfg = _make_minimal_config(subset)
        with pytest.raises((AttributeError, TypeError)):
            cfg.start_time = 999  # type: ignore[misc]


class TestFitSubsetBayesian:
    """Test fit_subset_bayesian (single-worker, slow)."""

    @pytest.mark.slow
    def test_returns_success_dict(self) -> None:
        subset = _make_subset()
        cfg = _make_minimal_config(subset)
        result = fit_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        assert result.get("success") is True
        assert result["key"] == "non_lectionary"

    @pytest.mark.slow
    def test_result_contains_required_keys(self) -> None:
        subset = _make_subset()
        cfg = _make_minimal_config(subset)
        result = fit_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        required = {
            "key",
            "label",
            "success",
            "n_individuals",
            "epoch_names",
            "birth_rates",
            "death_rates",
            "posterior_samples",
            "diagnostics",
            "num_divergences",
            "elapsed",
        }
        assert required <= result.keys()

    @pytest.mark.slow
    def test_all_arrays_are_numpy(self) -> None:
        """No JAX arrays should cross the function boundary."""
        subset = _make_subset()
        cfg = _make_minimal_config(subset)
        result = fit_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        assert isinstance(result["birth_rates"], np.ndarray)
        assert isinstance(result["death_rates"], np.ndarray)
        assert isinstance(result["posterior_samples"]["birth_rates"], np.ndarray)

    @pytest.mark.slow
    def test_posterior_samples_shape(self) -> None:
        subset = _make_subset()
        cfg = _make_minimal_config(subset)
        mcmc_cfg = _minimal_mcmc_config()
        result = fit_subset_bayesian(
            cfg,
            mcmc_config=mcmc_cfg,
            model_seed=42,
        )
        bs = result["posterior_samples"]["birth_rates"]
        # shape: [n_samples, n_epochs]
        assert bs.ndim == 2
        assert bs.shape[1] == len(cfg.epochs)
        assert bs.shape[0] == mcmc_cfg.num_samples

    def test_insufficient_data_returns_failure(self) -> None:
        """Subset with <2 rows should return success=False without raising."""
        tiny = DataSubset(
            key="tiny",
            label="Tiny",
            frame=pd.DataFrame({"year": [300.0]}),
        )
        from bdpinf.inference.bayesian import BayesianPriorConfig

        cfg = BayesianSubsetConfig(
            subset=tiny,
            epochs=((100, 2000, "100-2000 CE"),),
            start_time=100,
            prior_config=BayesianPriorConfig(),
            age_column="year",
        )
        result = fit_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        assert result["success"] is False


class TestFitSubsetsBayesian:
    """Test fit_subsets_bayesian (parallel driver, slow)."""

    @pytest.mark.slow
    def test_returns_dict_keyed_by_subset_key_n_jobs_1(self) -> None:
        selected = [_make_subset("a", seed=1), _make_subset("b", seed=2)]
        configs = [_make_minimal_config(s) for s in selected]
        results = fit_subsets_bayesian(
            configs,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=1,
        )
        assert set(results.keys()) == {"a", "b"}

    @pytest.mark.slow
    def test_parallel_n_jobs_2_matches_n_jobs_1_keys(self) -> None:
        selected = [_make_subset("a", seed=1), _make_subset("b", seed=2)]
        configs = [_make_minimal_config(s) for s in selected]
        r1 = fit_subsets_bayesian(
            configs,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=1,
        )
        r2 = fit_subsets_bayesian(
            configs,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=2,
        )
        assert set(r1.keys()) == set(r2.keys())

    @pytest.mark.slow
    def test_empty_config_list_returns_empty_dict(self) -> None:
        results = fit_subsets_bayesian(
            [],
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=1,
        )
        assert results == {}
