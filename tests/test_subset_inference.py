"""Tests for NTVMR subset-building and per-subset fit helpers."""

import pickle

import numpy as np
import pandas as pd
import pytest

from inf_bp_texts.analysis.subset_inference import (
    BayesianSubsetConfig,
    ManuscriptSubset,
    SubsetFitResult,
    build_manuscript_subsets,
    fit_manuscript_subset,
    fit_manuscript_subset_bayesian,
    fit_manuscript_subsets_bayesian,
)


def _make_test_df(seed: int = 42, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    early = rng.integers(300, 900, size=n).astype(float)
    late = early + rng.integers(0, 80, size=n).astype(float)
    # 12 papyri (1), 12 majuscules (2), 8 minuscules (3), 8 lectionaries (4)
    group = np.array([1] * 12 + [2] * 12 + [3] * 8 + [4] * 8)
    section_e = np.zeros(n, dtype=int)
    section_e[:10] = 1
    section_a = np.zeros(n, dtype=int)
    section_a[10:18] = 1
    section_p = np.zeros(n, dtype=int)
    section_p[18:26] = 1
    section_r = np.zeros(n, dtype=int)
    section_r[26:30] = 1
    return pd.DataFrame(
        {
            "earlyYear": early,
            "lateYear": late,
            "midYear": (early + late) / 2.0,
            "group": group,
            "section_e": section_e,
            "section_a": section_a,
            "section_p": section_p,
            "section_r": section_r,
        }
    )


class TestBuildManuscriptSubsets:
    """Test build_manuscript_subsets."""

    def test_returns_exactly_six_subsets(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        assert len(subsets) == 6

    def test_canonical_key_order(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        assert [s.key for s in subsets] == [
            "non_lectionary",
            "lectionary",
            "section_e",
            "section_a",
            "section_p",
            "section_r",
        ]

    def test_non_lectionary_excludes_group_4(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        assert (nl.frame["group"] == 4).sum() == 0

    def test_lectionary_contains_only_group_4(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        lect = next(s for s in subsets if s.key == "lectionary")
        assert (lect.frame["group"] != 4).sum() == 0

    def test_section_subsets_are_non_lectionary(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        for key, col in [
            ("section_e", "section_e"),
            ("section_a", "section_a"),
            ("section_p", "section_p"),
            ("section_r", "section_r"),
        ]:
            s = next(x for x in subsets if x.key == key)
            assert (s.frame["group"] == 4).sum() == 0, (
                f"{key} should exclude lectionaries"
            )
            assert (s.frame[col] != 1).sum() == 0, (
                f"{key} should only contain {col}==1 rows"
            )

    def test_all_nonempty_subsets_have_resampled_year(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        for s in subsets:
            if len(s.frame) > 0:
                assert "resampledYear" in s.frame.columns, (
                    f"{s.key} missing resampledYear"
                )

    def test_labels_are_strings(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        for s in subsets:
            assert isinstance(s.label, str) and s.label

    def test_midyear_computed_if_absent(self) -> None:
        df = _make_test_df().drop(columns=["midYear"])
        subsets = build_manuscript_subsets(df)
        # Should not raise and should still produce 6 subsets
        assert len(subsets) == 6

    def test_zero_variance_dates_fallback_to_midyear(self) -> None:
        # All manuscripts have identical earlyYear == lateYear → zero-variance KDE
        # build_manuscript_subsets must not raise; resampledYear falls back to midYear.
        df = _make_test_df()
        df["earlyYear"] = 500.0
        df["lateYear"] = 500.0
        df["midYear"] = 500.0
        subsets = build_manuscript_subsets(df)
        assert len(subsets) == 6
        for s in subsets:
            if len(s.frame) > 0:
                assert "resampledYear" in s.frame.columns, (
                    f"{s.key} missing resampledYear after KDE fallback"
                )


class TestFitManuscriptSubset:
    """Test fit_manuscript_subset."""

    def test_returns_subset_fit_result(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        fit = fit_manuscript_subset(
            nl,
            epochs=[(100, 2000, "100–2000 CE")],
            start_time=100,
            optimizers=["lbfgs"],
            random_seed=42,
        )
        assert isinstance(fit, SubsetFitResult)
        assert fit.key == "non_lectionary"

    def test_epoch_results_length_matches_epochs_on_success(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        epochs = [(100, 2000, "100–2000 CE")]
        fit = fit_manuscript_subset(
            nl,
            epochs=epochs,
            start_time=100,
            optimizers=["lbfgs"],
            random_seed=42,
        )
        if fit.success:
            assert len(fit.epoch_results) == len(epochs)
        else:
            assert fit.epoch_results == []

    def test_epoch_result_keys_present_on_success(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        fit = fit_manuscript_subset(
            nl,
            epochs=[(100, 2000, "100–2000 CE")],
            start_time=100,
            optimizers=["lbfgs"],
            random_seed=42,
        )
        if fit.success and fit.epoch_results:
            ep = fit.epoch_results[0]
            for key in (
                "period",
                "start",
                "end",
                "n_manuscripts",
                "birth_rates",
                "death_rates",
                "half_life",
            ):
                assert key in ep, f"epoch_result missing key '{key}'"

    def test_empty_subset_returns_failure_without_raising(self) -> None:
        empty = ManuscriptSubset(
            key="empty",
            label="Empty",
            frame=pd.DataFrame(
                {
                    "earlyYear": pd.Series([], dtype=float),
                    "lateYear": pd.Series([], dtype=float),
                    "midYear": pd.Series([], dtype=float),
                    "group": pd.Series([], dtype=int),
                    "section_e": pd.Series([], dtype=int),
                }
            ),
        )
        fit = fit_manuscript_subset(
            empty,
            epochs=[(100, 2000, "100–2000 CE")],
            start_time=100,
            random_seed=42,
        )
        assert fit.success is False
        assert fit.epoch_results == []

    def test_missing_age_column_returns_failure(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        fit = fit_manuscript_subset(
            nl,
            epochs=[(100, 2000, "100–2000 CE")],
            start_time=100,
            age_column="nonexistent_col",
            random_seed=42,
        )
        assert fit.success is False

    def test_n_manuscripts_matches_frame_length(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        for s in subsets:
            fit = fit_manuscript_subset(
                s,
                epochs=[(100, 2000, "100–2000 CE")],
                start_time=100,
                optimizers=["lbfgs"],
                random_seed=42,
            )
            assert fit.n_manuscripts == len(s.frame)


# ---------------------------------------------------------------------------
# Helpers for Bayesian driver tests
# ---------------------------------------------------------------------------


def _make_minimal_config(subset: ManuscriptSubset) -> BayesianSubsetConfig:
    """Return a BayesianSubsetConfig using a single epoch (fast)."""
    from inf_bp_texts.inference.bayesian import BayesianPriorConfig

    return BayesianSubsetConfig(
        subset=subset,
        epochs=((100, 2000, "100–2000 CE"),),
        start_time=100,
        prior_config=BayesianPriorConfig(
            birth_loc=-6.0,
            birth_scale=2.0,
            death_loc=-6.0,
            death_scale=2.0,
        ),
    )


def _minimal_mcmc_config():  # type: ignore[return]
    from inf_bp_texts.inference.bayesian import BayesianMCMCConfig

    return BayesianMCMCConfig(
        num_samples=10,
        warmup_steps=10,
        num_chains=1,
        disable_progressbar=True,
    )


class TestBayesianSubsetConfig:
    """Test BayesianSubsetConfig picklability and structure."""

    def test_is_picklable(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        cfg = _make_minimal_config(nl)
        blob = pickle.dumps(cfg)
        restored = pickle.loads(blob)
        assert restored.subset.key == "non_lectionary"
        assert restored.epochs == ((100, 2000, "100–2000 CE"),)

    def test_frozen_raises_on_assign(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        cfg = _make_minimal_config(nl)
        with pytest.raises((AttributeError, TypeError)):
            cfg.start_time = 999  # type: ignore[misc]


class TestFitManuscriptSubsetBayesian:
    """Test fit_manuscript_subset_bayesian (single-worker, slow)."""

    @pytest.mark.slow
    def test_returns_success_dict(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        cfg = _make_minimal_config(nl)
        result = fit_manuscript_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        assert result.get("success") is True
        assert result["key"] == "non_lectionary"

    @pytest.mark.slow
    def test_result_contains_required_keys(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        cfg = _make_minimal_config(nl)
        result = fit_manuscript_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        required = {
            "key",
            "label",
            "success",
            "n_manuscripts",
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
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        cfg = _make_minimal_config(nl)
        result = fit_manuscript_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        assert isinstance(result["birth_rates"], np.ndarray)
        assert isinstance(result["death_rates"], np.ndarray)
        assert isinstance(result["posterior_samples"]["birth_rates"], np.ndarray)

    @pytest.mark.slow
    def test_posterior_samples_shape(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        nl = next(s for s in subsets if s.key == "non_lectionary")
        cfg = _make_minimal_config(nl)
        mcmc_cfg = _minimal_mcmc_config()
        result = fit_manuscript_subset_bayesian(
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
        import pandas as pd

        tiny = ManuscriptSubset(
            key="tiny",
            label="Tiny",
            frame=pd.DataFrame({"earlyYear": [300.0], "lateYear": [400.0]}),
        )
        from inf_bp_texts.inference.bayesian import BayesianPriorConfig

        cfg = BayesianSubsetConfig(
            subset=tiny,
            epochs=((100, 2000, "100–2000 CE"),),
            start_time=100,
            prior_config=BayesianPriorConfig(),
        )
        result = fit_manuscript_subset_bayesian(
            cfg,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
        )
        assert result["success"] is False


class TestFitManuscriptSubsetsBayesian:
    """Test fit_manuscript_subsets_bayesian (parallel driver, slow)."""

    @pytest.mark.slow
    def test_returns_dict_keyed_by_subset_key_n_jobs_1(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        # Use only the two largest subsets to keep the test fast
        selected = [s for s in subsets if s.key in {"non_lectionary", "section_e"}]
        configs = [_make_minimal_config(s) for s in selected]
        results = fit_manuscript_subsets_bayesian(
            configs,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=1,
        )
        assert set(results.keys()) == {"non_lectionary", "section_e"}

    @pytest.mark.slow
    def test_parallel_n_jobs_2_matches_n_jobs_1_keys(self) -> None:
        subsets = build_manuscript_subsets(_make_test_df())
        selected = [s for s in subsets if s.key in {"non_lectionary", "section_e"}]
        configs = [_make_minimal_config(s) for s in selected]
        r1 = fit_manuscript_subsets_bayesian(
            configs,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=1,
        )
        r2 = fit_manuscript_subsets_bayesian(
            configs,
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=2,
        )
        assert set(r1.keys()) == set(r2.keys())

    @pytest.mark.slow
    def test_empty_config_list_returns_empty_dict(self) -> None:
        results = fit_manuscript_subsets_bayesian(
            [],
            mcmc_config=_minimal_mcmc_config(),
            model_seed=42,
            n_jobs=1,
        )
        assert results == {}
