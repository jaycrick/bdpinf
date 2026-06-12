"""Tests for NTVMR subset-building and per-subset fit helpers."""

import numpy as np
import pandas as pd

from inf_bp_texts.analysis.subset_inference import (
    ManuscriptSubset,
    SubsetFitResult,
    build_manuscript_subsets,
    fit_manuscript_subset,
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
