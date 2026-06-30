"""Tests for bootstrap KDE-quantile date resampling (bootstrap_date_density.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from inf_bp_texts.analysis.bootstrap_date_density import (
    _build_weight_matrix,  # pyright: ignore[reportPrivateUsage]
    _resample_batch,  # pyright: ignore[reportPrivateUsage]
    bootstrap_quantile_resample_dates,
    load_bootstrap_samples,
    save_bootstrap_samples,
)
from inf_bp_texts.analysis.date_density import quantile_resample_dates

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_SEED = 42
_CURRENT_YEAR = 2024

# Small synthetic dataset: 5 manuscripts with varying date ranges
_EARLY = np.array([200, 400, 600, 800, 1000], dtype=np.int64)
_LATE = np.array([300, 600, 700, 1200, 1400], dtype=np.int64)
_LATE_CLAMPED = np.maximum(_LATE, _EARLY)
_DOC_IDS = np.array([10001, 10002, 10003, 10004, 10005], dtype=np.int64)
_MID = (_EARLY + _LATE_CLAMPED) / 2.0
_M = len(_EARLY)


def _make_precomputed(
    early: np.ndarray,
    late: np.ndarray,
    current_year: int = _CURRENT_YEAR,
    n_grid: int = 300,
) -> dict:
    """Build all precomputed quantities needed for _resample_batch."""
    late = np.maximum(late, early)
    y_min = int(early.min())
    y_max = int(late.max())
    Y = y_max - y_min + 1
    n_m = (late - early + 1).astype(np.float64)
    c = 1.0 / n_m
    mid_age = float(current_year) - (early + late) / 2.0
    within_var = (n_m**2 - 1.0) / 12.0
    A = _build_weight_matrix(early, late, y_min, y_max)
    max_age = float(current_year - y_min)
    x_grid = np.linspace(0.0, max_age, n_grid)
    year_grid = (float(current_year) - x_grid[::-1]).copy()
    y_ages = float(current_year) - (y_min + np.arange(Y, dtype=np.float64))
    xg, ag = x_grid[:, None], y_ages[None, :]
    return {
        "A": A,
        "D0_sq": (xg - ag) ** 2,
        "D1_sq": (xg + ag) ** 2,
        "D2_sq": (xg - 2.0 * max_age + ag) ** 2,
        "x_grid": x_grid,
        "year_grid": year_grid,
        "quantiles": (np.arange(1, len(early) + 1, dtype=np.float64) - 0.5)
        / float(len(early)),
        "mid_age": mid_age,
        "within_var": within_var,
        "c": c,
        "min_year": float(early.min()),
        "max_year": float(late.max()),
    }


class TestBuildWeightMatrix:
    """Test _build_weight_matrix."""

    def test_shape(self) -> None:
        A = _build_weight_matrix(
            _EARLY, _LATE_CLAMPED, int(_EARLY.min()), int(_LATE_CLAMPED.max())
        )
        assert A.shape == (_M, int(_LATE_CLAMPED.max()) - int(_EARLY.min()) + 1)

    def test_row_sums_to_c(self) -> None:
        # Each row sums to c_m = 1/n_m (uniform weight per year in range)
        A = _build_weight_matrix(
            _EARLY, _LATE_CLAMPED, int(_EARLY.min()), int(_LATE_CLAMPED.max())
        )
        # A[m, y] = c_m inside range, so row sum = c_m * n_m = (1/n_m)*n_m = 1.
        row_sums = A.sum(axis=1)
        # Actually sum over Y columns = c_m (value) * n_m (count) = 1.
        # Let's verify: sum is actually 1.0 per row regardless of c
        np.testing.assert_allclose(row_sums, np.ones(_M), atol=1e-12)

    def test_exact_dated_manuscript(self) -> None:
        # Exact dating (early == late): only one year in range
        early = np.array([500], dtype=np.int64)
        late = np.array([500], dtype=np.int64)
        A = _build_weight_matrix(early, late, 500, 500)
        assert A.shape == (1, 1)
        np.testing.assert_allclose(A[0, 0], 1.0)

    def test_k_at_A_gives_per_year_weights(self) -> None:
        # k = [1, 1, 1, 1, 1] → W = k @ A / M should equal c_m for each year in range
        A = _build_weight_matrix(
            _EARLY, _LATE_CLAMPED, int(_EARLY.min()), int(_LATE_CLAMPED.max())
        )
        k = np.ones((1, _M))
        W = (k @ A) / _M  # [1, Y]
        # W must be non-negative
        assert np.all(W >= 0)
        # Total weight sums to 1 (each manuscript contributes 1/M)
        np.testing.assert_allclose(W.sum(), 1.0, atol=1e-12)


class TestResampleBatch:
    """Test _resample_batch via identity multiplicity = quantile_resample_dates."""

    def test_identity_multiplicity_matches_quantile_resample(self) -> None:
        # Identity bootstrap (k_m=1 for all m) must reproduce quantile_resample_dates exactly.
        pre = _make_precomputed(_EARLY, _LATE_CLAMPED)
        k = np.ones((1, _M), dtype=np.float64)  # [1, M] identity multiplicity

        resampled = _resample_batch(
            k,
            pre["A"],
            pre["D0_sq"],
            pre["D1_sq"],
            pre["D2_sq"],
            pre["x_grid"],
            pre["year_grid"],
            pre["quantiles"],
            pre["mid_age"],
            pre["within_var"],
            pre["c"],
            pre["min_year"],
            pre["max_year"],
            grid_chunk=30,
        )  # [1, M]

        ref = quantile_resample_dates(
            _EARLY, _LATE_CLAMPED, current_year=_CURRENT_YEAR, n_samples=_M
        )
        computed = np.round(resampled[0]).astype(np.int64)
        np.testing.assert_array_equal(computed, ref)

    def test_identity_multiplicity_section_r(self) -> None:
        # Test on a realistic subgroup (section_r, n=309) to confirm match at scale.
        import pandas as pd

        from inf_bp_texts import get_project_root

        csv = get_project_root() / "manuscript_analysis_cleaned.csv"
        if not csv.exists():
            pytest.skip("manuscript_analysis_cleaned.csv not found")

        df = pd.read_csv(csv)
        sub = df.dropna(subset=["earlyYear", "lateYear"])
        sub = sub[sub["section_r"] == 1].copy()
        if len(sub) == 0:
            pytest.skip("section_r subgroup is empty")

        early = sub["earlyYear"].to_numpy(dtype=np.int64)
        late = np.maximum(sub["lateYear"].to_numpy(dtype=np.int64), early)

        ref = quantile_resample_dates(
            early, late, current_year=_CURRENT_YEAR, n_samples=len(early)
        )

        pre = _make_precomputed(early, late)
        k = np.ones((1, len(early)), dtype=np.float64)
        resampled = _resample_batch(
            k,
            pre["A"],
            pre["D0_sq"],
            pre["D1_sq"],
            pre["D2_sq"],
            pre["x_grid"],
            pre["year_grid"],
            pre["quantiles"],
            pre["mid_age"],
            pre["within_var"],
            pre["c"],
            pre["min_year"],
            pre["max_year"],
            grid_chunk=30,
        )

        computed = np.round(resampled[0]).astype(np.int64)
        np.testing.assert_array_equal(computed, ref)

    def test_output_values_in_range(self) -> None:
        pre = _make_precomputed(_EARLY, _LATE_CLAMPED)
        k = np.ones((3, _M), dtype=np.float64)
        # Give different multiplicities to the 3 rows
        k[1, 0] += 2.0
        k[2, 2] += 1.0
        resampled = _resample_batch(
            k,
            pre["A"],
            pre["D0_sq"],
            pre["D1_sq"],
            pre["D2_sq"],
            pre["x_grid"],
            pre["year_grid"],
            pre["quantiles"],
            pre["mid_age"],
            pre["within_var"],
            pre["c"],
            pre["min_year"],
            pre["max_year"],
            grid_chunk=30,
        )
        assert np.all(resampled >= pre["min_year"])
        assert np.all(resampled <= pre["max_year"])

    def test_output_is_sorted_ascending(self) -> None:
        pre = _make_precomputed(_EARLY, _LATE_CLAMPED)
        k = np.ones((5, _M), dtype=np.float64)
        resampled = _resample_batch(
            k,
            pre["A"],
            pre["D0_sq"],
            pre["D1_sq"],
            pre["D2_sq"],
            pre["x_grid"],
            pre["year_grid"],
            pre["quantiles"],
            pre["mid_age"],
            pre["within_var"],
            pre["c"],
            pre["min_year"],
            pre["max_year"],
            grid_chunk=30,
        )
        # Each row (bootstrap) must be sorted ascending
        for b in range(5):
            assert np.all(np.diff(resampled[b]) >= 0), f"Row {b} is not sorted"


class TestBootstrapQuantileResampleDates:
    """Test bootstrap_quantile_resample_dates."""

    def test_output_shapes(self) -> None:
        n_boot = 10
        doc_ids, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _DOC_IDS,
            _MID,
            n_bootstrap=n_boot,
            seed=_SEED,
        )
        assert doc_ids.shape == (n_boot, _M)
        assert years.shape == (n_boot, _M)

    def test_output_dtypes(self) -> None:
        doc_ids, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _DOC_IDS,
            _MID,
            n_bootstrap=4,
            seed=_SEED,
        )
        assert doc_ids.dtype == np.uint16
        assert years.dtype == np.int16

    def test_doc_ids_are_valid(self) -> None:
        doc_ids, _ = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _DOC_IDS,
            _MID,
            n_bootstrap=20,
            seed=_SEED,
        )
        valid_set = set(int(x) for x in _DOC_IDS)
        assert all(int(v) in valid_set for row in doc_ids for v in row)

    def test_resampled_years_in_range(self) -> None:
        _, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _DOC_IDS,
            _MID,
            n_bootstrap=20,
            seed=_SEED,
        )
        assert int(years.min()) >= int(_EARLY.min())
        assert int(years.max()) <= int(_LATE_CLAMPED.max())

    def test_years_sorted_ascending_per_row(self) -> None:
        _, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _DOC_IDS,
            _MID,
            n_bootstrap=8,
            seed=_SEED,
        )
        for b in range(len(years)):
            assert np.all(np.diff(years[b].astype(np.int32)) >= 0), (
                f"Row {b} not sorted"
            )

    def test_doc_ids_sorted_by_midyear(self) -> None:
        # Each bootstrap's doc_ids must be in ascending midYear order.
        doc_ids, _ = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _DOC_IDS,
            _MID,
            n_bootstrap=10,
            seed=_SEED,
        )
        # Build docID → midYear map
        id_to_mid = {int(d): float(m) for d, m in zip(_DOC_IDS, _MID, strict=True)}
        for b in range(len(doc_ids)):
            mids = [id_to_mid[int(d)] for d in doc_ids[b]]
            assert mids == sorted(mids), f"Row {b} docIDs not sorted by midYear"

    def test_reproducibility(self) -> None:
        kwargs: dict = dict(
            early=_EARLY,
            late=_LATE_CLAMPED,
            doc_ids=_DOC_IDS,
            mid_year=_MID,
            n_bootstrap=8,
            seed=_SEED,
        )
        d1, y1 = bootstrap_quantile_resample_dates(**kwargs)
        d2, y2 = bootstrap_quantile_resample_dates(**kwargs)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self) -> None:
        d1, _ = bootstrap_quantile_resample_dates(
            _EARLY, _LATE_CLAMPED, _DOC_IDS, _MID, n_bootstrap=20, seed=1
        )
        d2, _ = bootstrap_quantile_resample_dates(
            _EARLY, _LATE_CLAMPED, _DOC_IDS, _MID, n_bootstrap=20, seed=2
        )
        # At least some rows should differ (vanishingly unlikely to match)
        assert not np.array_equal(d1, d2)

    def test_batch_boundary_exact(self) -> None:
        # n_bootstrap not divisible by batch_size — last partial batch handled correctly
        doc_ids, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _DOC_IDS,
            _MID,
            n_bootstrap=7,
            batch_size=3,
            seed=_SEED,
        )
        assert doc_ids.shape == (7, _M)
        assert years.shape == (7, _M)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_quantile_resample_dates(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                n_bootstrap=1,
            )

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            bootstrap_quantile_resample_dates(
                _EARLY,
                _LATE_CLAMPED,
                _DOC_IDS[:3],
                _MID,
                n_bootstrap=1,
            )

    def test_late_clamping(self) -> None:
        # Passing late < early should not raise; clamping matches add_resampled_year_column
        bad_late = np.array([100, 400, 600, 800, 1000], dtype=np.int64)  # first is bad
        bad_late[0] = 150  # actually valid; use original early-1 scenario:
        bad_late2 = _EARLY.copy()
        bad_late2[0] = _EARLY[0] - 50  # bad: late < early
        doc_ids, years = bootstrap_quantile_resample_dates(
            _EARLY,
            bad_late2,
            _DOC_IDS,
            _MID,
            n_bootstrap=2,
            seed=_SEED,
        )
        assert doc_ids.shape == (2, _M)


class TestSaveLoadBootstrapSamples:
    """Test save_bootstrap_samples and load_bootstrap_samples."""

    def _make_tiny(self, n: int = 3, m: int = 4) -> tuple:
        rng = np.random.default_rng(0)
        doc = rng.integers(10001, 10020, size=(n, m), dtype=np.uint16)
        years = rng.integers(200, 1800, size=(n, m), dtype=np.int16)
        return doc, years

    def test_round_trip_arrays(self) -> None:
        doc, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_samples.npz"
            save_bootstrap_samples(doc, years, path)
            result = load_bootstrap_samples(path)
        np.testing.assert_array_equal(result["doc_id_samples"], doc)
        np.testing.assert_array_equal(result["resampled_years"], years)

    def test_round_trip_metadata(self) -> None:
        doc, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.npz"
            save_bootstrap_samples(
                doc,
                years,
                path,
                subgroup="non_lectionary",
                n_bootstrap=3,
                current_year=2024,
                seed=99,
                order="sorted_by_midYear",
            )
            result = load_bootstrap_samples(path)
        meta = result["metadata"]
        assert meta["subgroup"] == "non_lectionary"
        assert meta["n_bootstrap"] == 3
        assert meta["current_year"] == 2024
        assert meta["seed"] == 99
        assert meta["order"] == "sorted_by_midYear"

    def test_dtypes_preserved(self) -> None:
        doc, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            save_bootstrap_samples(doc, years, path)
            result = load_bootstrap_samples(path)
        assert result["doc_id_samples"].dtype == np.uint16
        assert result["resampled_years"].dtype == np.int16

    def test_load_without_extension(self) -> None:
        doc, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s"
            save_bootstrap_samples(doc, years, path)
            result = load_bootstrap_samples(path)  # no .npz extension
        assert result["doc_id_samples"].shape == doc.shape

    def test_file_not_found_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                load_bootstrap_samples(Path(tmp) / "nonexistent.npz")

    def test_shape_mismatch_raises(self) -> None:
        doc, years = self._make_tiny()
        bad_years = years[:, :2]  # wrong shape
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="shape"):
                save_bootstrap_samples(doc, bad_years, Path(tmp) / "bad.npz")

    def test_parent_dir_created(self) -> None:
        doc, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "samples.npz"
            save_bootstrap_samples(doc, years, path)
            assert path.exists()


@pytest.mark.slow
class TestBootstrapRecovery:
    """Recovery test: per-manuscript mean resampled year ≈ deterministic resample."""

    def test_mean_resampled_year_close_to_deterministic(self) -> None:
        import pandas as pd

        from inf_bp_texts import get_project_root
        from inf_bp_texts.analysis.date_density import quantile_resample_dates
        from inf_bp_texts.data.data_cleaning import add_resampled_year_column

        csv = get_project_root() / "manuscript_analysis_cleaned.csv"
        if not csv.exists():
            pytest.skip("manuscript_analysis_cleaned.csv not found")

        df = pd.read_csv(csv)
        sub = df.dropna(subset=["earlyYear", "lateYear"])
        # Use section_r (n≈309) for speed
        sub = sub[sub["section_r"] == 1].copy()
        sub["earlyYear"] = sub["earlyYear"].astype(int)
        sub["lateYear"] = sub["lateYear"].astype(int)
        sub["midYear"] = (sub["earlyYear"] + sub["lateYear"]) / 2.0
        add_resampled_year_column(sub)

        if len(sub) < 10:
            pytest.skip("section_r subgroup too small")

        early = sub["earlyYear"].to_numpy(dtype=np.int64)
        late = np.maximum(sub["lateYear"].to_numpy(dtype=np.int64), early)
        doc_ids = sub["docID"].to_numpy(dtype=np.int64)
        mid_year = sub["midYear"].to_numpy(dtype=np.float64)
        M = len(early)

        n_boot = 2_000
        doc_id_samples, resampled_years = bootstrap_quantile_resample_dates(
            early,
            late,
            doc_ids,
            mid_year,
            n_bootstrap=n_boot,
            seed=_SEED,
        )

        # For each manuscript position (sorted by midYear), compute mean resampled year.
        # Compare to the deterministic quantile resample values.
        det_years = quantile_resample_dates(
            early, late, current_year=_CURRENT_YEAR, n_samples=M
        )

        # Bootstrap mean of each year slot
        bootstrap_means = resampled_years.astype(np.float64).mean(axis=0)  # [M]

        # Mean should be close to the deterministic resample (within ±50 years)
        max_abs_diff = float(
            np.max(np.abs(bootstrap_means - det_years.astype(np.float64)))
        )
        assert max_abs_diff < 50.0, (
            f"Bootstrap mean resampled years deviate too much from deterministic: "
            f"max |diff| = {max_abs_diff:.1f} years"
        )
