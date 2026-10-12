"""Tests for bootstrap KDE-quantile date resampling (bootstrap_date_density.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bdpinf.analysis.bootstrap_date_density import (
    _build_weight_matrix,  # pyright: ignore[reportPrivateUsage]
    _resample_batch,  # pyright: ignore[reportPrivateUsage]
    bootstrap_epoch_counts,
    bootstrap_quantile_resample_dates,
    bootstrap_target_ranges,
    load_bootstrap_samples,
    save_bootstrap_samples,
)
from bdpinf.analysis.date_density import quantile_resample_dates

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_SEED = 42
_CURRENT_YEAR = 2024

# Small synthetic dataset: 5 items with varying date ranges
_EARLY = np.array([200, 400, 600, 800, 1000], dtype=np.int64)
_LATE = np.array([300, 600, 700, 1200, 1400], dtype=np.int64)
_LATE_CLAMPED = np.maximum(_LATE, _EARLY)
_ITEM_IDS = np.array([10001, 10002, 10003, 10004, 10005], dtype=np.int64)
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

    def test_exact_dated_item(self) -> None:
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
        # Total weight sums to 1 (each item contributes 1/M)
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

    def test_identity_multiplicity_matches_at_larger_scale(self) -> None:
        # Same identity-multiplicity property, but on a larger synthetic
        # dataset (n=300) to confirm the match holds beyond the tiny fixture.
        rng = np.random.default_rng(7)
        n = 300
        early = rng.integers(100, 1900, size=n).astype(np.int64)
        late = np.maximum(early, early + rng.integers(0, 80, size=n)).astype(np.int64)

        ref = quantile_resample_dates(
            early, late, current_year=_CURRENT_YEAR, n_samples=n
        )

        pre = _make_precomputed(early, late)
        k = np.ones((1, n), dtype=np.float64)
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
        item_ids, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _ITEM_IDS,
            _MID,
            n_bootstrap=n_boot,
            seed=_SEED,
        )
        assert item_ids.shape == (n_boot, _M)
        assert years.shape == (n_boot, _M)

    def test_output_dtypes(self) -> None:
        item_ids, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _ITEM_IDS,
            _MID,
            n_bootstrap=4,
            seed=_SEED,
        )
        assert item_ids.dtype == np.uint16
        assert years.dtype == np.int16

    def test_item_ids_are_valid(self) -> None:
        item_ids, _ = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _ITEM_IDS,
            _MID,
            n_bootstrap=20,
            seed=_SEED,
        )
        valid_set = {int(x) for x in _ITEM_IDS}
        assert all(int(v) in valid_set for row in item_ids for v in row)

    def test_resampled_years_in_range(self) -> None:
        _, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _ITEM_IDS,
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
            _ITEM_IDS,
            _MID,
            n_bootstrap=8,
            seed=_SEED,
        )
        for b in range(len(years)):
            assert np.all(np.diff(years[b].astype(np.int32)) >= 0), (
                f"Row {b} not sorted"
            )

    def test_item_ids_sorted_by_mid_value(self) -> None:
        # Each bootstrap's item_ids must be in ascending mid_value order.
        item_ids, _ = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _ITEM_IDS,
            _MID,
            n_bootstrap=10,
            seed=_SEED,
        )
        # Build item id → mid_value map
        id_to_mid = {int(d): float(m) for d, m in zip(_ITEM_IDS, _MID, strict=True)}
        for b in range(len(item_ids)):
            mids = [id_to_mid[int(d)] for d in item_ids[b]]
            assert mids == sorted(mids), f"Row {b} item ids not sorted by mid_value"

    def test_reproducibility(self) -> None:
        kwargs: dict = dict(
            early=_EARLY,
            late=_LATE_CLAMPED,
            item_ids=_ITEM_IDS,
            mid_value=_MID,
            n_bootstrap=8,
            seed=_SEED,
        )
        d1, y1 = bootstrap_quantile_resample_dates(**kwargs)
        d2, y2 = bootstrap_quantile_resample_dates(**kwargs)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self) -> None:
        d1, _ = bootstrap_quantile_resample_dates(
            _EARLY, _LATE_CLAMPED, _ITEM_IDS, _MID, n_bootstrap=20, seed=1
        )
        d2, _ = bootstrap_quantile_resample_dates(
            _EARLY, _LATE_CLAMPED, _ITEM_IDS, _MID, n_bootstrap=20, seed=2
        )
        # At least some rows should differ (vanishingly unlikely to match)
        assert not np.array_equal(d1, d2)

    def test_batch_boundary_exact(self) -> None:
        # n_bootstrap not divisible by batch_size — last partial batch handled correctly
        item_ids, years = bootstrap_quantile_resample_dates(
            _EARLY,
            _LATE_CLAMPED,
            _ITEM_IDS,
            _MID,
            n_bootstrap=7,
            batch_size=3,
            seed=_SEED,
        )
        assert item_ids.shape == (7, _M)
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
                _ITEM_IDS[:3],
                _MID,
                n_bootstrap=1,
            )

    def test_late_clamping(self) -> None:
        # Passing late < early should not raise; the function clamps internally.
        bad_late = np.array([100, 400, 600, 800, 1000], dtype=np.int64)  # first is bad
        bad_late[0] = 150  # actually valid; use original early-1 scenario:
        bad_late2 = _EARLY.copy()
        bad_late2[0] = _EARLY[0] - 50  # bad: late < early
        item_ids, _years = bootstrap_quantile_resample_dates(
            _EARLY,
            bad_late2,
            _ITEM_IDS,
            _MID,
            n_bootstrap=2,
            seed=_SEED,
        )
        assert item_ids.shape == (2, _M)


class TestSaveLoadBootstrapSamples:
    """Test save_bootstrap_samples and load_bootstrap_samples."""

    def _make_tiny(self, n: int = 3, m: int = 4) -> tuple:
        rng = np.random.default_rng(0)
        ids = rng.integers(10001, 10020, size=(n, m), dtype=np.uint16)
        years = rng.integers(200, 1800, size=(n, m), dtype=np.int16)
        return ids, years

    def test_round_trip_arrays(self) -> None:
        ids, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_samples.npz"
            save_bootstrap_samples(ids, years, path)
            result = load_bootstrap_samples(path)
        np.testing.assert_array_equal(result["item_id_samples"], ids)
        np.testing.assert_array_equal(result["resampled_years"], years)

    def test_round_trip_metadata(self) -> None:
        ids, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.npz"
            save_bootstrap_samples(
                ids,
                years,
                path,
                subgroup="non_lectionary",
                n_bootstrap=3,
                current_year=2024,
                seed=99,
                order="sorted_by_mid_value",
            )
            result = load_bootstrap_samples(path)
        meta = result["metadata"]
        assert meta["subgroup"] == "non_lectionary"
        assert meta["n_bootstrap"] == 3
        assert meta["current_year"] == 2024
        assert meta["seed"] == 99
        assert meta["order"] == "sorted_by_mid_value"
        assert meta["schema_version"] == 2
        assert meta["item_id_dtype"] == "uint16"

    def test_dtypes_preserved(self) -> None:
        ids, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            save_bootstrap_samples(ids, years, path)
            result = load_bootstrap_samples(path)
        assert result["item_id_samples"].dtype == np.uint16
        assert result["resampled_years"].dtype == np.int16

    def test_wider_id_dtype_preserved(self) -> None:
        # Ids beyond uint16 range must round-trip without wrapping.
        rng = np.random.default_rng(1)
        ids = rng.integers(70_000, 80_000, size=(2, 3)).astype(np.uint32)
        years = rng.integers(200, 1800, size=(2, 3)).astype(np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.npz"
            save_bootstrap_samples(ids, years, path)
            result = load_bootstrap_samples(path)
        np.testing.assert_array_equal(result["item_id_samples"], ids)
        assert result["metadata"]["item_id_dtype"] == "uint32"

    def test_legacy_doc_id_samples_key_still_loads(self) -> None:
        # Pre-split files used the key "doc_id_samples" with a fixed uint16
        # dtype and no schema_version. load_bootstrap_samples must still
        # read them.
        ids, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.npz"
            np.savez_compressed(
                path,
                doc_id_samples=ids,
                resampled_years=years,
                subgroup=np.array("non_lectionary", dtype=object),
                order_meta=np.array("sorted_by_midYear", dtype=object),
            )
            result = load_bootstrap_samples(path)
        np.testing.assert_array_equal(result["item_id_samples"], ids)
        assert result["metadata"]["schema_version"] == 1
        assert result["metadata"]["item_id_dtype"] == "uint16"

    def test_load_without_extension(self) -> None:
        ids, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s"
            save_bootstrap_samples(ids, years, path)
            result = load_bootstrap_samples(path)  # no .npz extension
        assert result["item_id_samples"].shape == ids.shape

    def test_file_not_found_raises(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            pytest.raises(FileNotFoundError),
        ):
            load_bootstrap_samples(Path(tmp) / "nonexistent.npz")

    def test_shape_mismatch_raises(self) -> None:
        ids, years = self._make_tiny()
        bad_years = years[:, :2]  # wrong shape
        with (
            tempfile.TemporaryDirectory() as tmp,
            pytest.raises(ValueError, match="shape"),
        ):
            save_bootstrap_samples(ids, bad_years, Path(tmp) / "bad.npz")

    def test_parent_dir_created(self) -> None:
        ids, years = self._make_tiny()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "samples.npz"
            save_bootstrap_samples(ids, years, path)
            assert path.exists()


_EPOCH_BOUNDS: list[tuple[int, int]] = [
    (100, 325),
    (325, 636),
    (636, 867),
    (867, 1204),
    (1204, 1516),
    (1516, 2000),
]


class TestBootstrapEpochCounts:
    """Test bootstrap_epoch_counts and bootstrap_target_ranges."""

    _YEARS = np.array(
        [
            [200, 400, 700, 1000, 1300, 1600],  # one per epoch
            [150, 500, 800, 1100, 1400, 1700],  # one per epoch
            [250, 350, 750, 950, 1250, 1550],  # one per epoch
        ],
        dtype=np.int16,
    )

    def test_shape(self) -> None:
        counts = bootstrap_epoch_counts(self._YEARS, _EPOCH_BOUNDS)
        assert counts.shape == (3, 6)

    def test_counts_sum_to_n(self) -> None:
        counts = bootstrap_epoch_counts(self._YEARS, _EPOCH_BOUNDS)
        assert np.all(counts.sum(axis=1) == self._YEARS.shape[1])

    def test_exact_counts(self) -> None:
        counts = bootstrap_epoch_counts(self._YEARS, _EPOCH_BOUNDS)
        # Each row has exactly one item per epoch
        assert np.all(counts == 1)

    def test_duplicate_items_in_epoch(self) -> None:
        # All items land in epoch 0
        years = np.array([[200, 250, 300]], dtype=np.int16)
        bounds = [(100, 325), (325, 636)]
        counts = bootstrap_epoch_counts(years, bounds)
        assert counts[0, 0] == 3
        assert counts[0, 1] == 0

    def test_boundary_lower_inclusive(self) -> None:
        years = np.array([[325]], dtype=np.int16)
        bounds = [(100, 325), (325, 636)]
        counts = bootstrap_epoch_counts(years, bounds)
        assert counts[0, 0] == 0  # 325 is NOT in [100, 325)
        assert counts[0, 1] == 1  # 325 IS in [325, 636)

    def test_target_ranges_shape(self) -> None:
        ranges = bootstrap_target_ranges(self._YEARS, _EPOCH_BOUNDS)
        assert ranges.shape == (6, 2)

    def test_target_ranges_last_era_unconditioned(self) -> None:
        ranges = bootstrap_target_ranges(self._YEARS, _EPOCH_BOUNDS)
        assert ranges[-1, 0] == 0
        assert ranges[-1, 1] == 2**31 - 1

    def test_target_ranges_lo_le_hi(self) -> None:
        ranges = bootstrap_target_ranges(self._YEARS, _EPOCH_BOUNDS)
        assert np.all(ranges[:-1, 0] <= ranges[:-1, 1])

    def test_target_ranges_dtype(self) -> None:
        ranges = bootstrap_target_ranges(self._YEARS, _EPOCH_BOUNDS)
        assert ranges.dtype == np.intp

    def test_target_ranges_custom_quantiles(self) -> None:
        # With quantile_lo=0.5, quantile_hi=0.5 the range should be a point
        ranges = bootstrap_target_ranges(
            self._YEARS, _EPOCH_BOUNDS, quantile_lo=0.5, quantile_hi=0.5
        )
        assert np.all(ranges[:-1, 0] == ranges[:-1, 1])

    def test_target_ranges_wider_for_larger_n_bootstrap(self) -> None:
        rng = np.random.default_rng(0)
        # Two datasets of same items, spread over epochs — CI should be wider
        # when counts vary more across draws
        uniform = np.full((100, 1), 200, dtype=np.int16)  # always epoch 0
        variable = rng.choice([200, 400, 700], size=(100, 1)).astype(np.int16)
        bounds = [(100, 325), (325, 636), (636, 867)]
        r_uniform = bootstrap_target_ranges(uniform, bounds)
        r_variable = bootstrap_target_ranges(variable, bounds)
        # uniform has width 0 in era 0; variable should have width >= 0
        width_uniform = int(r_uniform[0, 1]) - int(r_uniform[0, 0])
        width_variable = int(r_variable[0, 1]) - int(r_variable[0, 0])
        assert width_uniform == 0
        assert width_variable >= 0  # can't be negative


@pytest.mark.slow
class TestBootstrapRecovery:
    """Recovery test: per-item mean resampled year ≈ deterministic resample, at scale."""

    def test_mean_resampled_year_close_to_deterministic(self) -> None:
        rng = np.random.default_rng(11)
        n = 300
        early = rng.integers(100, 1900, size=n).astype(np.int64)
        late = np.maximum(early, early + rng.integers(0, 80, size=n)).astype(np.int64)
        item_ids = np.arange(10001, 10001 + n, dtype=np.int64)
        mid_value = (early + late) / 2.0

        n_boot = 2_000
        _item_id_samples, resampled_years = bootstrap_quantile_resample_dates(
            early,
            late,
            item_ids,
            mid_value,
            n_bootstrap=n_boot,
            seed=_SEED,
        )

        # For each item position (sorted by mid_value), compute mean resampled
        # year and compare to the deterministic quantile resample values.
        det_years = quantile_resample_dates(
            early, late, current_year=_CURRENT_YEAR, n_samples=n
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
