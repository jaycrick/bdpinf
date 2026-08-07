"""Tests for item date density estimation and quantile resampling."""

import numpy as np
import pytest

from bdpinf.analysis.date_density import (
    date_range_density,
    explode_date_ranges,
    quantile_resample_dates,
)


class TestDateDensity:
    """Test date density estimation for items with date ranges."""

    # -----------------------------------------------------------------
    # explode_date_ranges
    # -----------------------------------------------------------------

    def test_explode_exact_dates_give_weight_one(self) -> None:
        early = np.array([500, 600], dtype=np.int64)
        late = np.array([500, 600], dtype=np.int64)
        years, weights = explode_date_ranges(early, late)
        assert years.tolist() == [500.0, 600.0]
        np.testing.assert_allclose(weights, [1.0, 1.0])

    def test_explode_range_correct_count_and_weights(self) -> None:
        early = np.array([100], dtype=np.int64)
        late = np.array([104], dtype=np.int64)
        years, weights = explode_date_ranges(early, late)
        assert len(years) == 5  # 100, 101, 102, 103, 104
        np.testing.assert_allclose(weights, np.full(5, 0.2))

    def test_explode_total_weight_equals_item_count(self) -> None:
        early = np.array([100, 200, 300], dtype=np.int64)
        late = np.array([110, 250, 300], dtype=np.int64)
        _, weights = explode_date_ranges(early, late)
        np.testing.assert_allclose(weights.sum(), 3.0, rtol=1e-12)

    def test_explode_per_range_weights_sum_to_one(self) -> None:
        # Each item's pseudo-points should sum to 1.
        early = np.array([100, 200], dtype=np.int64)
        late = np.array([109, 204], dtype=np.int64)
        years, weights = explode_date_ranges(early, late)
        # First item: years 100-109 (10 points)
        w_first = weights[years < 200]
        np.testing.assert_allclose(w_first.sum(), 1.0, rtol=1e-12)
        # Second item: years 200-204 (5 points)
        w_second = weights[years >= 200]
        np.testing.assert_allclose(w_second.sum(), 1.0, rtol=1e-12)

    def test_explode_years_span_correct_range(self) -> None:
        early = np.array([300, 500], dtype=np.int64)
        late = np.array([302, 505], dtype=np.int64)
        years, _ = explode_date_ranges(early, late)
        assert int(years.min()) == 300
        assert int(years.max()) == 505

    def test_explode_rejects_late_before_early(self) -> None:
        early = np.array([500], dtype=np.int64)
        late = np.array([499], dtype=np.int64)
        with pytest.raises(ValueError, match="late"):
            explode_date_ranges(early, late)

    def test_explode_rejects_mismatched_lengths(self) -> None:
        early = np.array([100, 200], dtype=np.int64)
        late = np.array([110], dtype=np.int64)
        with pytest.raises(ValueError):
            explode_date_ranges(early, late)

    # -----------------------------------------------------------------
    # date_range_density
    # -----------------------------------------------------------------

    def test_density_integrates_to_one(self) -> None:
        early = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        late = np.array([150, 250, 350, 450, 550], dtype=np.int64)
        year_grid, density = date_range_density(early, late, current_year=2024)
        integral = float(np.trapezoid(density, year_grid))
        assert abs(integral - 1.0) < 0.02

    def test_density_nonnegative(self) -> None:
        early = np.array([100, 200, 300], dtype=np.int64)
        late = np.array([150, 250, 350], dtype=np.int64)
        _, density = date_range_density(early, late, current_year=2024)
        assert np.all(density >= 0.0)

    def test_density_year_grid_spans_data(self) -> None:
        early = np.array([400, 800], dtype=np.int64)
        late = np.array([500, 900], dtype=np.int64)
        year_grid, _ = date_range_density(early, late, current_year=2024)
        assert year_grid[0] <= 400
        assert year_grid[-1] >= 900

    def test_density_year_grid_monotonically_increasing(self) -> None:
        early = np.array([100, 500, 900], dtype=np.int64)
        late = np.array([200, 600, 1000], dtype=np.int64)
        year_grid, _ = date_range_density(early, late, current_year=2024)
        assert np.all(np.diff(year_grid) > 0)

    # -----------------------------------------------------------------
    # quantile_resample_dates
    # -----------------------------------------------------------------

    def test_resample_returns_correct_count(self) -> None:
        early = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        late = np.array([200, 300, 400, 500, 600], dtype=np.int64)
        result = quantile_resample_dates(early, late, current_year=2024, n_samples=5)
        assert len(result) == 5

    def test_resample_dates_within_data_range(self) -> None:
        early = np.array([300, 500, 700], dtype=np.int64)
        late = np.array([400, 600, 800], dtype=np.int64)
        result = quantile_resample_dates(early, late, current_year=2024, n_samples=10)
        assert int(result.min()) >= 300
        assert int(result.max()) <= 800

    def test_resample_result_sorted(self) -> None:
        early = np.array([100, 300, 500, 700], dtype=np.int64)
        late = np.array([200, 400, 600, 800], dtype=np.int64)
        result = quantile_resample_dates(early, late, current_year=2024, n_samples=8)
        assert np.all(np.diff(result) >= 0), (
            "quantile_resample_dates should return sorted dates"
        )

    def test_resample_returns_integer_years(self) -> None:
        early = np.array([100, 200, 300], dtype=np.int64)
        late = np.array([150, 250, 350], dtype=np.int64)
        result = quantile_resample_dates(early, late, current_year=2024, n_samples=6)
        assert result.dtype in (np.int32, np.int64, np.intp)

    def test_resample_different_n_samples(self) -> None:
        early = np.array([100, 200, 300, 400], dtype=np.int64)
        late = np.array([150, 250, 350, 450], dtype=np.int64)
        for n in [1, 4, 20]:
            result = quantile_resample_dates(
                early, late, current_year=2024, n_samples=n
            )
            assert len(result) == n


class TestQuantileResampleDatesDefaults:
    """Test quantile_resample_dates default-argument and list-input behaviour."""

    def test_default_n_samples_equals_len_early(self) -> None:
        early = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        late = np.array([200, 300, 400, 500, 600], dtype=np.int64)
        result = quantile_resample_dates(early, late)
        assert len(result) == len(early)

    def test_all_values_within_data_range(self) -> None:
        early = np.array([300, 500, 700], dtype=np.int64)
        late = np.array([400, 600, 800], dtype=np.int64)
        result = quantile_resample_dates(early, late)
        assert int(result.min()) >= 300
        assert int(result.max()) <= 800

    def test_accepts_plain_python_lists(self) -> None:
        early = [100, 200, 300]
        late = [150, 250, 350]
        result = quantile_resample_dates(early, late)
        assert len(result) == 3

    def test_default_n_samples_matches_explicit_len(self) -> None:
        early = np.array([100, 200, 300, 400], dtype=np.int64)
        late = np.array([150, 250, 350, 450], dtype=np.int64)
        result_default = quantile_resample_dates(early, late)
        result_explicit = quantile_resample_dates(
            early, late, current_year=2024, n_samples=len(early)
        )
        np.testing.assert_array_equal(result_default, result_explicit)
