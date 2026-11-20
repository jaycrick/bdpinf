"""Test range_bounds and its use by AgeData.from_year_ranges."""

import numpy as np
import pytest

from bdpinf.inference import AgeData, range_bounds


class TestRangeBounds:
    def test_rows_aligned_without_merge_or_reorder(self) -> None:
        lower, upper = range_bounds([300, 100, 100, 200], [350, 150, 150, 200])
        np.testing.assert_array_equal(lower, [300, 100, 100, 200])
        np.testing.assert_array_equal(upper, [351, 151, 151, 201])

    def test_clips_to_support(self) -> None:
        lower, upper = range_bounds([-50, 900], [100, 1100], 0.0, 1000.0)
        np.testing.assert_array_equal(lower, [0, 900])
        np.testing.assert_array_equal(upper, [101, 1000])

    def test_one_year_minimum_width_outside_support(self) -> None:
        lower, upper = range_bounds([-80, 1200], [-60, 1300], 0.0, 1000.0)
        np.testing.assert_array_equal(upper - lower, [1.0, 1.0])
        np.testing.assert_array_equal(lower, [0, 999])
        np.testing.assert_array_equal(upper, [1, 1000])

    def test_late_below_early_raised_to_early(self) -> None:
        lower, upper = range_bounds([200], [100])
        assert (lower[0], upper[0]) == (200.0, 201.0)

    def test_no_start_year(self) -> None:
        lower, upper = range_bounds([-50, 900], [100, 1100], None, 1000.0)
        np.testing.assert_array_equal(lower, [-50, 900])
        np.testing.assert_array_equal(upper, [101, 1000])

    def test_no_observation_year(self) -> None:
        lower, upper = range_bounds([-50, 900], [100, 1100], 0.0, None)
        np.testing.assert_array_equal(lower, [0, 900])
        np.testing.assert_array_equal(upper, [101, 1101])

    def test_both_none(self) -> None:
        lower, upper = range_bounds([-50, 900], [100, 1100])
        np.testing.assert_array_equal(lower, [-50, 900])
        np.testing.assert_array_equal(upper, [101, 1101])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            range_bounds([1, 2], [3])

    def test_from_year_ranges_matches_from_ranges(self) -> None:
        early = np.array([-50.0, 100.0, 100.0, 900.0, 1200.0, 300.0])
        late = np.array([100.0, 150.0, 150.0, 1100.0, 1300.0, 250.0])
        start, obs = 0.0, 1000.0
        lower, upper = range_bounds(early, late, start, obs)
        expected = AgeData.from_ranges(obs - upper, obs - lower, obs - start)
        got = AgeData.from_year_ranges(
            early, late, start_year=start, observation_year=obs
        )
        assert got.end_time == expected.end_time
        for name in ("non_spine_ages", "upper_ages", "weights"):
            np.testing.assert_array_equal(
                np.asarray(getattr(got, name)), np.asarray(getattr(expected, name))
            )
