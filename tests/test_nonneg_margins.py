"""Test nonneg_margins of the spline rates against a dense grid."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bdpinf.models.rate_functions import PiecewiseCubicRate, PiecewiseQuadraticRate

_TIMES = jnp.array([3.0, 7.0, 10.0])
_GRID = np.linspace(0.0, 10.0, 20001)


def _cubic(
    rng: np.random.Generator, zero_at: tuple[int, ...] = ()
) -> PiecewiseCubicRate:
    values = rng.uniform(0.0, 1.0, 4)
    values[list(zero_at)] = 0.0
    slopes = rng.normal(0.0, 0.5, 2)
    return PiecewiseCubicRate.clamped_spline(
        jnp.asarray(values), float(slopes[0]), float(slopes[1]), _TIMES
    )


def _quadratic(
    rng: np.random.Generator, zero_at: tuple[int, ...] = ()
) -> PiecewiseQuadraticRate:
    values = rng.uniform(0.0, 1.0, 4)
    values[list(zero_at)] = 0.0
    return PiecewiseQuadraticRate.spline(
        jnp.asarray(values), float(rng.normal(0.0, 0.5)), _TIMES
    )


def _dense_min(rate: PiecewiseCubicRate | PiecewiseQuadraticRate) -> float:
    return float(np.min(np.asarray(rate(jnp.asarray(_GRID)))))


class TestNonnegMargins:
    """Test that the margins are non-negative exactly when the rate is."""

    @pytest.mark.parametrize("make", [_cubic, _quadratic])
    def test_sign_matches_dense_grid(self, make) -> None:
        rng = np.random.default_rng(0)
        n_feasible = n_infeasible = 0
        for _ in range(150):
            rate = make(rng)
            margins = np.asarray(rate.nonneg_margins())
            truly = _dense_min(rate)
            if truly > 1e-6:
                assert margins.min() >= 0.0
                n_feasible += 1
            elif truly < -1e-6:
                assert margins.min() < 0.0
                n_infeasible += 1
        assert n_feasible > 10
        assert n_infeasible > 10

    @pytest.mark.parametrize("make", [_cubic, _quadratic])
    def test_minimum_agrees_with_min_values(self, make) -> None:
        rng = np.random.default_rng(1)
        for _ in range(50):
            rate = make(rng)
            margins = np.asarray(rate.nonneg_margins())
            assert np.isclose(margins.min(), float(rate.min_values().min()))

    @pytest.mark.parametrize("make", [_cubic, _quadratic])
    @pytest.mark.parametrize("zero_at", [(3,), (0,), (1,), (0, 3)])
    def test_pinned_zero_ends(self, make, zero_at: tuple[int, ...]) -> None:
        rng = np.random.default_rng(2)
        agree = 0
        for _ in range(150):
            rate = make(rng, zero_at)
            margins = np.asarray(rate.nonneg_margins(pinned_zero=zero_at))
            # The value rows of the pinned knots are replaced by slope rows.
            labels = rate.nonneg_margin_labels(pinned_zero=zero_at)
            assert len(labels) == len(margins)
            for j in zero_at:
                if j == 3:
                    assert any(lab == "slope at end of epoch 2" for lab in labels)
                else:
                    assert f"slope at start of epoch {j}" in labels
            truly = _dense_min(rate)
            if truly > -1e-7:
                assert margins.min() >= -1e-7
                agree += 1
            elif truly < -1e-4:
                assert margins.min() < 0.0
                agree += 1
        assert agree > 40

    def test_interior_pinned_knot_has_both_one_sided_rows(self) -> None:
        rate = _cubic(np.random.default_rng(3), zero_at=(1,))
        labels = rate.nonneg_margin_labels(pinned_zero=(1,))
        assert "slope at end of epoch 0" in labels
        assert "slope at start of epoch 1" in labels
        assert "value at end of epoch 0" not in labels

    def test_include_values_false_drops_value_rows(self) -> None:
        rate = _cubic(np.random.default_rng(4))
        full = rate.nonneg_margins()
        slim = rate.nonneg_margins(include_values=False)
        assert len(slim) == len(full) - 4
        assert len(rate.nonneg_margin_labels(include_values=False)) == len(slim)

    def test_discontinuous_has_every_start_value(self) -> None:
        rate = _cubic(np.random.default_rng(5))
        assert (
            len(rate.nonneg_margins(continuous=False)) == len(rate.nonneg_margins()) + 2
        )

    def test_bad_pinned_index_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            _cubic(np.random.default_rng(6)).nonneg_margins(pinned_zero=(9,))

    def test_jittable(self) -> None:
        rate = _cubic(np.random.default_rng(7))
        jitted = jax.jit(lambda r: r.nonneg_margins(pinned_zero=(3,)))
        assert np.allclose(jitted(rate), rate.nonneg_margins(pinned_zero=(3,)))
