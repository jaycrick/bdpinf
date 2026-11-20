"""Test the knot-aware information criteria."""

import math

import numpy as np
import pytest

from bdpinf.inference.selection import effective_df, information_criteria


class TestEffectiveDf:
    """Test the effective degrees of freedom."""

    @pytest.mark.parametrize("degree", [1, 2, 3])
    @pytest.mark.parametrize("n_knots", [0, 1, 4])
    @pytest.mark.parametrize("cost", [2.0, 3.0, 4.0])
    def test_single_spline_is_d_plus_one_plus_ck(
        self, degree: int, n_knots: int, cost: float
    ) -> None:
        p_rates = degree + 1 + n_knots
        assert effective_df(p_rates, n_knots, cost) == pytest.approx(
            degree + 1 + cost * n_knots
        )

    def test_default_cost_and_no_knots(self) -> None:
        assert effective_df(5, 2) == 9.0
        assert effective_df(5, 0, 4.0) == 5.0


class TestInformationCriteria:
    """Test AIC, AICc and BIC from the likelihood."""

    def test_formulas(self) -> None:
        out = information_criteria(-50.0, 200, 6.0)
        assert out["aic"] == pytest.approx(112.0)
        assert out["bic"] == pytest.approx(100.0 + math.log(200) * 6.0)
        assert out["aicc"] == pytest.approx(112.0 + 2 * 6 * 7 / (200 - 7))

    @pytest.mark.parametrize(("n", "df"), [(10, 9.0), (10, 12.0), (5, 4.0)])
    def test_aicc_infinite_without_room(self, n: int, df: float) -> None:
        assert information_criteria(-1.0, n, df)["aicc"] == np.inf

    def test_recommendation_threshold(self) -> None:
        assert information_criteria(-1.0, 399, 10.0)["aicc_recommended"] is True
        assert information_criteria(-1.0, 400, 10.0)["aicc_recommended"] is False
        assert information_criteria(-1.0, 401, 10.0)["aicc_recommended"] is False
