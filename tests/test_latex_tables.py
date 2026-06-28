"""Tests for the booktabs/siunitx LaTeX table renderer."""

import math

import numpy as np

from inf_bp_texts.analysis.latex_tables import (
    BAYESIAN_COLUMNS,
    MLE_COLUMNS,
    LatexParamGroup,
    LatexRateRow,
    bayesian_results_latex_by_epoch,
    bayesian_results_latex_by_subset,
    rates_latex_table,
    subset_fits_latex_by_epoch,
    subset_fits_latex_by_subset,
)
from inf_bp_texts.analysis.subset_inference import SubsetFitResult

# ---------------------------------------------------------------------------
# Minimal SubsetFitResult fixtures
# ---------------------------------------------------------------------------

_EPOCHS = [
    ("Ante Nicaea", 100, 325),
    ("Post Nicaea", 325, 636),
]


def _make_fit(
    key: str,
    label: str,
    success: bool = True,
    include_se_ci: bool = True,
) -> SubsetFitResult:
    """Return a minimal SubsetFitResult with two epochs."""
    if not success:
        return SubsetFitResult(
            key=key,
            label=label,
            success=False,
            n_manuscripts=0,
            times=np.zeros(2),
            ages=np.empty(0),
            nll_history=[],
            log_likelihood=float("-inf"),
        )
    epoch_results = []
    for i, (name, start, end) in enumerate(_EPOCHS):
        entry = {
            "period": name,
            "start": start,
            "end": end,
            "n_manuscripts": 10 + i,
            "birth_rates": 0.001 * (i + 1),
            "death_rates": 0.000001 * (i + 1),
            "half_life": math.log(2) / (0.000001 * (i + 1)),
            "success": True,
        }
        if include_se_ci:
            entry["birth_rate_se"] = 0.0001 * (i + 1)
            entry["death_rate_se"] = 0.0000001 * (i + 1)
            entry["birth_rate_ci"] = (0.0009 * (i + 1), 0.0011 * (i + 1))
            entry["death_rate_ci"] = (0.0000009 * (i + 1), 0.0000011 * (i + 1))
        epoch_results.append(entry)
    return SubsetFitResult(
        key=key,
        label=label,
        success=True,
        n_manuscripts=20,
        times=np.array([325.0, 636.0]),
        ages=np.arange(20, dtype=float),
        nll_history=[1.0, 0.5],
        log_likelihood=-100.0,
        epoch_results=epoch_results,
    )


def _make_bayesian_entry(key: str, label: str, n_draws: int = 50) -> dict:
    """Return a minimal bayesian_results entry with two epochs."""
    rng = np.random.default_rng(42)
    ep_names = [e[0] for e in _EPOCHS]
    n_epochs = len(ep_names)
    bs = rng.lognormal(mean=-7.0, sigma=0.3, size=(n_draws, n_epochs))
    ds = rng.lognormal(mean=-14.0, sigma=0.3, size=(n_draws, n_epochs))
    return {
        "key": key,
        "label": label,
        "success": True,
        "n_manuscripts": 20,
        "epoch_names": ep_names,
        "n_epochs": n_epochs,
        "birth_rates": bs.mean(axis=0),
        "death_rates": ds.mean(axis=0),
        "posterior_samples": {"birth_rates": bs, "death_rates": ds},
        "diagnostics": {
            "birth_rates": {"r_hat": np.ones(n_epochs)},
            "death_rates": {"r_hat": np.ones(n_epochs)},
        },
        "num_divergences": 0,
        "mle_birth_rates": None,
        "mle_death_rates": None,
        "mle_success": False,
        "elapsed": 1.0,
    }


# ---------------------------------------------------------------------------
# Core renderer tests
# ---------------------------------------------------------------------------


class TestRatesLatexTable:
    """Test rates_latex_table."""

    def _two_group_table(self, **kwargs: object) -> str:
        groups = [
            LatexParamGroup(
                symbol=r"\lambda",
                rows=[
                    LatexRateRow(
                        "Gk",
                        {
                            "estimate": 0.001230,
                            "se": 0.000016,
                            "ci": (0.001199, 0.001262),
                        },
                    ),
                    LatexRateRow(
                        "Ar K",
                        {
                            "estimate": 0.002018,
                            "se": 0.000201,
                            "ci": (0.001661, 0.002453),
                        },
                    ),
                ],
            ),
            LatexParamGroup(
                symbol=r"\mu",
                rows=[
                    LatexRateRow(
                        "Gk",
                        {
                            "estimate": 0.000003,
                            "se": 0.000002,
                            "ci": (0.000001, 0.000009),
                        },
                    ),
                    LatexRateRow(
                        "Ar K",
                        {
                            "estimate": 0.000064,
                            "se": 0.000063,
                            "ci": (0.000009, 0.000443),
                        },
                    ),
                ],
            ),
        ]
        return rates_latex_table(
            groups,
            MLE_COLUMNS,
            caption="Test caption",
            label="test",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_contains_begin_table(self) -> None:
        assert r"\begin{table}" in self._two_group_table()

    def test_contains_booktabs_rules(self) -> None:
        t = self._two_group_table()
        assert r"\toprule" in t
        assert r"\midrule" in t
        assert r"\bottomrule" in t

    def test_contains_multirow(self) -> None:
        t = self._two_group_table()
        assert r"\multirow{2}{*}{$\lambda$}" in t
        assert r"\multirow{2}{*}{$\mu$}" in t

    def test_contains_s_column_format(self) -> None:
        t = self._two_group_table()
        assert "S[table-format=1.6]" in t

    def test_contains_num_brackets_in_ci(self) -> None:
        t = self._two_group_table()
        assert r"\num{" in t
        assert r"[\num{" in t

    def test_numeric_precision_six_decimal(self) -> None:
        t = self._two_group_table()
        # 0.001230 at precision=6 → "0.001230"
        assert "0.001230" in t

    def test_custom_num_format(self) -> None:
        t = self._two_group_table(num_format="2.7")
        assert "S[table-format=2.7]" in t

    def test_custom_row_header(self) -> None:
        t = self._two_group_table(row_header="Epoch")
        assert "Epoch" in t

    def test_nan_value_renders_as_dash(self) -> None:
        groups = [
            LatexParamGroup(
                symbol=r"\lambda",
                rows=[
                    LatexRateRow(
                        "X", {"estimate": float("nan"), "se": None, "ci": None}
                    ),
                ],
            )
        ]
        t = rates_latex_table(groups, MLE_COLUMNS, caption="c", label="l")
        assert r"{--}" in t

    def test_none_ci_renders_as_dash(self) -> None:
        groups = [
            LatexParamGroup(
                symbol=r"\lambda",
                rows=[LatexRateRow("X", {"estimate": 0.001, "se": 0.0001, "ci": None})],
            )
        ]
        t = rates_latex_table(groups, MLE_COLUMNS, caption="c", label="l")
        assert r"{--}" in t

    def test_label_in_output(self) -> None:
        t = self._two_group_table()
        assert r"\label{tab:test}" in t

    def test_caption_in_output(self) -> None:
        t = self._two_group_table()
        assert "Test caption" in t

    def test_midrule_count_equals_groups_minus_one(self) -> None:
        t = self._two_group_table()
        # 2 groups → 1 \midrule between groups + 1 after \toprule header = 2
        assert t.count(r"\midrule") == 2  # header + inter-group

    def test_bayesian_columns(self) -> None:
        groups = [
            LatexParamGroup(
                symbol=r"\lambda",
                rows=[
                    LatexRateRow(
                        "Gk",
                        {
                            "mean": 0.001230,
                            "median": 0.001215,
                            "cri": (0.001100, 0.001350),
                        },
                    ),
                ],
            )
        ]
        t = rates_latex_table(groups, BAYESIAN_COLUMNS, caption="c", label="l")
        assert "Mean" in t
        assert "Median" in t
        assert r"95\% CrI" in t

    def test_end_table_present(self) -> None:
        t = self._two_group_table()
        assert r"\end{table}" in t
        assert r"\end{tabular}" in t


# ---------------------------------------------------------------------------
# MLE assembly helper tests
# ---------------------------------------------------------------------------


class TestSubsetFitsLatexHelpers:
    """Test subset_fits_latex_by_epoch and subset_fits_latex_by_subset."""

    def _fits(self) -> list[SubsetFitResult]:
        return [
            _make_fit("nl", "Non-Lectionary"),
            _make_fit("lect", "Lectionary"),
        ]

    def _epoch_names(self) -> list[str]:
        return [e[0] for e in _EPOCHS]

    def test_by_epoch_returns_dict_keyed_by_epoch(self) -> None:
        tables = subset_fits_latex_by_epoch(self._fits(), self._epoch_names())
        assert set(tables.keys()) == set(self._epoch_names())

    def test_by_epoch_each_value_is_string(self) -> None:
        tables = subset_fits_latex_by_epoch(self._fits(), self._epoch_names())
        for v in tables.values():
            assert isinstance(v, str)

    def test_by_epoch_contains_tradition_label(self) -> None:
        tables = subset_fits_latex_by_epoch(self._fits(), self._epoch_names())
        for t in tables.values():
            assert "Non-Lectionary" in t

    def test_by_epoch_skips_failed_fits(self) -> None:
        fits = [_make_fit("fail", "Failed", success=False)]
        tables = subset_fits_latex_by_epoch(fits, self._epoch_names())
        assert len(tables) == 0

    def test_by_subset_returns_dict_keyed_by_label(self) -> None:
        tables = subset_fits_latex_by_subset(self._fits())
        assert set(tables.keys()) == {"Non-Lectionary", "Lectionary"}

    def test_by_subset_each_value_is_string(self) -> None:
        tables = subset_fits_latex_by_subset(self._fits())
        for v in tables.values():
            assert isinstance(v, str)

    def test_by_subset_contains_epoch_label(self) -> None:
        tables = subset_fits_latex_by_subset(self._fits())
        for t in tables.values():
            assert "Ante Nicaea" in t

    def test_by_subset_skips_failed_fits(self) -> None:
        tables = subset_fits_latex_by_subset(
            [_make_fit("fail", "Failed", success=False)]
        )
        assert len(tables) == 0

    def test_by_epoch_no_se_renders_dash(self) -> None:
        """Fit without SE/CI should still render with {--} for those cells."""
        fits = [_make_fit("nl", "Non-Lectionary", include_se_ci=False)]
        tables = subset_fits_latex_by_epoch(fits, self._epoch_names())
        for t in tables.values():
            assert r"{--}" in t

    def test_row_header_epoch_in_by_subset(self) -> None:
        tables = subset_fits_latex_by_subset(self._fits())
        for t in tables.values():
            assert "Epoch" in t

    def test_row_header_tradition_in_by_epoch(self) -> None:
        tables = subset_fits_latex_by_epoch(self._fits(), self._epoch_names())
        for t in tables.values():
            assert "Tradition" in t


# ---------------------------------------------------------------------------
# Bayesian assembly helper tests
# ---------------------------------------------------------------------------


class TestBayesianLatexHelpers:
    """Test bayesian_results_latex_by_epoch and bayesian_results_latex_by_subset."""

    def _results(self) -> dict[str, dict]:
        return {
            "nl": _make_bayesian_entry("nl", "Non-Lectionary"),
            "lect": _make_bayesian_entry("lect", "Lectionary"),
        }

    def _epoch_names(self) -> list[str]:
        return [e[0] for e in _EPOCHS]

    def test_by_epoch_returns_dict_keyed_by_epoch(self) -> None:
        tables = bayesian_results_latex_by_epoch(self._results(), self._epoch_names())
        assert set(tables.keys()) == set(self._epoch_names())

    def test_by_epoch_contains_mean_and_median_headers(self) -> None:
        tables = bayesian_results_latex_by_epoch(self._results(), self._epoch_names())
        for t in tables.values():
            assert "Mean" in t
            assert "Median" in t

    def test_by_epoch_contains_cri_column(self) -> None:
        tables = bayesian_results_latex_by_epoch(self._results(), self._epoch_names())
        for t in tables.values():
            assert r"95\% CrI" in t

    def test_by_epoch_contains_num_brackets(self) -> None:
        tables = bayesian_results_latex_by_epoch(self._results(), self._epoch_names())
        for t in tables.values():
            assert r"\num{" in t

    def test_by_subset_returns_dict_keyed_by_label(self) -> None:
        tables = bayesian_results_latex_by_subset(self._results())
        assert set(tables.keys()) == {"Non-Lectionary", "Lectionary"}

    def test_by_subset_contains_epoch_names(self) -> None:
        tables = bayesian_results_latex_by_subset(self._results())
        for t in tables.values():
            assert "Ante Nicaea" in t

    def test_by_epoch_skips_subset_missing_epoch(self) -> None:
        """Subset whose epoch_names excludes 'Ante Nicaea' should be absent from that tab."""
        results = {
            "lect": {
                **_make_bayesian_entry("lect", "Lectionary"),
                # Lectionary only has Post Nicaea
                "epoch_names": ["Post Nicaea"],
                "birth_rates": np.array([0.001]),
                "death_rates": np.array([0.000001]),
                "posterior_samples": {
                    "birth_rates": np.random.default_rng(0).lognormal(-7, 0.3, (50, 1)),
                    "death_rates": np.random.default_rng(1).lognormal(
                        -14, 0.3, (50, 1)
                    ),
                },
            }
        }
        tables = bayesian_results_latex_by_epoch(
            results, ["Ante Nicaea", "Post Nicaea"]
        )
        # "Ante Nicaea" table should be absent (no subsets have that epoch)
        assert "Ante Nicaea" not in tables
        assert "Post Nicaea" in tables

    def test_cri_prob_reflected_in_caption(self) -> None:
        tables = bayesian_results_latex_by_epoch(
            self._results(), self._epoch_names(), cri_prob=0.89
        )
        for t in tables.values():
            assert "89" in t
