r"""Publication-quality LaTeX table generation for birth-death rate estimates.

Renders booktabs + siunitx tables matching the project's house style: multirow
parameter groups, ``S[table-format=…]`` numeric columns, ``\\num{}``-bracketed
CI/CrI column.  Requires ``booktabs``, ``siunitx``, and ``multirow`` in the
consuming document's LaTeX preamble.

Two canonical renderings are supported:

* **Per-epoch** tables (rows = traditions, one table per epoch) — compare how a
  given historical period's rates differ across manuscript subsets.
* **Per-tradition** tables (rows = epochs, one table per tradition) — show how
  rates evolve over time within a single manuscript group.

Assembly helpers extract values from the existing result structures
(:class:`~inf_bp_texts.analysis.subset_inference.SubsetFitResult` for MLE,
``bayesian_results`` dicts from the parallel driver for Bayesian) and return a
``{name: latex_string}`` dict ready to be printed as fenced code blocks in
Quarto notebooks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from inf_bp_texts.analysis.subset_inference import SubsetFitResult


# ---------------------------------------------------------------------------
# Generic renderer data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatexColumn:
    """One numeric column in the rendered table."""

    header: str
    """Column header, e.g. ``"Est."``, ``"SE"``, ``"Mean"``, ``"Median"``."""
    kind: Literal["num", "ci"]
    """``"num"`` → S[table-format=…] column; ``"ci"`` → l column with ``\\num{}``."""
    key: str
    """Key into a :class:`LatexRateRow`'s ``values`` mapping."""


@dataclass(frozen=True)
class LatexRateRow:
    """One data row in a parameter group."""

    label: str
    """Row label, e.g. ``"Non-Lect."`` or ``"Ante Nicaea"``."""
    values: Mapping[str, float | tuple[float, float] | None]
    """Values keyed by :attr:`LatexColumn.key`.

    *  ``float`` → used for ``"num"`` columns (Est., SE, Mean, Median).
    *  ``tuple[float, float]`` → ``(lo, hi)`` used for ``"ci"`` columns.
    *  ``None`` or ``float("nan")`` → rendered as ``{--}``.
    """


@dataclass(frozen=True)
class LatexParamGroup:
    r"""A ``\multirow`` parameter block (e.g. λ or μ)."""

    symbol: str
    r"""LaTeX symbol string, e.g. ``r"\lambda"``.  Wrapped in ``$…$`` automatically."""
    rows: Sequence[LatexRateRow]
    """Ordered rows belonging to this parameter group."""


# ---------------------------------------------------------------------------
# Pre-built column definitions
# ---------------------------------------------------------------------------

#: MLE column set: point estimate, standard error, 95 % confidence interval.
MLE_COLUMNS: tuple[LatexColumn, ...] = (
    LatexColumn(header="Est.", kind="num", key="estimate"),
    LatexColumn(header="SE", kind="num", key="se"),
    LatexColumn(header=r"95\% CI", kind="ci", key="ci"),
)

#: Bayesian column set: posterior mean, median, 95 % credible interval.
BAYESIAN_COLUMNS: tuple[LatexColumn, ...] = (
    LatexColumn(header="Mean", kind="num", key="mean"),
    LatexColumn(header="Median", kind="num", key="median"),
    LatexColumn(header=r"95\% CrI", kind="ci", key="cri"),
)


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------


def _fmt_num(value: float | None, precision: int) -> str:
    """Format a scalar for an ``S`` column cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return r"{--}"
    return f"{value:.{precision}f}"


def _fmt_ci(pair: tuple[float, float] | None, precision: int) -> str:
    r"""Format ``(lo, hi)`` as ``{[\num{lo}, \num{hi}]}`` or ``{--}``."""
    if pair is None:
        return r"{--}"
    lo, hi = pair
    if math.isnan(lo) or math.isnan(hi):
        return r"{--}"
    return (
        r"{[\num{"
        + f"{lo:.{precision}f}"
        + r"}, \num{"
        + f"{hi:.{precision}f}"
        + r"}]}"
    )


def rates_latex_table(
    groups: Sequence[LatexParamGroup],
    columns: Sequence[LatexColumn],
    *,
    caption: str,
    label: str,
    row_header: str = "Dataset",
    num_format: str = "1.6",
    num_precision: int = 6,
) -> str:
    r"""Render a booktabs + siunitx LaTeX table for rate estimates.

    Produces a ``\begin{table}…\end{table}`` block with:

    * ``\multirow`` parameter groups separated by ``\midrule``,
    * ``S[table-format=…]`` columns for scalar estimates,
    * A plain ``l`` column with ``\num{…}``-bracketed intervals for CI/CrI.

    Parameters
    ----------
    groups : Sequence[LatexParamGroup]
        Parameter groups (λ then μ, typically), each rendered as a
        ``\multirow`` block.
    columns : Sequence[LatexColumn]
        Ordered list of data columns after the row-label column.
    caption : str
        Table caption.
    label : str
        LaTeX ``\label`` value (written as ``\label{tab:<label>}``).
    row_header : str
        Header for the row-label column (``"Dataset"`` or ``"Epoch"``).
    num_format : str
        ``table-format`` value for ``S`` columns (default ``"1.6"``).
    num_precision : int
        Decimal places for numeric cells and ``\num{}`` arguments.

    Returns
    -------
    str
        A complete ``\begin{table}…\end{table}`` string.
    """
    # Build tabular column specification
    s_col = f"S[table-format={num_format}]"
    col_parts: list[str] = ["@{}", "l", "l"]
    for col in columns:
        col_parts.append(s_col if col.kind == "num" else "l")
    col_spec = "\n      ".join(col_parts) + "@{}"

    # Header row — all data-column headers wrapped in {} so siunitx S columns
    # do not attempt to parse them as numbers.
    header_cells = ["Param.", row_header]
    for col in columns:
        header_cells.append("{" + col.header + "}")
    header_row = " & ".join(header_cells) + r" \\"

    lines: list[str] = [
        r"\begin{table}",
        r"  \centering",
        r"  \footnotesize",
        f"  \\caption{{{caption}}}",
        f"  \\label{{tab:{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header_row}",
        r"    \midrule",
    ]

    for g_idx, group in enumerate(groups):
        n_rows = len(group.rows)
        for r_idx, row in enumerate(group.rows):
            # Build the data cells (label + value columns)
            data_parts: list[str] = [row.label]
            for col in columns:
                raw = row.values.get(col.key)
                if col.kind == "ci":
                    ci_val = raw if isinstance(raw, tuple) else None
                    data_parts.append(_fmt_ci(ci_val, num_precision))
                else:
                    num_val = (
                        raw
                        if isinstance(raw, (int, float)) and not isinstance(raw, bool)
                        else None
                    )
                    data_parts.append(_fmt_num(num_val, num_precision))  # type: ignore[arg-type]

            if r_idx == 0:
                # First row: \multirow cell on its own line, then data
                lines.append(f"    \\multirow{{{n_rows}}}{{*}}{{${group.symbol}$}}")
                lines.append("    & " + " & ".join(data_parts) + r" \\")
            else:
                lines.append("    & " + " & ".join(data_parts) + r" \\")

        if g_idx < len(groups) - 1:
            lines.append(r"    \midrule")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MLE assembly helpers
# ---------------------------------------------------------------------------


def subset_fits_latex_by_epoch(
    subset_fits: list[SubsetFitResult],
    epoch_names: Sequence[str],
    *,
    num_format: str = "1.6",
    num_precision: int = 6,
) -> dict[str, str]:
    """Return ``{epoch_name: latex_table}`` for per-epoch MLE tables.

    Each table has rows = manuscript traditions (subset fits) and columns
    Est. / SE / 95 % CI.  A tradition is omitted for a given epoch when it
    has zero manuscripts in that epoch (e.g. Lectionary in Ante Nicaea).

    Parameters
    ----------
    subset_fits : list[SubsetFitResult]
        Converged or not-converged MLE fits from
        :func:`~inf_bp_texts.analysis.subset_inference.fit_manuscript_subset`.
    epoch_names : Sequence[str]
        Canonical epoch names in display order.
    num_format : str
        Forwarded to :func:`rates_latex_table`.
    num_precision : int
        Forwarded to :func:`rates_latex_table`.

    Returns
    -------
    dict[str, str]
        Mapping from epoch name to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for epoch_name in epoch_names:
        birth_rows: list[LatexRateRow] = []
        death_rows: list[LatexRateRow] = []
        for fit in subset_fits:
            if not fit.success or not fit.epoch_results:
                continue
            ep = next((r for r in fit.epoch_results if r["period"] == epoch_name), None)
            if ep is None or ep.get("n_manuscripts", 0) == 0:
                continue
            birth_rows.append(
                LatexRateRow(
                    label=fit.label,
                    values={
                        "estimate": float(ep["birth_rates"]),
                        "se": (
                            float(ep["birth_rate_se"])
                            if ep.get("birth_rate_se") is not None
                            else None
                        ),
                        "ci": ep.get("birth_rate_ci"),
                    },
                )
            )
            death_rows.append(
                LatexRateRow(
                    label=fit.label,
                    values={
                        "estimate": float(ep["death_rates"]),
                        "se": (
                            float(ep["death_rate_se"])
                            if ep.get("death_rate_se") is not None
                            else None
                        ),
                        "ci": ep.get("death_rate_ci"),
                    },
                )
            )

        if not birth_rows:
            continue

        safe_name = epoch_name.lower().replace(" ", "-")
        tables[epoch_name] = rates_latex_table(
            [
                LatexParamGroup(symbol=r"\lambda", rows=birth_rows),
                LatexParamGroup(symbol=r"\mu", rows=death_rows),
            ],
            MLE_COLUMNS,
            caption=(
                f"MLE birth and death rate estimates with 95\\% confidence"
                f" intervals --- {epoch_name}"
            ),
            label=f"mle-{safe_name}",
            row_header="Tradition",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables


def subset_fits_latex_by_subset(
    subset_fits: list[SubsetFitResult],
    *,
    num_format: str = "1.6",
    num_precision: int = 6,
) -> dict[str, str]:
    """Return ``{subset_label: latex_table}`` for per-tradition MLE tables.

    Each table has rows = epochs and columns Est. / SE / 95 % CI.  Epochs with
    zero manuscripts are omitted.

    Parameters
    ----------
    subset_fits : list[SubsetFitResult]
        Converged MLE fits.
    num_format : str
        Forwarded to :func:`rates_latex_table`.
    num_precision : int
        Forwarded to :func:`rates_latex_table`.

    Returns
    -------
    dict[str, str]
        Mapping from tradition label to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for fit in subset_fits:
        if not fit.success or not fit.epoch_results:
            continue
        birth_rows = [
            LatexRateRow(
                label=ep["period"],
                values={
                    "estimate": float(ep["birth_rates"]),
                    "se": (
                        float(ep["birth_rate_se"])
                        if ep.get("birth_rate_se") is not None
                        else None
                    ),
                    "ci": ep.get("birth_rate_ci"),
                },
            )
            for ep in fit.epoch_results
            if ep.get("n_manuscripts", 0) > 0
        ]
        death_rows = [
            LatexRateRow(
                label=ep["period"],
                values={
                    "estimate": float(ep["death_rates"]),
                    "se": (
                        float(ep["death_rate_se"])
                        if ep.get("death_rate_se") is not None
                        else None
                    ),
                    "ci": ep.get("death_rate_ci"),
                },
            )
            for ep in fit.epoch_results
            if ep.get("n_manuscripts", 0) > 0
        ]
        if not birth_rows:
            continue

        tables[fit.label] = rates_latex_table(
            [
                LatexParamGroup(symbol=r"\lambda", rows=birth_rows),
                LatexParamGroup(symbol=r"\mu", rows=death_rows),
            ],
            MLE_COLUMNS,
            caption=(
                f"MLE birth and death rate estimates with 95\\% confidence"
                f" intervals --- {fit.label}"
            ),
            label=f"mle-{fit.key}",
            row_header="Epoch",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables


# ---------------------------------------------------------------------------
# Bayesian assembly helpers
# ---------------------------------------------------------------------------


def _posterior_stats(
    samples: np.ndarray,
    col_idx: int,
    cri_prob: float = 0.95,
) -> dict[str, float | tuple[float, float]]:
    """Compute mean, median, and CrI for one parameter column from posterior samples.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``[n_draws, n_params]``.
    col_idx : int
        Column index selecting the parameter of interest.
    cri_prob : float
        Credible interval probability (default 0.95 → 95 % CrI).

    Returns
    -------
    dict
        Keys ``"mean"``, ``"median"``, ``"cri"`` (tuple ``(lo, hi)``).
    """
    col = samples[:, col_idx]
    alpha = (1.0 - cri_prob) / 2.0
    lo, med, hi = (
        float(np.quantile(col, alpha)),
        float(np.median(col)),
        float(np.quantile(col, 1.0 - alpha)),
    )
    return {"mean": float(np.mean(col)), "median": med, "cri": (lo, hi)}


def bayesian_results_latex_by_epoch(
    bayesian_results: dict[str, dict[str, Any]],
    epoch_names: Sequence[str],
    *,
    num_format: str = "1.6",
    num_precision: int = 6,
    cri_prob: float = 0.95,
) -> dict[str, str]:
    """Return ``{epoch_name: latex_table}`` for per-epoch Bayesian tables.

    Each table has rows = traditions and columns Mean / Median / 95 % CrI.
    A tradition is omitted for a given epoch when that epoch does not appear in
    its ``epoch_names`` list (e.g. Lectionary subset fitted from 380 CE lacks
    ``"Ante Nicaea"``).

    Parameters
    ----------
    bayesian_results : dict[str, dict]
        Output of
        :func:`~inf_bp_texts.analysis.subset_inference.fit_manuscript_subsets_bayesian`.
        Keys are subset keys; each entry must contain ``"label"``,
        ``"epoch_names"``, ``"birth_rates"``, ``"death_rates"``, and
        ``"posterior_samples"`` (``{"birth_rates": ndarray, "death_rates": ndarray}``).
    epoch_names : Sequence[str]
        Canonical epoch names in display order.
    num_format : str
        Forwarded to :func:`rates_latex_table`.
    num_precision : int
        Forwarded to :func:`rates_latex_table`.
    cri_prob : float
        Credible interval probability (default 0.95).

    Returns
    -------
    dict[str, str]
        Mapping from epoch name to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for epoch_name in epoch_names:
        birth_rows: list[LatexRateRow] = []
        death_rows: list[LatexRateRow] = []
        for entry in bayesian_results.values():
            ep_names: list[str] = entry["epoch_names"]
            if epoch_name not in ep_names:
                continue
            i = ep_names.index(epoch_name)
            bs = np.asarray(entry["posterior_samples"]["birth_rates"])
            ds = np.asarray(entry["posterior_samples"]["death_rates"])
            birth_rows.append(
                LatexRateRow(
                    label=entry["label"], values=_posterior_stats(bs, i, cri_prob)
                )
            )
            death_rows.append(
                LatexRateRow(
                    label=entry["label"], values=_posterior_stats(ds, i, cri_prob)
                )
            )

        if not birth_rows:
            continue

        safe_name = epoch_name.lower().replace(" ", "-")
        tables[epoch_name] = rates_latex_table(
            [
                LatexParamGroup(symbol=r"\lambda", rows=birth_rows),
                LatexParamGroup(symbol=r"\mu", rows=death_rows),
            ],
            BAYESIAN_COLUMNS,
            caption=(
                f"Posterior birth and death rate estimates"
                f" (mean, median, {int(cri_prob * 100)}\\% CrI) --- {epoch_name}"
            ),
            label=f"bayes-{safe_name}",
            row_header="Tradition",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables


def _halflife_mle_values(
    ep: dict[str, Any],
) -> dict[str, float | tuple[float, float] | None]:
    """Derive half-life Est./SE/95% CI values from one MLE epoch result dict.

    Parameters
    ----------
    ep : dict
        One entry from
        :attr:`~inf_bp_texts.analysis.subset_inference.SubsetFitResult.epoch_results`.
        Expected keys: ``"half_life"``, optionally ``"death_rate_se"``,
        ``"death_rates"``, ``"death_rate_ci"``.

    Returns
    -------
    dict
        Keys ``"estimate"``, ``"se"``, ``"ci"`` ready for :class:`LatexRateRow`.
        ``None`` is returned for any component that cannot be computed.
    """
    hl = ep.get("half_life")
    estimate: float | None
    if hl is None or (isinstance(hl, float) and (math.isinf(hl) or math.isnan(hl))):
        estimate = None
    else:
        estimate = float(hl)

    # SE via delta method: SE_t = (ln2 / μ²) · SE_μ
    se: float | None = None
    mu = float(ep.get("death_rates", 0.0))
    se_mu = ep.get("death_rate_se")
    if se_mu is not None and mu > 0.0:
        se = float(math.log(2) / (mu**2)) * float(se_mu)

    # CI: invert and swap the μ CI bounds → (ln2/hi_μ, ln2/lo_μ)
    ci: tuple[float, float] | None = None
    raw_ci = ep.get("death_rate_ci")
    if raw_ci is not None:
        lo_mu, hi_mu = float(raw_ci[0]), float(raw_ci[1])
        if lo_mu > 0.0 and hi_mu > 0.0:
            ci = (math.log(2) / hi_mu, math.log(2) / lo_mu)

    return {"estimate": estimate, "se": se, "ci": ci}


def subset_fits_halflife_latex_by_epoch(
    subset_fits: list[SubsetFitResult],
    epoch_names: Sequence[str],
    *,
    num_format: str = "5.0",
    num_precision: int = 0,
) -> dict[str, str]:
    r"""Return ``{epoch_name: latex_table}`` for per-epoch MLE half-life tables.

    Each table has one :class:`LatexParamGroup` (symbol ``t_{1/2}``) with rows
    = manuscript traditions and columns Est. / SE / 95 % CI.  The half-life
    estimate is ``ln(2)/μ`` (years); SE comes from the delta method and CI from
    inverting and swapping the death-rate confidence-interval bounds.  A
    tradition is omitted for a given epoch when it has zero manuscripts in that
    epoch.

    Parameters
    ----------
    subset_fits : list[SubsetFitResult]
        Converged or not-converged MLE fits from
        :func:`~inf_bp_texts.analysis.subset_inference.fit_manuscript_subset`.
    epoch_names : Sequence[str]
        Canonical epoch names in display order.
    num_format : str
        ``table-format`` value for ``S`` columns (default ``"5.0"`` for
        whole-year values).
    num_precision : int
        Decimal places for numeric cells and ``\\num{}`` arguments
        (default ``0``).

    Returns
    -------
    dict[str, str]
        Mapping from epoch name to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for epoch_name in epoch_names:
        rows: list[LatexRateRow] = []
        for fit in subset_fits:
            if not fit.success or not fit.epoch_results:
                continue
            ep = next((r for r in fit.epoch_results if r["period"] == epoch_name), None)
            if ep is None or ep.get("n_manuscripts", 0) == 0:
                continue
            rows.append(LatexRateRow(label=fit.label, values=_halflife_mle_values(ep)))

        if not rows:
            continue

        safe_name = epoch_name.lower().replace(" ", "-")
        tables[epoch_name] = rates_latex_table(
            [LatexParamGroup(symbol=r"t_{1/2}", rows=rows)],
            MLE_COLUMNS,
            caption=(
                f"MLE manuscript half-lives (years) with 95\\% confidence"
                f" intervals --- {epoch_name}"
            ),
            label=f"mle-halflife-{safe_name}",
            row_header="Tradition",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables


def subset_fits_halflife_latex_by_subset(
    subset_fits: list[SubsetFitResult],
    *,
    num_format: str = "5.0",
    num_precision: int = 0,
) -> dict[str, str]:
    r"""Return ``{subset_label: latex_table}`` for per-tradition MLE half-life tables.

    Each table has one :class:`LatexParamGroup` (symbol ``t_{1/2}``) with rows
    = epochs and columns Est. / SE / 95 % CI.  Epochs with zero manuscripts are
    omitted.

    Parameters
    ----------
    subset_fits : list[SubsetFitResult]
        Converged MLE fits.
    num_format : str
        ``table-format`` value for ``S`` columns (default ``"5.0"``).
    num_precision : int
        Decimal places for numeric cells and ``\\num{}`` arguments
        (default ``0``).

    Returns
    -------
    dict[str, str]
        Mapping from tradition label to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for fit in subset_fits:
        if not fit.success or not fit.epoch_results:
            continue
        rows = [
            LatexRateRow(label=ep["period"], values=_halflife_mle_values(ep))
            for ep in fit.epoch_results
            if ep.get("n_manuscripts", 0) > 0
        ]
        if not rows:
            continue

        tables[fit.label] = rates_latex_table(
            [LatexParamGroup(symbol=r"t_{1/2}", rows=rows)],
            MLE_COLUMNS,
            caption=(
                f"MLE manuscript half-lives (years) with 95\\% confidence"
                f" intervals --- {fit.label}"
            ),
            label=f"mle-halflife-{fit.key}",
            row_header="Epoch",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables


def bayesian_results_halflife_latex_by_epoch(
    bayesian_results: dict[str, dict[str, Any]],
    epoch_names: Sequence[str],
    *,
    num_format: str = "5.0",
    num_precision: int = 0,
    cri_prob: float = 0.95,
) -> dict[str, str]:
    r"""Return ``{epoch_name: latex_table}`` for per-epoch Bayesian half-life tables.

    Half-lives are computed by transforming the death-rate posterior samples
    element-wise (``t = ln(2) / μ``); mean, median, and CrI are then derived
    directly from the transformed samples via :func:`_posterior_stats`.

    Parameters
    ----------
    bayesian_results : dict[str, dict]
        Output of
        :func:`~inf_bp_texts.analysis.subset_inference.fit_manuscript_subsets_bayesian`.
        Each entry must contain ``"label"``, ``"epoch_names"``, and
        ``"posterior_samples"`` with key ``"death_rates"`` (``ndarray``).
    epoch_names : Sequence[str]
        Canonical epoch names in display order.
    num_format : str
        ``table-format`` value for ``S`` columns (default ``"5.0"``).
    num_precision : int
        Decimal places for numeric cells and ``\\num{}`` arguments
        (default ``0``).
    cri_prob : float
        Credible interval probability (default 0.95).

    Returns
    -------
    dict[str, str]
        Mapping from epoch name to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for epoch_name in epoch_names:
        rows: list[LatexRateRow] = []
        for entry in bayesian_results.values():
            ep_names: list[str] = entry["epoch_names"]
            if epoch_name not in ep_names:
                continue
            i = ep_names.index(epoch_name)
            ds = np.asarray(entry["posterior_samples"]["death_rates"])
            hl_samples = np.log(2) / ds
            rows.append(
                LatexRateRow(
                    label=entry["label"],
                    values=_posterior_stats(hl_samples, i, cri_prob),
                )
            )

        if not rows:
            continue

        safe_name = epoch_name.lower().replace(" ", "-")
        tables[epoch_name] = rates_latex_table(
            [LatexParamGroup(symbol=r"t_{1/2}", rows=rows)],
            BAYESIAN_COLUMNS,
            caption=(
                f"Posterior manuscript half-lives (years)"
                f" (mean, median, {int(cri_prob * 100)}\\% CrI) --- {epoch_name}"
            ),
            label=f"bayes-halflife-{safe_name}",
            row_header="Tradition",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables


def bayesian_results_halflife_latex_by_subset(
    bayesian_results: dict[str, dict[str, Any]],
    *,
    num_format: str = "5.0",
    num_precision: int = 0,
    cri_prob: float = 0.95,
) -> dict[str, str]:
    r"""Return ``{label: latex_table}`` for per-tradition Bayesian half-life tables.

    Each table has one :class:`LatexParamGroup` (symbol ``t_{1/2}``) with rows
    = epochs.  Half-lives are derived from the posterior death-rate samples via
    an element-wise ``ln(2)/μ`` transform.

    Parameters
    ----------
    bayesian_results : dict[str, dict]
        Output of
        :func:`~inf_bp_texts.analysis.subset_inference.fit_manuscript_subsets_bayesian`.
    num_format : str
        ``table-format`` value for ``S`` columns (default ``"5.0"``).
    num_precision : int
        Decimal places for numeric cells and ``\\num{}`` arguments
        (default ``0``).
    cri_prob : float
        Credible interval probability (default 0.95).

    Returns
    -------
    dict[str, str]
        Mapping from tradition label to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for subset_key, entry in bayesian_results.items():
        ep_names: list[str] = entry["epoch_names"]
        ds = np.asarray(entry["posterior_samples"]["death_rates"])
        hl_samples = np.log(2) / ds

        rows = [
            LatexRateRow(label=ep, values=_posterior_stats(hl_samples, i, cri_prob))
            for i, ep in enumerate(ep_names)
        ]
        if not rows:
            continue

        label_str: str = entry["label"]
        tables[label_str] = rates_latex_table(
            [LatexParamGroup(symbol=r"t_{1/2}", rows=rows)],
            BAYESIAN_COLUMNS,
            caption=(
                f"Posterior manuscript half-lives (years)"
                f" (mean, median, {int(cri_prob * 100)}\\% CrI) --- {label_str}"
            ),
            label=f"bayes-halflife-{subset_key}",
            row_header="Epoch",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables


def bayesian_results_latex_by_subset(
    bayesian_results: dict[str, dict[str, Any]],
    *,
    num_format: str = "1.6",
    num_precision: int = 6,
    cri_prob: float = 0.95,
) -> dict[str, str]:
    """Return ``{subset_label: latex_table}`` for per-tradition Bayesian tables.

    Each table has rows = epochs and columns Mean / Median / 95 % CrI.

    Parameters
    ----------
    bayesian_results : dict[str, dict]
        Output of
        :func:`~inf_bp_texts.analysis.subset_inference.fit_manuscript_subsets_bayesian`.
    num_format : str
        Forwarded to :func:`rates_latex_table`.
    num_precision : int
        Forwarded to :func:`rates_latex_table`.
    cri_prob : float
        Credible interval probability (default 0.95).

    Returns
    -------
    dict[str, str]
        Mapping from tradition label to LaTeX table string.
    """
    tables: dict[str, str] = {}
    for subset_key, entry in bayesian_results.items():
        ep_names: list[str] = entry["epoch_names"]
        bs = np.asarray(entry["posterior_samples"]["birth_rates"])
        ds = np.asarray(entry["posterior_samples"]["death_rates"])

        birth_rows = [
            LatexRateRow(label=ep, values=_posterior_stats(bs, i, cri_prob))
            for i, ep in enumerate(ep_names)
        ]
        death_rows = [
            LatexRateRow(label=ep, values=_posterior_stats(ds, i, cri_prob))
            for i, ep in enumerate(ep_names)
        ]
        if not birth_rows:
            continue

        label_str: str = entry["label"]
        tables[label_str] = rates_latex_table(
            [
                LatexParamGroup(symbol=r"\lambda", rows=birth_rows),
                LatexParamGroup(symbol=r"\mu", rows=death_rows),
            ],
            BAYESIAN_COLUMNS,
            caption=(
                f"Posterior birth and death rate estimates"
                f" (mean, median, {int(cri_prob * 100)}\\% CrI) --- {label_str}"
            ),
            label=f"bayes-{subset_key}",
            row_header="Epoch",
            num_format=num_format,
            num_precision=num_precision,
        )
    return tables
