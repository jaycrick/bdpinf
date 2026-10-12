r"""Panels showing time-varying rates above the age densities they induce.

Each scenario is a pair of :class:`~bdpinf.models.rate_functions.RateFunction`
objects ``(birth, death)``.
:func:`scenario_panel` stacks the rates against calendar time $\tau$
over the age density against $a = T - \tau$, which makes the mirroring
between the two axes visible.
The *primary* curve is the one being studied (non-spine or spine age);
the other can be drawn faintly for reference.
The spine age has an atom at $a = T$, drawn as a point on a stem whose
height is a probability mass rather than a density.
"""

from typing import Literal

import altair as alt
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bdpinf.models.math_age_densities import (
    non_spine_age_log_density,
    spine_age_log_atom,
    spine_age_log_density,
)
from bdpinf.models.rate_functions import RateFunction
from bdpinf.visualization.altair_theme import PTOLQUAL

__all__ = [
    "age_density_frame",
    "rate_frame",
    "scenario_panel",
    "scenario_row",
]

AgeKind = Literal["non_spine", "spine"]
Scenario = tuple[RateFunction, RateFunction]

BIRTH_LABEL = "λ (birth)"
DEATH_LABEL = "μ (death)"
_RATE_COLORS = alt.Scale(domain=[BIRTH_LABEL, DEATH_LABEL], range=PTOLQUAL[:2])
_AGE_COLORS: dict[AgeKind, str] = {"non_spine": PTOLQUAL[3], "spine": PTOLQUAL[2]}
_DENSITY_HEIGHT = 140
_SECONDARY_OPACITY = 0.45


def _default_ages(end_time: jax.Array, n: int = 400) -> jax.Array:
    return jnp.linspace(1e-3, end_time - 1e-3, n)


def rate_frame(
    label: str, birth: RateFunction, death: RateFunction, *, n: int = 301
) -> pd.DataFrame:
    """Both rates against calendar time (columns ``t``, ``rate``, ``kind``)."""
    t = jnp.linspace(0.0, birth.end_time, n)
    return pd.DataFrame(
        {
            "scenario": label,
            "t": np.tile(np.asarray(t), 2),
            "rate": np.concatenate([np.asarray(birth(t)), np.asarray(death(t))]),
            "kind": [BIRTH_LABEL] * n + [DEATH_LABEL] * n,
        }
    )


def age_density_frame(
    label: str,
    birth: RateFunction,
    death: RateFunction,
    ages: jax.Array | None = None,
    *,
    which: AgeKind,
) -> pd.DataFrame:
    """Continuous age density (columns ``scenario``, ``age``, ``density``)."""
    ages = _default_ages(birth.end_time) if ages is None else ages
    log_density = (
        non_spine_age_log_density if which == "non_spine" else spine_age_log_density
    )
    return pd.DataFrame(
        {
            "scenario": label,
            "age": np.asarray(ages),
            "density": np.exp(np.asarray(log_density(ages, birth, death))),
        }
    )


def _density_layer(
    frame: pd.DataFrame, end: float, y_max: float, which: AgeKind, *, faint: bool
) -> alt.Chart:
    mark = alt.Chart(frame).mark_line(
        color=_AGE_COLORS[which],
        opacity=_SECONDARY_OPACITY if faint else 1.0,
        strokeDash=[4, 3] if faint else [1, 0],
    )
    return mark.encode(
        x=alt.X("age:Q", title="age a", scale=alt.Scale(domain=[0, end])),
        y=alt.Y("density:Q", title="density", scale=alt.Scale(domain=[0, y_max])),
    )


def _atom_layer(
    birth: RateFunction, death: RateFunction, *, faint: bool
) -> alt.LayerChart:
    end = float(birth.end_time)
    atom = pd.DataFrame(
        {"age": [end], "mass": [float(jnp.exp(spine_age_log_atom(birth, death)))]}
    )
    color = _AGE_COLORS["spine"]
    opacity = _SECONDARY_OPACITY if faint else 1.0
    stem = (
        alt.Chart(atom)
        .mark_rule(color=color, opacity=opacity)
        .encode(x="age:Q", y="mass:Q", y2=alt.value(_DENSITY_HEIGHT))
    )
    head = (
        alt.Chart(atom)
        .mark_point(color=color, opacity=opacity, filled=True, size=50)
        .encode(x="age:Q", y="mass:Q", tooltip=["mass:Q"])
    )
    return stem + head


def _y_extent(
    birth: RateFunction,
    death: RateFunction,
    ages: jax.Array | None,
    kinds: list[AgeKind],
) -> float:
    extent = 0.0
    for kind in kinds:
        extent = max(
            extent,
            float(
                age_density_frame("", birth, death, ages, which=kind)["density"].max()
            ),
        )
        if kind == "spine":
            extent = max(extent, float(jnp.exp(spine_age_log_atom(birth, death))))
    return extent


def _kinds(primary: AgeKind, show_secondary: bool) -> list[AgeKind]:
    secondary: AgeKind = "spine" if primary == "non_spine" else "non_spine"
    return [primary, secondary] if show_secondary else [primary]


def scenario_panel(
    label: str,
    birth: RateFunction,
    death: RateFunction,
    *,
    y_max: float,
    r_max: float,
    primary: AgeKind = "non_spine",
    show_secondary: bool = True,
    ages: jax.Array | None = None,
    width: int = 220,
) -> alt.VConcatChart:
    """Rates against calendar time above the age density against age.

    Parameters
    ----------
    label : str
        Panel title.
    birth, death : RateFunction
        The scenario's rates.
    y_max, r_max : float
        Upper limits of the density and rate axes (share across panels).
    primary : {"non_spine", "spine"}
        Age distribution drawn solid.
    show_secondary : bool
        Also draw the other age distribution faint and dashed.
    ages : jax.Array | None
        Age grid; defaults to 400 points in ``(0, T)``.
    width : int
        Panel width in pixels.

    """
    end = float(birth.end_time)
    rates = (
        alt.Chart(rate_frame(label, birth, death))
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="calendar time τ", scale=alt.Scale(domain=[0, end])),
            y=alt.Y("rate:Q", title="rate", scale=alt.Scale(domain=[0, r_max])),
            color=alt.Color("kind:N", scale=_RATE_COLORS, legend=None),
        )
        .properties(title=label, width=width, height=110)
    )
    layers: list[alt.Chart | alt.LayerChart] = []
    for kind in reversed(_kinds(primary, show_secondary)):
        faint = kind != primary
        frame = age_density_frame(label, birth, death, ages, which=kind)
        layers.append(_density_layer(frame, end, y_max, kind, faint=faint))
        if kind == "spine":
            layers.append(_atom_layer(birth, death, faint=faint))
    density = alt.layer(*layers).properties(width=width, height=_DENSITY_HEIGHT)
    return alt.vconcat(rates, density)


def scenario_row(
    scenarios: dict[str, Scenario],
    *,
    primary: AgeKind = "non_spine",
    show_secondary: bool = True,
    ages: jax.Array | None = None,
) -> alt.HConcatChart:
    """Panels for several scenarios side by side on common y-scales."""
    kinds = _kinds(primary, show_secondary)
    y_max = 1.05 * max(_y_extent(*v, ages, kinds) for v in scenarios.values())
    r_max = 1.1 * max(
        float(jnp.max(rate(jnp.linspace(0.0, rate.end_time, 301))))
        for v in scenarios.values()
        for rate in v
    )
    return alt.hconcat(
        *[
            scenario_panel(
                k,
                *v,
                y_max=y_max,
                r_max=r_max,
                primary=primary,
                show_secondary=show_secondary,
                ages=ages,
            )
            for k, v in scenarios.items()
        ]
    )
