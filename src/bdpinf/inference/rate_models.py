r"""Parametric families for the birth and death rates that can be fitted.

A *rate model* maps a vector of natural-scale parameters to a
:class:`~bdpinf.models.rate_functions.RateFunction`.
Each parameter carries a transform (:class:`Param`) that sends it to an
unconstrained scale for optimisation, so non-negativity of a rate (or a
proportion staying in $(0, 1)$) is enforced by construction.

Provided families:

- :class:`PiecewiseConstantModel`: one rate per epoch;
- :class:`PiecewiseLinearModel`: values at epoch boundaries (``continuous=True``,
  the default) or separate start and end values per epoch, which allows jumps.
  A jump-form linear model whose start and end values are tied
  (``flat_groups``) is the piecewise-constant model;
- :class:`SplineModel`: smooth quadratic or cubic rates through values at the
  knots (``quadratic``, ``cubic_natural``, ``cubic_clamped``, ``pchip`` and
  ``cubic_hermite``), kept non-negative by a guard in the likelihood.

:class:`BirthDeathModel` pairs a birth and a death family and owns the full
parameter list.
A new family (for example a sinusoid with amplitude, period and phase
parameters) only needs to satisfy the :class:`RateModel` protocol.

The epoch boundaries (knots) are fixed by default.
Passing a :class:`FreeKnots` to :class:`BirthDeathModel` makes them parameters
(``tau0 .. tau{K-1}``, shared by the birth and death rates) that are fitted
along with the rates, inside the ranges the caller allows.
"""

import dataclasses
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from bdpinf.models.rate_functions import (
    PiecewiseConstantRate,
    PiecewiseCubicRate,
    PiecewiseLinearRate,
    PiecewiseQuadraticRate,
    RateFunction,
)

if TYPE_CHECKING:
    from bdpinf.inference.constraints import Constraints

__all__ = [
    "BirthDeathModel",
    "FreeKnots",
    "Param",
    "PiecewiseConstantModel",
    "PiecewiseLinearModel",
    "RateModel",
    "SplineKind",
    "SplineModel",
    "Transform",
]

Transform = Literal["log", "identity", "logit", "interval"]
"""How a natural-scale parameter maps to the unconstrained optimisation scale.

``"interval"`` maps to ``(Param.lower, Param.upper)`` through a scaled logistic.
``"log"`` maps to ``(Param.lower, inf)`` as ``lower + exp(u)``; the default
``lower = 0`` is the plain log transform.
"""

_EDGE_FRACTION = 1e-9
_PAIR = 2

EpochTimes = Sequence[float] | np.ndarray | jax.Array
"""Epoch end times as given by the caller (stored as a tuple of floats)."""


@dataclasses.dataclass(frozen=True)
class Param:
    """A named scalar parameter and its transform.

    ``lower`` and ``upper`` bound the parameter when ``transform="interval"``.
    With ``transform="log"``, ``lower`` is a minimum value:
    ``natural = lower + exp(u)``, so the natural value is always above it
    (the unconstrained ``u`` is ``log(natural - lower)``).
    """

    name: str
    transform: Transform = "log"
    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        """Check that a log-transform minimum is finite and non-negative."""
        if self.transform == "log" and not (
            np.isfinite(self.lower) and self.lower >= 0.0
        ):
            msg = f"minimum value of '{self.name}' must be finite and >= 0"
            raise ValueError(msg)


class RateModel(Protocol):
    """A parametric family of rate functions on ``[0, T]``."""

    @property
    def params(self) -> tuple[Param, ...]:
        """The parameters, in the order :meth:`build` expects them."""
        ...

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        ...

    @property
    def n_epochs(self) -> int:
        """Number of epochs (one more than the number of interior knots)."""
        ...

    def build(
        self, values: jax.Array, epoch_end_times: jax.Array | None = None
    ) -> RateFunction:
        """Return the rate function for natural-scale ``values``.

        ``epoch_end_times`` replaces the family's own epochs (used when the knot
        times are being fitted); it may be a traced array.
        """
        ...

    def constant_values(self, value: float) -> jax.Array:
        """Parameter values for the constant rate ``value``."""
        ...

    def values_from_rate(self, rate: RateFunction) -> jax.Array:
        """Parameter values that approximate ``rate`` (to warm-start a fit)."""
        ...


_MIN_START_FACTOR = 1.0 + 1e-6


def _check_min_value(min_value: float) -> float:
    value = float(min_value)
    if not (np.isfinite(value) and value >= 0.0):
        msg = "min_value must be finite and >= 0"
        raise ValueError(msg)
    return value


def _above_minimum(values: jax.Array, min_value: float) -> jax.Array:
    """Lift ``values`` strictly above ``min_value`` (unchanged when it is 0)."""
    if min_value <= 0.0:
        return values
    return jnp.maximum(values, min_value * _MIN_START_FACTOR)


def _as_epoch_tuple(epoch_end_times: EpochTimes) -> tuple[float, ...]:
    times = tuple(float(t) for t in np.atleast_1d(np.asarray(epoch_end_times)))
    if not times or times[0] <= 0 or any(b <= a for a, b in itertools.pairwise(times)):
        msg = "epoch_end_times must be positive and strictly increasing"
        raise ValueError(msg)
    return times


@dataclasses.dataclass(frozen=True)
class PiecewiseConstantModel:
    """One non-negative rate per epoch.

    Parameters
    ----------
    epoch_end_times : sequence of float
        Strictly increasing epoch end times; the last is the observation time.
    min_value : float
        Minimum rate: every parameter satisfies ``value >= min_value`` by
        construction (``lower`` of its log transform).
        Default 0 (any positive rate).

    """

    epoch_end_times: EpochTimes
    min_value: float = 0.0

    def __post_init__(self) -> None:
        """Validate and freeze the epoch end times."""
        object.__setattr__(
            self, "epoch_end_times", _as_epoch_tuple(self.epoch_end_times)
        )
        object.__setattr__(self, "min_value", _check_min_value(self.min_value))

    @property
    def params(self) -> tuple[Param, ...]:
        """One log-scale parameter ``v{k}`` per epoch."""
        return tuple(
            Param(f"v{k}", lower=self.min_value)
            for k in range(len(self.epoch_end_times))
        )

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return len(self.epoch_end_times)

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        return float(self.epoch_end_times[-1])

    def build(
        self, values: jax.Array, epoch_end_times: jax.Array | None = None
    ) -> PiecewiseConstantRate:
        """Return the piecewise-constant rate with the given epoch values."""
        times = jnp.asarray(
            self.epoch_end_times if epoch_end_times is None else epoch_end_times
        )
        return PiecewiseConstantRate(values, times)

    def constant_values(self, value: float) -> jax.Array:
        """Return the same rate in every epoch."""
        return _above_minimum(
            jnp.full(len(self.epoch_end_times), value, dtype=jnp.float64),
            self.min_value,
        )

    def values_from_rate(self, rate: RateFunction) -> jax.Array:
        """Evaluate ``rate`` at the epoch midpoints."""
        ends = jnp.asarray(self.epoch_end_times)
        starts = jnp.concatenate([jnp.zeros(1), ends[:-1]])
        return _above_minimum(
            jnp.asarray(rate(0.5 * (starts + ends)), dtype=jnp.float64),
            self.min_value,
        )


@dataclasses.dataclass(frozen=True)
class PiecewiseLinearModel:
    r"""Rate that is linear within each epoch.

    With ``continuous=True`` the parameters are the values at time 0 and at
    each epoch end (``knot0 .. knotK``), so the rate has no jumps.
    With ``continuous=False`` each epoch has its own ``start{k}`` and
    ``end{k}`` values and the rate may jump at epoch boundaries.

    Parameters
    ----------
    epoch_end_times : sequence of float
        Strictly increasing epoch end times; the last is the observation time.
    continuous : bool
        Whether the rate is continuous across epoch boundaries.
    min_value : float
        Minimum rate: every parameter (and so the rate everywhere) satisfies
        ``value >= min_value`` by construction.
        Default 0.

    """

    epoch_end_times: EpochTimes
    continuous: bool = True
    min_value: float = 0.0

    def __post_init__(self) -> None:
        """Validate and freeze the epoch end times."""
        object.__setattr__(
            self, "epoch_end_times", _as_epoch_tuple(self.epoch_end_times)
        )
        object.__setattr__(self, "min_value", _check_min_value(self.min_value))

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return len(self.epoch_end_times)

    @property
    def params(self) -> tuple[Param, ...]:
        """Log-scale knot values, or start and end values per epoch."""
        low = self.min_value
        if self.continuous:
            return tuple(Param(f"knot{k}", lower=low) for k in range(self.n_epochs + 1))
        return tuple(
            Param(f"start{k}", lower=low) for k in range(self.n_epochs)
        ) + tuple(Param(f"end{k}", lower=low) for k in range(self.n_epochs))

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        return float(self.epoch_end_times[-1])

    def build(
        self, values: jax.Array, epoch_end_times: jax.Array | None = None
    ) -> PiecewiseLinearRate:
        """Return the piecewise-linear rate with the given parameter values."""
        times = jnp.asarray(
            self.epoch_end_times if epoch_end_times is None else epoch_end_times
        )
        if self.continuous:
            return PiecewiseLinearRate.continuous(values, times)
        return PiecewiseLinearRate(
            values[: self.n_epochs], values[self.n_epochs :], times
        )

    def constant_values(self, value: float) -> jax.Array:
        """Return a flat rate."""
        n = self.n_epochs + 1 if self.continuous else 2 * self.n_epochs
        return _above_minimum(jnp.full(n, value, dtype=jnp.float64), self.min_value)

    def values_from_rate(self, rate: RateFunction) -> jax.Array:
        """Evaluate ``rate`` at the knots (averaging the limits either side)."""
        ends = jnp.asarray(self.epoch_end_times)
        end = ends[-1]
        eps = _EDGE_FRACTION * end
        if self.continuous:
            knots = jnp.concatenate([jnp.zeros(1), ends])
            left = rate(jnp.clip(knots - eps, 0.0, end))
            right = rate(jnp.clip(knots + eps, 0.0, end))
            return _above_minimum(
                jnp.asarray(0.5 * (left + right), dtype=jnp.float64), self.min_value
            )
        starts = jnp.concatenate([jnp.zeros(1), ends[:-1]])
        return _above_minimum(
            jnp.asarray(
                jnp.concatenate([rate(starts + eps), rate(ends - eps)]),
                dtype=jnp.float64,
            ),
            self.min_value,
        )

    def flat_groups(self, prefix: str = "") -> list[list[str]]:
        """Equality groups that make a jump-form model piecewise constant.

        Each epoch's start and end value are tied, so the fitted rate is
        constant within every epoch.
        ``prefix`` is prepended to the parameter names (``"birth."`` for use with
        :class:`BirthDeathModel`).
        """
        if self.continuous:
            msg = "flat_groups needs continuous=False (epochs must be able to jump)"
            raise ValueError(msg)
        return [[f"{prefix}start{k}", f"{prefix}end{k}"] for k in range(self.n_epochs)]


SplineKind = Literal[
    "quadratic", "cubic_natural", "cubic_clamped", "pchip", "cubic_hermite"
]
"""Smooth rate families of :class:`SplineModel`."""

_SPLINE_KINDS = (
    "quadratic",
    "cubic_natural",
    "cubic_clamped",
    "pchip",
    "cubic_hermite",
)


@dataclasses.dataclass(frozen=True)
class SplineModel:
    r"""Smooth rate through its values at the knots, in one of several spline forms.

    The parameters are the rate values at time 0 and at each epoch end
    (``knot0 .. knotK``, log scale, at least ``min_value``) plus,
    depending on ``kind``, slope parameters (identity scale):

    ===============  =====================================  ========  ==========
    ``kind``         parameters                             count     continuity
    ===============  =====================================  ========  ==========
    quadratic        ``knot*``, ``slope0``                  K + 2     C1
    cubic_natural    ``knot*``                              K + 1     C2
    cubic_clamped    ``knot*``, ``slope0``, ``slopeT``      K + 3     C2
    pchip            ``knot*``                              K + 1     C1
    cubic_hermite    ``knot*``, ``slope0 .. slopeK``        2K + 2    C1
    ===============  =====================================  ========  ==========

    with $K$ epochs.
    A slope parameter is *scaled*: its natural value is the derivative times the
    observation time $T$ (so it has the units of a rate, like the knot values),
    and the rate built uses ``value / T``.
    ``quadratic`` is :meth:`PiecewiseQuadraticRate.spline`;
    ``cubic_natural``, ``cubic_clamped`` and ``pchip`` are the matching
    :class:`~bdpinf.models.rate_functions.PiecewiseCubicRate` constructors;
    ``cubic_hermite`` is the C1 cubic with free slope at every knot.

    Parameters
    ----------
    epoch_end_times : sequence of float
        Strictly increasing epoch end times; the last is the observation time.
    kind : str
        One of ``"quadratic"``, ``"cubic_natural"``, ``"cubic_clamped"``,
        ``"pchip"`` and ``"cubic_hermite"``.
    min_value : float
        Lower bound on every knot value (the ``lower`` of its log transform).
        Between knots a spline can still dip below the knot values and even
        below zero (never for ``pchip``);
        the likelihood returns ``-inf`` for a rate that is negative anywhere,
        so the fitter stays on the non-negative side.
        Default 0.

    """

    epoch_end_times: EpochTimes
    kind: SplineKind = "cubic_natural"
    min_value: float = 0.0

    def __post_init__(self) -> None:
        """Validate and freeze the epoch end times; check the kind."""
        object.__setattr__(
            self, "epoch_end_times", _as_epoch_tuple(self.epoch_end_times)
        )
        object.__setattr__(self, "min_value", _check_min_value(self.min_value))
        if self.kind not in _SPLINE_KINDS:
            msg = f"kind must be one of {_SPLINE_KINDS}, got '{self.kind}'"
            raise ValueError(msg)

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return len(self.epoch_end_times)

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        return float(self.epoch_end_times[-1])

    @property
    def _slope_names(self) -> tuple[str, ...]:
        if self.kind == "quadratic":
            return ("slope0",)
        if self.kind == "cubic_clamped":
            return ("slope0", "slopeT")
        if self.kind == "cubic_hermite":
            return tuple(f"slope{k}" for k in range(self.n_epochs + 1))
        return ()

    @property
    def params(self) -> tuple[Param, ...]:
        """Log-scale knot values, then identity-scale scaled slopes."""
        knots = tuple(
            Param(f"knot{k}", lower=self.min_value) for k in range(self.n_epochs + 1)
        )
        return knots + tuple(Param(n, "identity") for n in self._slope_names)

    def build(
        self, values: jax.Array, epoch_end_times: jax.Array | None = None
    ) -> PiecewiseQuadraticRate | PiecewiseCubicRate:
        """Return the spline rate for the given parameter values."""
        times = jnp.asarray(
            self.epoch_end_times if epoch_end_times is None else epoch_end_times
        )
        n = self.n_epochs + 1
        knots, slopes = values[:n], values[n:] / self.end_time
        match self.kind:
            case "quadratic":
                return PiecewiseQuadraticRate.spline(knots, slopes[0], times)
            case "cubic_natural":
                return PiecewiseCubicRate.natural_spline(knots, times)
            case "cubic_clamped":
                return PiecewiseCubicRate.clamped_spline(
                    knots, slopes[0], slopes[1], times
                )
            case "pchip":
                return PiecewiseCubicRate.pchip(knots, times)
            case _:
                return PiecewiseCubicRate.hermite(knots, slopes, times)

    def constant_values(self, value: float) -> jax.Array:
        """Flat rate: equal knot values and zero slopes."""
        knots = _above_minimum(
            jnp.full(self.n_epochs + 1, value, dtype=jnp.float64), self.min_value
        )
        return jnp.concatenate([knots, jnp.zeros(len(self._slope_names))])

    def values_from_rate(self, rate: RateFunction) -> jax.Array:
        """Knot values (and slopes) of ``rate`` at this family's knots.

        A piecewise linear, quadratic or cubic rate is evaluated exactly at the
        knots (left limits) and its own derivative gives the slopes;
        any other rate is averaged just either side of each knot and slopes come
        from central differences.
        For ``quadratic``, ``cubic_natural`` and ``cubic_clamped`` the spline
        built from these values equals ``rate`` whenever ``rate`` is itself such
        a spline on knots that are a subset of this family's, so a knot can be
        inserted without changing the function.
        """
        ends = jnp.asarray(self.epoch_end_times)
        end = ends[-1]
        knots_t = jnp.concatenate([jnp.zeros(1), ends])
        exact = isinstance(
            rate, PiecewiseLinearRate | PiecewiseQuadraticRate | PiecewiseCubicRate
        )
        if exact:
            values = jnp.asarray(rate(knots_t), dtype=jnp.float64)
            slope_at = rate.derivative  # type: ignore[union-attr]
        else:
            eps = _EDGE_FRACTION * end
            values = jnp.asarray(
                0.5
                * (
                    rate(jnp.clip(knots_t - eps, 0.0, end))
                    + rate(jnp.clip(knots_t + eps, 0.0, end))
                ),
                dtype=jnp.float64,
            )
            step = 1e-5 * end

            def slope_at(t: jax.Array) -> jax.Array:
                lo, hi = jnp.clip(t - step, 0.0, end), jnp.clip(t + step, 0.0, end)
                return (rate(hi) - rate(lo)) / (hi - lo)

        knot_values = _above_minimum(values, self.min_value)
        scale = end
        if self.kind == "quadratic":
            slopes = scale * slope_at(knots_t[:1])
        elif self.kind == "cubic_clamped":
            slopes = scale * slope_at(knots_t[jnp.array([0, -1])])
        elif self.kind == "cubic_hermite":
            slopes = scale * slope_at(knots_t)
        else:
            slopes = jnp.zeros(0)
        return jnp.concatenate([knot_values, jnp.asarray(slopes, dtype=jnp.float64)])


KnotBounds = Sequence[float] | Sequence[Sequence[float]]
"""One ``(low, high)`` range shared by all knots, or one range per knot."""


def _knot_ranges(
    n_knots: int, end_time: float, bounds: KnotBounds | None
) -> tuple[tuple[float, float], ...]:
    """Validate ``bounds`` and return one ``(low, high)`` pair per knot."""
    if bounds is None:
        return ((0.0, end_time),) * n_knots
    items = list(bounds)
    shared = len(items) == _PAIR and all(np.ndim(v) == 0 for v in items)
    if shared:
        ranges = ((float(items[0]), float(items[1])),) * n_knots  # type: ignore[arg-type]
    else:
        if len(items) != n_knots:
            msg = f"bounds must be one (low, high) pair or {n_knots} pairs"
            raise ValueError(msg)
        pairs = [np.asarray(item, dtype=np.float64) for item in items]
        if any(pair.shape != (_PAIR,) for pair in pairs):
            msg = "each knot range must be a (low, high) pair"
            raise ValueError(msg)
        ranges = tuple((float(a), float(b)) for a, b in pairs)
    if any(not (0.0 <= lo < hi <= end_time) for lo, hi in ranges):
        msg = f"knot ranges must satisfy 0 <= low < high <= end_time ({end_time})"
        raise ValueError(msg)
    if not shared and any(a[1] > b[0] for a, b in itertools.pairwise(ranges)):
        msg = "per-knot ranges must be ascending and must not overlap"
        raise ValueError(msg)
    return ranges


def _spacing_feasible(ranges: Sequence[tuple[float, float]], spacing: float) -> bool:
    """Whether ascending knots in ``ranges`` can be at least ``spacing`` apart."""
    position = -np.inf
    for lo, hi in ranges:
        position = max(lo, position + spacing)
        if position > hi + 1e-9:
            return False
    return True


def _spread_apart(
    times: Sequence[float], ranges: Sequence[tuple[float, float]], spacing: float
) -> tuple[float, ...]:
    """Nudge sorted ``times`` to be ``spacing`` apart inside their ranges."""
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    out = np.clip(np.sort(np.asarray(times, dtype=np.float64)), lo, hi)
    for j in range(1, len(out)):
        out[j] = max(out[j], out[j - 1] + spacing)
    for j in range(len(out) - 1, -1, -1):
        out[j] = min(out[j], hi[j])
        if j + 1 < len(out):
            out[j] = min(out[j], out[j + 1] - spacing)
    return tuple(float(t) for t in out)


def _checked_spacing(
    n_knots: int, ranges: Sequence[tuple[float, float]], min_spacing: float
) -> float:
    """Validate ``min_spacing`` and that the knots fit in ``ranges`` at that spacing."""
    spacing = float(min_spacing)
    if not np.isfinite(spacing) or spacing < 0:
        msg = "min_spacing must be a non-negative finite number"
        raise ValueError(msg)
    if spacing > 0 and not _spacing_feasible(ranges, spacing):
        msg = (
            f"{n_knots} knots at least {spacing} apart do not fit "
            f"in the knot ranges {ranges}"
        )
        raise ValueError(msg)
    return spacing


def _default_initial(
    n_knots: int, ranges: Sequence[tuple[float, float]], spacing: float
) -> tuple[float, ...]:
    """Evenly spaced (shared range) or midpoint (per-knot) starts, nudged apart."""
    lo, hi = ranges[0]
    if all(r == ranges[0] for r in ranges):
        initial = tuple(
            lo + (hi - lo) * (k + 1) / (n_knots + 1) for k in range(n_knots)
        )
    else:
        initial = tuple(0.5 * (a + b) for a, b in ranges)
    if spacing > 0 and any(b - a < spacing for a, b in itertools.pairwise(initial)):
        initial = _spread_apart(initial, ranges, spacing)
    return initial


def _checked_initial(
    initial: Sequence[float],
    n_knots: int,
    ranges: Sequence[tuple[float, float]],
    spacing: float,
) -> tuple[float, ...]:
    """Validate user-given starting times: count, order, ranges and spacing."""
    times = tuple(float(t) for t in initial)
    if len(times) != n_knots:
        msg = f"initial has {len(times)} times, expected {n_knots}"
        raise ValueError(msg)
    if any(b <= a for a, b in itertools.pairwise(times)) or any(
        not lo < t < hi for t, (lo, hi) in zip(times, ranges, strict=True)
    ):
        msg = "initial must be strictly increasing and inside the knot ranges"
        raise ValueError(msg)
    if any(b - a < spacing - 1e-9 for a, b in itertools.pairwise(times)):
        msg = f"initial knots must be at least min_spacing ({spacing}) apart"
        raise ValueError(msg)
    return times


@dataclasses.dataclass(frozen=True)
class FreeKnots:
    r"""Epoch boundaries that are fitted rather than fixed.

    The $K$ interior knots $\tau_1 < \dots < \tau_K$ become parameters
    ``tau0 .. tau{K-1}``, shared by the birth and death rates, so a model with
    free knots has $K + 1$ epochs ending at the observation time.
    Give only the number of knots to let each lie anywhere in $(0, T)$, or also
    the range it may lie in.

    Parameters
    ----------
    n_knots : int
        Number of interior knots.
    end_time : float
        Observation time $T$ (the last epoch end, not a parameter).
    bounds : (low, high) or sequence of K (low, high), optional
        One range shared by all knots (the knots are then ordered after
        fitting), or one range per knot, which must be ascending and not
        overlap.
        Default: $(0, T)$ for every knot.
    initial : sequence of float, optional
        Starting knot times, strictly increasing and inside their ranges.
        Default: evenly spaced in a shared range, the midpoints of per-knot ranges
        (nudged apart when that would break ``min_spacing``).
    min_spacing : float
        Minimum distance between any two sorted knots.
        :func:`~bdpinf.inference.knots.fit_knots` never evaluates or returns a
        placement closer than this (fixed knots included).
        Default 0 allows any placement.

    Examples
    --------
    Two knots anywhere in the window, or one knot in $[300, 900]$:

    >>> FreeKnots(2, end_time=1200.0).n_knots
    2
    >>> FreeKnots(1, end_time=1200.0, bounds=(300.0, 900.0)).initial
    (600.0,)

    """

    n_knots: int
    end_time: float
    bounds: KnotBounds | None = None
    initial: Sequence[float] | None = None
    min_spacing: float = 0.0

    def __post_init__(self) -> None:
        """Validate the ranges and freeze them (and the initial times) as tuples."""
        if self.n_knots < 1:
            msg = "n_knots must be at least 1"
            raise ValueError(msg)
        end = float(self.end_time)
        if not end > 0:
            msg = "end_time must be positive"
            raise ValueError(msg)
        ranges = _knot_ranges(self.n_knots, end, self.bounds)
        spacing = _checked_spacing(self.n_knots, ranges, self.min_spacing)
        if self.initial is None:
            initial = _default_initial(self.n_knots, ranges, spacing)
        else:
            initial = _checked_initial(self.initial, self.n_knots, ranges, spacing)
        object.__setattr__(self, "min_spacing", spacing)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "bounds", ranges)
        object.__setattr__(self, "initial", initial)

    @property
    def ranges(self) -> tuple[tuple[float, float], ...]:
        """The ``(low, high)`` range of each knot."""
        return self.bounds  # type: ignore[return-value]

    @property
    def initial_times(self) -> tuple[float, ...]:
        """Starting knot times."""
        return self.initial  # type: ignore[return-value]

    @property
    def params(self) -> tuple[Param, ...]:
        """One interval-scale parameter ``tau{k}`` per knot."""
        return tuple(
            Param(f"tau{k}", "interval", lo, hi)
            for k, (lo, hi) in enumerate(self.ranges)
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names ``tau0 .. tau{K-1}``."""
        return tuple(p.name for p in self.params)

    def epoch_end_times(self, values: jax.Array) -> jax.Array:
        """Sorted knot ``values`` followed by the observation time (differentiable)."""
        return jnp.concatenate(
            [jnp.sort(values), jnp.full(1, self.end_time, dtype=jnp.float64)]
        )


@dataclasses.dataclass(frozen=True)
class BirthDeathModel:
    """A birth-rate family paired with a death-rate family.

    Parameters are named ``birth.<name>`` and ``death.<name>`` and listed birth
    first.
    With ``knots`` the shared epoch boundaries ``tau<k>`` follow, and both
    families must have ``knots.n_knots + 1`` epochs.
    """

    birth: RateModel
    death: RateModel
    knots: FreeKnots | None = None

    def __post_init__(self) -> None:
        """Check the two families share an observation window (and the knots)."""
        if not np.isclose(self.birth.end_time, self.death.end_time):
            msg = "birth and death models must share the observation time"
            raise ValueError(msg)
        if self.knots is not None:
            expected = self.knots.n_knots + 1
            if self.birth.n_epochs != expected or self.death.n_epochs != expected:
                msg = (
                    f"with {self.knots.n_knots} free knots both families "
                    f"need {expected} epochs"
                )
                raise ValueError(msg)
            if not np.isclose(self.knots.end_time, self.end_time):
                msg = "knots and rate families must share the observation time"
                raise ValueError(msg)

    @classmethod
    def piecewise_constant(
        cls,
        epoch_end_times: EpochTimes | FreeKnots,
        *,
        min_birth: float = 0.0,
        min_death: float = 0.0,
    ) -> "BirthDeathModel":
        """Piecewise-constant birth and death rates on shared epochs.

        Pass a :class:`FreeKnots` to fit the epoch boundaries as well.
        ``min_birth`` and ``min_death`` are lower bounds on every birth and
        death rate (see :class:`PiecewiseConstantModel`).
        """
        epochs, knots = _template(epoch_end_times)
        return cls(
            PiecewiseConstantModel(epochs, min_birth),
            PiecewiseConstantModel(epochs, min_death),
            knots,
        )

    @classmethod
    def piecewise_linear(
        cls,
        epoch_end_times: EpochTimes | FreeKnots,
        *,
        continuous: bool = True,
        min_birth: float = 0.0,
        min_death: float = 0.0,
    ) -> "BirthDeathModel":
        """Piecewise-linear birth and death rates on shared epochs.

        Pass a :class:`FreeKnots` to fit the epoch boundaries as well.
        ``min_birth`` and ``min_death`` are lower bounds on every birth and
        death parameter, hence on the rates themselves.
        """
        epochs, knots = _template(epoch_end_times)
        return cls(
            PiecewiseLinearModel(epochs, continuous, min_birth),
            PiecewiseLinearModel(epochs, continuous, min_death),
            knots,
        )

    @classmethod
    def spline(
        cls,
        epoch_end_times: EpochTimes | FreeKnots,
        birth_kind: SplineKind = "cubic_natural",
        death_kind: SplineKind | Literal["linear"] = "linear",
        *,
        min_birth: float = 0.0,
        min_death: float = 0.0,
    ) -> "BirthDeathModel":
        """Smooth birth and death rates on shared epochs.

        ``birth_kind`` and ``death_kind`` are a :class:`SplineModel` kind, or
        ``"linear"`` for a continuous :class:`PiecewiseLinearModel`
        (the default death rate, so only the birth rate is smoothed).
        Pass a :class:`FreeKnots` to fit the epoch boundaries as well.
        ``min_birth`` and ``min_death`` bound the knot values from below.
        Spline rates that dip below zero between knots make the likelihood
        ``-inf`` (see :class:`SplineModel`).
        """
        epochs, knots = _template(epoch_end_times)

        def family(kind: str, minimum: float) -> RateModel:
            if kind == "linear":
                return PiecewiseLinearModel(epochs, True, minimum)
            return SplineModel(epochs, kind, minimum)  # type: ignore[arg-type]

        return cls(family(birth_kind, min_birth), family(death_kind, min_death), knots)

    def n_free(self, constraints: "Constraints | None" = None) -> int:
        """Count the free unconstrained coordinates (the $p$ of AIC and BIC).

        Fixed parameters and tied parameters (an equality group counts once)
        are removed; free knot times are included.
        This is the count :func:`~bdpinf.inference.fitting.fit` optimises over.
        """
        from bdpinf.inference.constraints import Constraints

        return (constraints or Constraints()).compile(self.params).n_free

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        return self.birth.end_time

    @property
    def n_birth(self) -> int:
        """Number of birth parameters."""
        return len(self.birth.params)

    @property
    def n_death(self) -> int:
        """Number of death parameters."""
        return len(self.death.params)

    @property
    def n_knots(self) -> int:
        """Number of free knot times (0 when the epochs are fixed)."""
        return 0 if self.knots is None else self.knots.n_knots

    @property
    def params(self) -> tuple[Param, ...]:
        """All parameters: birth, death, then free knot times, with name prefixes."""
        knots = () if self.knots is None else self.knots.params
        return (
            tuple(
                dataclasses.replace(p, name=f"birth.{p.name}")
                for p in self.birth.params
            )
            + tuple(
                dataclasses.replace(p, name=f"death.{p.name}")
                for p in self.death.params
            )
            + knots
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names in order."""
        return tuple(p.name for p in self.params)

    @property
    def birth_names(self) -> list[str]:
        """Names of the birth parameters."""
        return [f"birth.{p.name}" for p in self.birth.params]

    @property
    def death_names(self) -> list[str]:
        """Names of the death parameters."""
        return [f"death.{p.name}" for p in self.death.params]

    @property
    def knot_names(self) -> list[str]:
        """Names of the free knot-time parameters (empty with fixed epochs)."""
        return [] if self.knots is None else list(self.knots.names)

    def rates(self, natural: jax.Array) -> tuple[RateFunction, RateFunction]:
        """Return the ``(birth, death)`` rate functions for natural parameters."""
        n_b, n_d = self.n_birth, self.n_death
        birth_values, death_values = natural[:n_b], natural[n_b : n_b + n_d]
        if self.knots is None:
            return self.birth.build(birth_values), self.death.build(death_values)
        ends = self.knots.epoch_end_times(natural[n_b + n_d :])
        return (
            self.birth.build(birth_values, ends),
            self.death.build(death_values, ends),
        )

    def _knot_block(self, knot_times: ArrayLike | None) -> list[jax.Array]:
        if self.knots is None:
            return []
        times = self.knots.initial_times if knot_times is None else knot_times
        return [jnp.asarray(times, dtype=jnp.float64)]

    def _at_knots(self, family: RateModel, knot_times: ArrayLike | None) -> RateModel:
        """``family`` with its epochs replaced by the knot times (if knots are free)."""
        if self.knots is None or not dataclasses.is_dataclass(family):
            return family
        times = self.knots.initial_times if knot_times is None else knot_times
        ends = (*(float(t) for t in np.sort(np.asarray(times))), self.end_time)
        return dataclasses.replace(family, epoch_end_times=ends)  # type: ignore[type-var]

    def constant_natural(
        self, birth: float, death: float, knot_times: ArrayLike | None = None
    ) -> jax.Array:
        """Natural parameters for constant ``birth`` and ``death`` rates.

        Free knots start at ``knot_times`` (default: the initial times).
        """
        return jnp.concatenate(
            [
                self.birth.constant_values(birth),
                self.death.constant_values(death),
                *self._knot_block(knot_times),
            ]
        )

    def natural_from_rates(
        self,
        birth: RateFunction,
        death: RateFunction,
        knot_times: ArrayLike | None = None,
    ) -> jax.Array:
        """Natural parameters approximating the given rate functions.

        Free knots are placed at ``knot_times`` (default: the initial times) and
        the rates are evaluated on the epochs they define.
        """
        return jnp.concatenate(
            [
                self._at_knots(self.birth, knot_times).values_from_rate(birth),
                self._at_knots(self.death, knot_times).values_from_rate(death),
                *self._knot_block(knot_times),
            ]
        )

    def swapped(self, natural: jax.Array) -> jax.Array | None:
        """Return ``natural`` with birth and death exchanged, if the layouts match.

        Free knot times stay where they are.
        Exchanged values below a family's minimum are lifted above it when
        mapped to the unconstrained scale.
        """
        if [(p.name, p.transform) for p in self.birth.params] != [
            (p.name, p.transform) for p in self.death.params
        ]:
            return None
        n_b, n_d = self.n_birth, self.n_death
        return jnp.concatenate(
            [natural[n_b : n_b + n_d], natural[:n_b], natural[n_b + n_d :]]
        )

    def canonical(self, natural: ArrayLike) -> NDArray[np.float64]:
        """Return ``natural`` with the knot times in increasing order.

        Works on any array whose last axis lists the parameters, so it also
        orders posterior draws.
        Models with fixed epochs are returned unchanged.
        """
        out = np.array(natural, dtype=np.float64)
        if self.knots is not None:
            start = self.n_birth + self.n_death
            out[..., start:] = np.sort(out[..., start:], axis=-1)
        return out


def _template(
    epoch_end_times: EpochTimes | FreeKnots,
) -> tuple[EpochTimes, FreeKnots | None]:
    """Epochs for the family templates, and the knot spec when knots are free."""
    if isinstance(epoch_end_times, FreeKnots):
        epochs = (*epoch_end_times.initial_times, epoch_end_times.end_time)
        return epochs, epoch_end_times
    return epoch_end_times, None
