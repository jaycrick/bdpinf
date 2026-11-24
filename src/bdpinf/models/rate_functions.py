r"""Time-varying birth and death rate functions.

A *rate function* maps calendar time $t \in [0, T]$ to a non-negative
rate.
The age densities in :mod:`bdpinf.models.math_age_densities` need only three
things from a rate:

- its value $w(t)$ (``__call__``);
- its exact integral $\int_s^e w(t)\,dt$ (``integral``);
- the interior times at which quadrature should be split (``breakpoints``):
  where the rate is not smooth,
  or, for an oscillating rate, often enough that each piece
  is resolved by a single Gauss-Legendre panel.

Any class providing these (the :class:`RateFunction` protocol) can be used,
which leaves room for further spline or rational rates.
These families are provided: :class:`PiecewiseConstantRate`,
:class:`PiecewiseLinearRate`, :class:`PiecewiseExponentialRate`
(linear on the log scale, with a :meth:`~PiecewiseExponentialRate.gompertz`
constructor), :class:`PiecewiseQuadraticRate`
(including C1 quadratic splines), :class:`PiecewiseCubicRate`
(cubic Hermite pieces, with C1 and C2 spline constructors),
:class:`SinusoidalRate`,
the parametric hazards :class:`PowerRate` (with Weibull and Rayleigh
constructors), :class:`LogLogisticRate` and :class:`LomaxRate`,
and :class:`PiecewiseRate`, which glues any rate functions together
epoch by epoch in local time.
All are JAX pytrees, so they can be passed through ``jit``/``grad``.
"""

import dataclasses
from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from bdpinf.models.validation import is_concrete, is_concrete_value

__all__ = [
    "LogLogisticRate",
    "LomaxRate",
    "PiecewiseConstantRate",
    "PiecewiseCubicRate",
    "PiecewiseExponentialRate",
    "PiecewiseLinearRate",
    "PiecewiseQuadraticRate",
    "PiecewiseRate",
    "PowerRate",
    "RateFunction",
    "SinusoidalRate",
]


@runtime_checkable
class RateFunction(Protocol):
    r"""Protocol for a non-negative rate $w(t)$ on $[0, T]$."""

    @property
    def end_time(self) -> jax.Array:
        """Observation time $T$ (scalar)."""
        ...

    @property
    def breakpoints(self) -> jax.Array:
        """Increasing interior times at which to split quadrature, shape ``(k,)``.

        These are the times where the rate is not smooth.
        An oscillating rate also lists enough interior times
        that a single Gauss-Legendre panel resolves each piece.
        """
        ...

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at times ``t`` (any shape)."""
        ...

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w(t)\,dt$ for ``0 <= s <= e <= T``."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_f64(x: object) -> object:
    """Coerce lists, scalars and arrays to float64; leave tracers/sentinels alone."""
    if isinstance(x, jax.core.Tracer):
        return x
    if isinstance(x, list | tuple | float | int | np.ndarray | jax.Array):
        return jnp.asarray(x, dtype=jnp.float64)
    return x


def _epoch_starts(times: jax.Array) -> jax.Array:
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), times[:-1]])


def epoch_index(times: jax.Array, t: jax.Array) -> jax.Array:
    """Index of the epoch ``(tau_{k-1}, tau_k]`` containing each ``t`` (clipped)."""
    return jnp.clip(jnp.searchsorted(times, t, side="left"), 0, times.shape[0] - 1)


def _linear_value(
    v0: jax.Array, v1: jax.Array, times: jax.Array, t: jax.Array
) -> jax.Array:
    starts = _epoch_starts(times)
    slope = (v1 - v0) / (times - starts)
    k = epoch_index(times, t)
    return v0[k] + slope[k] * (t - starts[k])


def _linear_cumulative(
    v0: jax.Array, v1: jax.Array, times: jax.Array, t: jax.Array
) -> jax.Array:
    r"""Cumulative integral $\int_0^t w$, with ``t`` clipped to ``[0, T]``."""
    starts = _epoch_starts(times)
    dur = times - starts
    slope = (v1 - v0) / dur
    cum_before = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(0.5 * (v0 + v1) * dur)[:-1]]
    )
    t = jnp.clip(t, 0.0, times[-1])
    k = epoch_index(times, t)
    d = t - starts[k]
    return cum_before[k] + v0[k] * d + 0.5 * slope[k] * d * d


def _checked_epoch_times(times: object) -> np.ndarray:
    """Return ``times`` as a host array after checking it is a valid epoch grid."""
    t = np.asarray(times)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("epoch_end_times must be a non-empty 1-D array")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("epoch_end_times must be positive and strictly increasing")
    return t


def _validate_epochs(times: object, *values: object, positive: bool = False) -> None:
    """Value checks; skipped when any input is traced or not a concrete array.

    With ``positive=True`` the values must be strictly positive, not just non-negative.
    """
    if not is_concrete(times, *values):
        return
    t = _checked_epoch_times(times)
    for v in values:
        arr = np.asarray(v)
        if arr.shape != t.shape:
            raise ValueError(f"rate values shape {arr.shape} != times shape {t.shape}")
        if positive:
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError("rate values must be finite and strictly positive")
        elif np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError("rate values must be finite and non-negative")


# ---------------------------------------------------------------------------
# Piecewise constant
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PiecewiseConstantRate:
    """Rate that is constant within each epoch.

    Parameters
    ----------
    values : jax.Array
        Rate in each epoch, shape ``(n_epochs,)``, non-negative.
    epoch_end_times : jax.Array
        Strictly increasing epoch end times, shape ``(n_epochs,)``.
        The last entry is the observation time $T$.

    """

    values: jax.Array
    epoch_end_times: jax.Array

    def __post_init__(self) -> None:
        """Coerce to float64 arrays and validate concrete inputs."""
        object.__setattr__(self, "values", _as_f64(self.values))
        object.__setattr__(self, "epoch_end_times", _as_f64(self.epoch_end_times))
        _validate_epochs(self.epoch_end_times, self.values)

    @classmethod
    def constant(cls, value: float, end_time: float) -> "PiecewiseConstantRate":
        """Single-epoch constant rate on ``[0, end_time]``."""
        return cls(jnp.array([value]), jnp.array([end_time]))

    @property
    def end_time(self) -> jax.Array:
        """Observation time $T$."""
        return self.epoch_end_times[-1]

    @property
    def breakpoints(self) -> jax.Array:
        """Interior epoch boundaries."""
        return self.epoch_end_times[:-1]

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        return self.values[epoch_index(self.epoch_end_times, jnp.asarray(t))]

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$."""
        times = self.epoch_end_times
        return _linear_cumulative(self.values, self.values, times, jnp.asarray(e)) - (
            _linear_cumulative(self.values, self.values, times, jnp.asarray(s))
        )


# ---------------------------------------------------------------------------
# Piecewise linear
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PiecewiseLinearRate:
    r"""Rate that is linear within each epoch, with optional jumps between epochs.

    On epoch $k = (\tau_{k-1}, \tau_k]$ the rate interpolates linearly
    from ``start_values[k]`` at $\tau_{k-1}$ to ``end_values[k]`` at
    $\tau_k$.
    Setting ``start_values[k+1] != end_values[k]`` gives a jump at
    $\tau_k$; :meth:`continuous` builds the jump-free case.

    Parameters
    ----------
    start_values, end_values : jax.Array
        Rate at the start and end of each epoch, shape ``(n_epochs,)``,
        non-negative.
    epoch_end_times : jax.Array
        Strictly increasing epoch end times, shape ``(n_epochs,)``.

    """

    start_values: jax.Array
    end_values: jax.Array
    epoch_end_times: jax.Array

    def __post_init__(self) -> None:
        """Coerce to float64 arrays and validate concrete inputs."""
        object.__setattr__(self, "start_values", _as_f64(self.start_values))
        object.__setattr__(self, "end_values", _as_f64(self.end_values))
        object.__setattr__(self, "epoch_end_times", _as_f64(self.epoch_end_times))
        _validate_epochs(self.epoch_end_times, self.start_values, self.end_values)

    @classmethod
    def continuous(
        cls, knot_values: jax.Array, epoch_end_times: jax.Array
    ) -> "PiecewiseLinearRate":
        """Jump-free rate from values at ``0`` and at each epoch end.

        ``knot_values`` has length ``n_epochs + 1``.
        """
        knots = jnp.asarray(knot_values, dtype=jnp.float64)
        return cls(knots[:-1], knots[1:], jnp.asarray(epoch_end_times))

    @classmethod
    def from_constant(
        cls, values: jax.Array, epoch_end_times: jax.Array
    ) -> "PiecewiseLinearRate":
        """Piecewise-constant rate expressed as a (zero-slope) linear rate."""
        v = jnp.asarray(values, dtype=jnp.float64)
        return cls(v, v, jnp.asarray(epoch_end_times))

    @property
    def end_time(self) -> jax.Array:
        """Observation time $T$."""
        return self.epoch_end_times[-1]

    @property
    def breakpoints(self) -> jax.Array:
        """Interior epoch boundaries."""
        return self.epoch_end_times[:-1]

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        return _linear_value(
            self.start_values, self.end_values, self.epoch_end_times, jnp.asarray(t)
        )

    def derivative(self, t: jax.Array) -> jax.Array:
        """Slope at ``t`` (the left limit at an epoch boundary)."""
        times = self.epoch_end_times
        k = epoch_index(times, jnp.asarray(t))
        dur = times - _epoch_starts(times)
        return ((self.end_values - self.start_values) / dur)[k]

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$."""
        args = (self.start_values, self.end_values, self.epoch_end_times)
        return _linear_cumulative(*args, jnp.asarray(e)) - _linear_cumulative(
            *args, jnp.asarray(s)
        )


# ---------------------------------------------------------------------------
# Piecewise exponential (linear on the log scale)
# ---------------------------------------------------------------------------

_EXP_SERIES_THRESHOLD = 1e-3


def _expm1_over(x: jax.Array) -> jax.Array:
    r"""$(e^x - 1)/x$, equal to 1 at $x = 0$, accurate and gradient-safe near it."""
    small = jnp.abs(x) < _EXP_SERIES_THRESHOLD
    x_safe = jnp.where(small, 1.0, x)
    series = 1.0 + x / 2.0 + x * x / 6.0 + x * x * x / 24.0
    return jnp.where(small, series, jnp.expm1(x_safe) / x_safe)


def _log_growth(v0: jax.Array, v1: jax.Array, dur: jax.Array) -> jax.Array:
    r"""Log-scale slope $c = \ln(v_1 / v_0) / d$ of each epoch."""
    return (jnp.log(v1) - jnp.log(v0)) / dur


@dataclasses.dataclass(frozen=True)
class PiecewiseExponentialRate:
    r"""Rate that is linear on the log scale within each epoch.

    On epoch $k = (\tau_{k-1}, \tau_k]$ of duration $d_k$,

    $$
    w(t) = a_k \left(\frac{b_k}{a_k}\right)^{(t - \tau_{k-1})/d_k}
         = a_k\, e^{c_k (t - \tau_{k-1})}, \qquad
    c_k = \frac{\ln(b_k / a_k)}{d_k},
    $$

    where $a_k$ is ``start_values[k]`` and $b_k$ is ``end_values[k]``.
    Setting ``start_values[k+1] != end_values[k]`` gives a jump at
    $\tau_k$; :meth:`continuous` builds the jump-free case.

    Parameters
    ----------
    start_values, end_values : jax.Array
        Rate at the start and end of each epoch, shape ``(n_epochs,)``,
        finite and strictly positive (the log scale excludes zero).
    epoch_end_times : jax.Array
        Strictly increasing epoch end times, shape ``(n_epochs,)``.

    Notes
    -----
    Concrete inputs are validated; validation is skipped under tracing.

    The integral over an epoch from its start to the offset $u$ is
    $a_k\,\mathrm{expm1}(c_k u)/c_k$, evaluated as
    $a_k u\,\varphi(c_k u)$ with $\varphi(x) = (e^x - 1)/x$,
    using a short series for $|x| < 10^{-3}$ so that a constant epoch
    ($c_k = 0$) is exact and its gradient is finite.

    """

    start_values: jax.Array
    end_values: jax.Array
    epoch_end_times: jax.Array

    def __post_init__(self) -> None:
        """Coerce to float64 arrays and validate concrete inputs."""
        object.__setattr__(self, "start_values", _as_f64(self.start_values))
        object.__setattr__(self, "end_values", _as_f64(self.end_values))
        object.__setattr__(self, "epoch_end_times", _as_f64(self.epoch_end_times))
        _validate_epochs(
            self.epoch_end_times, self.start_values, self.end_values, positive=True
        )

    @classmethod
    def continuous(
        cls, knot_values: jax.Array, epoch_end_times: jax.Array
    ) -> "PiecewiseExponentialRate":
        """Jump-free rate from values at ``0`` and at each epoch end.

        ``knot_values`` has length ``n_epochs + 1`` and must be strictly positive.
        """
        knots = jnp.asarray(knot_values, dtype=jnp.float64)
        return cls(knots[:-1], knots[1:], jnp.asarray(epoch_end_times))

    @classmethod
    def from_constant(
        cls, values: jax.Array, epoch_end_times: jax.Array
    ) -> "PiecewiseExponentialRate":
        """Piecewise-constant rate expressed as a (zero log-slope) exponential rate."""
        v = jnp.asarray(values, dtype=jnp.float64)
        return cls(v, v, jnp.asarray(epoch_end_times))

    @classmethod
    def gompertz(
        cls, a: float | jax.Array, b: float | jax.Array, end_time: float
    ) -> "PiecewiseExponentialRate":
        r"""Gompertz rate $w(t) = a e^{b t}$ on $[0, T]$ as a single epoch.

        Here $a > 0$ and $b$ is any real number
        (the rate decays when $b < 0$);
        the epoch runs from $a$ at time 0 to $a e^{bT}$ at time $T$.
        """
        a = jnp.asarray(a, dtype=jnp.float64)
        b = jnp.asarray(b, dtype=jnp.float64)
        end = jnp.asarray(end_time, dtype=jnp.float64)
        return cls(a[None], (a * jnp.exp(b * end))[None], end[None])

    @property
    def end_time(self) -> jax.Array:
        """Observation time $T$."""
        return self.epoch_end_times[-1]

    @property
    def breakpoints(self) -> jax.Array:
        """Interior epoch boundaries."""
        return self.epoch_end_times[:-1]

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        times = self.epoch_end_times
        starts = _epoch_starts(times)
        growth = _log_growth(self.start_values, self.end_values, times - starts)
        t = jnp.asarray(t)
        k = epoch_index(times, t)
        return self.start_values[k] * jnp.exp(growth[k] * (t - starts[k]))

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$, as a difference of cumulative integrals.

        Within epoch $k$, $\int_{\tau_{k-1}}^{\tau_{k-1}+u} w
        = a_k\,\mathrm{expm1}(c_k u)/c_k$.
        """
        return self._cumulative(jnp.asarray(e)) - self._cumulative(jnp.asarray(s))

    def _cumulative(self, t: jax.Array) -> jax.Array:
        times = self.epoch_end_times
        starts = _epoch_starts(times)
        dur = times - starts
        growth = _log_growth(self.start_values, self.end_values, dur)
        full = self.start_values * dur * _expm1_over(growth * dur)
        cum_before = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(full)[:-1]]
        )
        t = jnp.clip(t, 0.0, times[-1])
        k = epoch_index(times, t)
        u = t - starts[k]
        return cum_before[k] + self.start_values[k] * u * _expm1_over(growth[k] * u)


# ---------------------------------------------------------------------------
# Piecewise quadratic
# ---------------------------------------------------------------------------


def _quadratic_coefficients(
    v0: jax.Array, vm: jax.Array, v1: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Coefficients of $q(u) = a_0 + a_1 u + a_2 u^2$ for $u \in [0, 1]$."""
    return v0, -3.0 * v0 + 4.0 * vm - v1, 2.0 * v0 - 4.0 * vm + 2.0 * v1


def _validate_quadratic(
    times: object,
    start: object,
    mid: object,
    end: object,
    *,
    nonnegative: bool = True,
) -> None:
    """Finite and well-shaped, and (optionally) non-negative on every epoch."""
    if not is_concrete(times, start, mid, end):
        return
    t = _checked_epoch_times(times)
    v0, vm, v1 = (np.asarray(x, dtype=float) for x in (start, mid, end))
    for v in (v0, vm, v1):
        if v.shape != t.shape:
            raise ValueError(f"rate values shape {v.shape} != times shape {t.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("rate values must be finite")
    if not nonnegative:
        return
    a0, a1, a2 = (np.asarray(c) for c in _quadratic_coefficients(v0, vm, v1))
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = np.where(a2 != 0, -a1 / (2.0 * a2), np.nan)
        vertex_value = a0 + a1 * vertex + a2 * vertex * vertex
    interior = (a2 > 0) & (vertex > 0) & (vertex < 1)
    lowest = np.minimum(np.minimum(v0, v1), np.where(interior, vertex_value, np.inf))
    scale = np.maximum(np.maximum(np.abs(v0), np.abs(vm)), np.abs(v1))
    bad = np.flatnonzero(lowest < -1e-12 * scale)
    if bad.size:
        k = bad[0]
        raise ValueError(
            f"rate must be non-negative on every epoch; epoch {k} dips to "
            f"{lowest[k]:.6g} (start {v0[k]:.6g}, mid {vm[k]:.6g}, end {v1[k]:.6g})"
        )


@dataclasses.dataclass(frozen=True)
class PiecewiseQuadraticRate:
    r"""Rate that is a quadratic in time within each epoch.

    On epoch $k = (\tau_{k-1}, \tau_k]$ of duration $d_k$, with
    $u = (t - \tau_{k-1}) / d_k \in [0, 1]$, the rate is the quadratic
    through its values $v_0$, $v_m$, $v_1$ at the start, midpoint and end
    (three-point Lagrange form),

    $$
    w = v_0 (2u - 1)(u - 1) + 4 v_m\,u(1 - u) + v_1\,u(2u - 1).
    $$

    Specifying values at points on the plot, rather than polynomial
    coefficients, is the point of this parametrisation.
    Setting the next epoch's start value different from this epoch's end value
    gives a jump; :meth:`continuous`, :meth:`spline` and :meth:`from_linear`
    build common cases.

    Parameters
    ----------
    start_values, mid_values, end_values : jax.Array
        Rate at the start, midpoint and end of each epoch, shape ``(n_epochs,)``.
    epoch_end_times : jax.Array
        Strictly increasing epoch end times, shape ``(n_epochs,)``.

    Notes
    -----
    Concrete inputs must be finite with matching shapes
    (checked in the constructor, and skipped under tracing).
    The constructor does **not** check non-negativity,
    so that unflattening the pytree (``jax.tree.map``, ``jit``) never raises
    and the class matches :class:`PiecewiseCubicRate`:
    a rate that dips below zero is handled by :meth:`min_values`
    and the likelihood guard, which returns $-\infty$.
    Only the :meth:`continuous` and :meth:`from_linear` constructors check
    non-negativity (at the three stored values and, when the vertex lies
    inside the epoch, at the vertex,
    with a rounding allowance of $10^{-12}$ times the largest stored value;
    the error message names the offending epoch).
    :meth:`spline`, and so ``SplineModel.build``, never checks it.

    The integral is exact: from the epoch start to offset $u$ it is
    $d_k\,(a_0 u + a_1 u^2 / 2 + a_2 u^3 / 3)$ with
    $a_0 = v_0$, $a_1 = -3 v_0 + 4 v_m - v_1$, $a_2 = 2 v_0 - 4 v_m + 2 v_1$;
    over a full epoch this is Simpson's rule $d_k (v_0 + 4 v_m + v_1) / 6$.

    """

    start_values: jax.Array
    mid_values: jax.Array
    end_values: jax.Array
    epoch_end_times: jax.Array

    def __post_init__(self) -> None:
        """Coerce to float64 arrays and validate concrete inputs."""
        object.__setattr__(self, "start_values", _as_f64(self.start_values))
        object.__setattr__(self, "mid_values", _as_f64(self.mid_values))
        object.__setattr__(self, "end_values", _as_f64(self.end_values))
        object.__setattr__(self, "epoch_end_times", _as_f64(self.epoch_end_times))
        _validate_quadratic(
            self.epoch_end_times,
            self.start_values,
            self.mid_values,
            self.end_values,
            nonnegative=False,
        )

    @classmethod
    def continuous(
        cls,
        knot_values: jax.Array,
        mid_values: jax.Array,
        epoch_end_times: jax.Array,
    ) -> "PiecewiseQuadraticRate":
        """Jump-free rate from values at ``0`` and at each epoch end, plus midpoints.

        ``knot_values`` has length ``n_epochs + 1``;
        ``mid_values`` has length ``n_epochs``.
        """
        knots = jnp.asarray(knot_values, dtype=jnp.float64)
        mid = jnp.asarray(mid_values, dtype=jnp.float64)
        times = jnp.asarray(epoch_end_times)
        _validate_quadratic(times, knots[:-1], mid, knots[1:])
        return cls(knots[:-1], mid, knots[1:], times)

    @classmethod
    def spline(
        cls,
        knot_values: jax.Array,
        initial_slope: float | jax.Array,
        epoch_end_times: jax.Array,
    ) -> "PiecewiseQuadraticRate":
        r"""C1 quadratic spline through the knot values.

        ``knot_values`` has length ``n_epochs + 1`` and ``initial_slope`` is
        the derivative at time 0.
        On epoch $k$ the quadratic is fixed by its start value $v_k$,
        start slope $s_k$ and end value $v_{k+1}$;
        the next epoch's start slope is this epoch's end slope,
        $s_{k+1} = 2 (v_{k+1} - v_k) / d_k - s_k$.
        The rate may dip below zero between knots; this does not raise
        (finiteness and shape are still checked).
        Use :meth:`min_values` or the likelihood guard to handle a dip.
        """
        knots = jnp.asarray(knot_values, dtype=jnp.float64)
        times = jnp.asarray(epoch_end_times, dtype=jnp.float64)
        dur = times - _epoch_starts(times)
        delta = knots[1:] - knots[:-1]

        def step(
            slope: jax.Array, xs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            d, dv = xs
            return 2.0 * dv / d - slope, slope

        _, start_slopes = jax.lax.scan(
            step, jnp.asarray(initial_slope, dtype=jnp.float64), (dur, delta)
        )
        curvature = (delta - start_slopes * dur) / (dur * dur)
        mid = knots[:-1] + 0.5 * start_slopes * dur + 0.25 * curvature * dur * dur
        return cls(knots[:-1], mid, knots[1:], times)

    @classmethod
    def from_linear(
        cls,
        start_values: jax.Array,
        end_values: jax.Array,
        epoch_end_times: jax.Array,
    ) -> "PiecewiseQuadraticRate":
        """Zero-curvature rate equal to the matching :class:`PiecewiseLinearRate`."""
        v0 = jnp.asarray(start_values, dtype=jnp.float64)
        v1 = jnp.asarray(end_values, dtype=jnp.float64)
        mid = 0.5 * (v0 + v1)
        times = jnp.asarray(epoch_end_times)
        _validate_quadratic(times, v0, mid, v1)
        return cls(v0, mid, v1, times)

    @property
    def end_time(self) -> jax.Array:
        """Observation time $T$."""
        return self.epoch_end_times[-1]

    @property
    def breakpoints(self) -> jax.Array:
        """Interior epoch boundaries."""
        return self.epoch_end_times[:-1]

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        times = self.epoch_end_times
        starts = _epoch_starts(times)
        t = jnp.asarray(t)
        k = epoch_index(times, t)
        a0, a1, a2 = _quadratic_coefficients(
            self.start_values[k], self.mid_values[k], self.end_values[k]
        )
        u = (t - starts[k]) / (times[k] - starts[k])
        return a0 + u * (a1 + u * a2)

    def flat_unless(self, ok: jax.Array) -> "PiecewiseQuadraticRate":
        """Return this rate where ``ok`` is true, else the constant rate 1.

        Lets a guard evaluate a harmless rate on infeasible parameters,
        so the gradient through ``jnp.where`` carries no NaN.
        """
        one = jnp.ones_like(self.start_values)
        return PiecewiseQuadraticRate(
            jnp.where(ok, self.start_values, one),
            jnp.where(ok, self.mid_values, one),
            jnp.where(ok, self.end_values, one),
            self.epoch_end_times,
        )

    def local_coefficients(self) -> np.ndarray:
        r"""Host array ``(n_epochs, 3)`` of local-time coefficients $(a, b, c)$."""
        times = np.asarray(self.epoch_end_times, dtype=np.float64)
        dur = times - np.concatenate([[0.0], times[:-1]])
        a0, a1, a2 = (
            np.asarray(c, dtype=np.float64)
            for c in _quadratic_coefficients(
                self.start_values, self.mid_values, self.end_values
            )
        )
        return np.stack([a0, a1 / dur, a2 / dur**2], axis=1)

    def min_values(self) -> jax.Array:
        """Exact minimum of the rate on each epoch, shape ``(n_epochs,)``.

        Endpoint values and, when the vertex lies inside the epoch,
        the vertex value.
        Jittable (no Python branching on traced values).
        """
        v0, v1 = self.start_values, self.end_values
        a0, a1, a2 = _quadratic_coefficients(v0, self.mid_values, v1)
        curved = a2 != 0
        vertex = jnp.where(curved, -a1 / jnp.where(curved, 2.0 * a2, 1.0), 0.0)
        interior = (a2 > 0) & (vertex > 0) & (vertex < 1)
        vertex_value = a0 + vertex * (a1 + vertex * a2)
        return jnp.minimum(
            jnp.minimum(v0, v1), jnp.where(interior, vertex_value, jnp.inf)
        )

    def derivative(self, t: jax.Array) -> jax.Array:
        """Slope at ``t`` (the left limit at an epoch boundary)."""
        times = self.epoch_end_times
        t = jnp.asarray(t)
        k = epoch_index(times, t)
        starts = _epoch_starts(times)
        _, a1, a2 = _quadratic_coefficients(
            self.start_values[k], self.mid_values[k], self.end_values[k]
        )
        dur = times[k] - starts[k]
        u = (t - starts[k]) / dur
        return (a1 + 2.0 * a2 * u) / dur

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$, as a difference of cumulative integrals."""
        return self._cumulative(jnp.asarray(e)) - self._cumulative(jnp.asarray(s))

    def _cumulative(self, t: jax.Array) -> jax.Array:
        times = self.epoch_end_times
        starts = _epoch_starts(times)
        dur = times - starts
        full = dur * (self.start_values + 4.0 * self.mid_values + self.end_values) / 6.0
        cum_before = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(full)[:-1]]
        )
        t = jnp.clip(t, 0.0, times[-1])
        k = epoch_index(times, t)
        a0, a1, a2 = _quadratic_coefficients(
            self.start_values[k], self.mid_values[k], self.end_values[k]
        )
        u = (t - starts[k]) / dur[k]
        return cum_before[k] + dur[k] * u * (a0 + u * (a1 / 2.0 + u * a2 / 3.0))


# ---------------------------------------------------------------------------
# Piecewise cubic (Hermite form)
# ---------------------------------------------------------------------------


def _cubic_coefficients(
    v0: jax.Array, v1: jax.Array, m0: jax.Array, m1: jax.Array, dur: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Coefficients of $c_0 + c_1 u + c_2 u^2 + c_3 u^3$, $u \in [0, 1]$.

    ``m0`` and ``m1`` are slopes in time units; the Hermite form uses ``dur * m``.
    """
    s0, s1 = dur * m0, dur * m1
    return (
        v0,
        s0,
        -3.0 * v0 - 2.0 * s0 + 3.0 * v1 - s1,
        2.0 * v0 + s0 - 2.0 * v1 + s1,
    )


def _validate_cubic(times: object, *values: object) -> None:
    """Shapes and finiteness of concrete inputs (non-negativity is the guard's job)."""
    if not is_concrete(times, *values):
        return
    t = _checked_epoch_times(times)
    for v in values:
        arr = np.asarray(v, dtype=float)
        if arr.shape != t.shape:
            raise ValueError(f"rate values shape {arr.shape} != times shape {t.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("rate values and slopes must be finite")


def _knot_grid(times: jax.Array) -> jax.Array:
    """Knot positions ``0, tau_1, ..., tau_K`` (length ``n_epochs + 1``)."""
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), times])


def _spline_slopes(
    values: jax.Array,
    times: jax.Array,
    start: jax.Array | None,
    end: jax.Array | None,
) -> jax.Array:
    """Knot slopes of the C2 spline (natural end where ``start``/``end`` is None)."""
    x = _knot_grid(times)
    h = x[1:] - x[:-1]
    delta = (values[1:] - values[:-1]) / h
    n = values.shape[0]
    a = jnp.zeros((n, n), dtype=jnp.float64)
    rhs = jnp.zeros(n, dtype=jnp.float64)
    inner = jnp.arange(1, n - 1)
    a = a.at[inner, inner - 1].set(h[1:])
    a = a.at[inner, inner].set(2.0 * (h[:-1] + h[1:]))
    a = a.at[inner, inner + 1].set(h[:-1])
    rhs = rhs.at[inner].set(3.0 * (h[1:] * delta[:-1] + h[:-1] * delta[1:]))
    if start is None:  # natural: f'' = 0, i.e. 2 m_0 + m_1 = 3 delta_0
        a = a.at[0, 0].set(2.0).at[0, 1].set(1.0)
        rhs = rhs.at[0].set(3.0 * delta[0])
    else:
        a = a.at[0, 0].set(1.0)
        rhs = rhs.at[0].set(jnp.asarray(start, dtype=jnp.float64))
    if end is None:
        a = a.at[n - 1, n - 2].set(1.0).at[n - 1, n - 1].set(2.0)
        rhs = rhs.at[n - 1].set(3.0 * delta[-1])
    else:
        a = a.at[n - 1, n - 1].set(1.0)
        rhs = rhs.at[n - 1].set(jnp.asarray(end, dtype=jnp.float64))
    return jnp.linalg.solve(a, rhs)


def _safe_div(num: jax.Array, den: jax.Array) -> jax.Array:
    """``num / den`` with 0 where ``den == 0`` (gradient-safe)."""
    ok = den != 0
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def _pchip_end_slope(h0: jax.Array, h1: jax.Array, d0: jax.Array, d1: jax.Array):
    """Shape-preserving three-point end slope (Fritsch-Carlson, as in SciPy)."""
    slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    slope = jnp.where(jnp.sign(slope) != jnp.sign(d0), 0.0, slope)
    clip = (jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(slope) > 3.0 * jnp.abs(d0))
    return jnp.where(clip, 3.0 * d0, slope)


def _pchip_slopes(values: jax.Array, times: jax.Array) -> jax.Array:
    """Knot slopes of the Fritsch-Carlson monotone (PCHIP) interpolant."""
    x = _knot_grid(times)
    h = x[1:] - x[:-1]
    delta = (values[1:] - values[:-1]) / h
    if delta.shape[0] == 1:
        return jnp.concatenate([delta, delta])
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    same_sign = delta[:-1] * delta[1:] > 0
    d_prev = jnp.where(same_sign, delta[:-1], 1.0)
    d_next = jnp.where(same_sign, delta[1:], 1.0)
    inner = jnp.where(same_sign, (w1 + w2) / (w1 / d_prev + w2 / d_next), 0.0)
    first = _pchip_end_slope(h[0], h[1], delta[0], delta[1])
    last = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return jnp.concatenate([first[None], inner, last[None]])


@dataclasses.dataclass(frozen=True)
class PiecewiseCubicRate:
    r"""Rate that is a cubic Hermite polynomial in time within each epoch.

    On epoch $k = (\tau_{k-1}, \tau_k]$ of duration $d_k$, with
    $u = (t - \tau_{k-1}) / d_k \in [0, 1]$, the rate is the cubic
    with the given values $v_0, v_1$ and slopes $m_0, m_1$ (derivatives
    with respect to time) at the two ends,

    $$
    w = v_0 h_{00}(u) + d_k m_0 h_{10}(u) + v_1 h_{01}(u) + d_k m_1 h_{11}(u).
    $$

    Setting the next epoch's start value or slope different from this epoch's
    end value or slope gives a jump or a kink;
    the constructors :meth:`hermite`, :meth:`natural_spline`,
    :meth:`clamped_spline` and :meth:`pchip` build continuous rates through
    knot values.

    Parameters
    ----------
    start_values, end_values : jax.Array
        Rate at the start and end of each epoch, shape ``(n_epochs,)``.
    start_slopes, end_slopes : jax.Array
        Time derivative of the rate at the start and end of each epoch.
    epoch_end_times : jax.Array
        Strictly increasing epoch end times, shape ``(n_epochs,)``.

    Notes
    -----
    Concrete inputs are checked for shape and finiteness only
    (and the check is skipped under tracing).
    Unlike :class:`PiecewiseQuadraticRate`, a rate that dips below zero is
    *not* rejected here:
    :meth:`min_values` reports the exact minimum of each epoch
    and the likelihood (:func:`~bdpinf.inference.likelihood.log_likelihood_rates`)
    returns ``-inf`` for a rate that is negative anywhere.
    The integral over a full epoch is exact,
    $d_k (v_0 + v_1) / 2 + d_k^2 (m_0 - m_1) / 12$.

    """

    start_values: jax.Array
    end_values: jax.Array
    start_slopes: jax.Array
    end_slopes: jax.Array
    epoch_end_times: jax.Array

    def __post_init__(self) -> None:
        """Coerce to float64 arrays and validate concrete inputs."""
        for name in (
            "start_values",
            "end_values",
            "start_slopes",
            "end_slopes",
            "epoch_end_times",
        ):
            object.__setattr__(self, name, _as_f64(getattr(self, name)))
        _validate_cubic(
            self.epoch_end_times,
            self.start_values,
            self.end_values,
            self.start_slopes,
            self.end_slopes,
        )

    @classmethod
    def hermite(
        cls,
        knot_values: jax.Array,
        knot_slopes: jax.Array,
        epoch_end_times: jax.Array,
    ) -> "PiecewiseCubicRate":
        """C1 rate from values and slopes at ``0`` and at each epoch end.

        Both arrays have length ``n_epochs + 1``.
        """
        v = jnp.asarray(knot_values, dtype=jnp.float64)
        m = jnp.asarray(knot_slopes, dtype=jnp.float64)
        return cls(v[:-1], v[1:], m[:-1], m[1:], jnp.asarray(epoch_end_times))

    @classmethod
    def natural_spline(
        cls, knot_values: jax.Array, epoch_end_times: jax.Array
    ) -> "PiecewiseCubicRate":
        """C2 natural cubic spline ($w''=0$ at the ends) through the knot values."""
        v = jnp.asarray(knot_values, dtype=jnp.float64)
        times = jnp.asarray(epoch_end_times, dtype=jnp.float64)
        return cls.hermite(v, _spline_slopes(v, times, None, None), times)

    @classmethod
    def clamped_spline(
        cls,
        knot_values: jax.Array,
        slope_start: float | jax.Array,
        slope_end: float | jax.Array,
        epoch_end_times: jax.Array,
    ) -> "PiecewiseCubicRate":
        """C2 cubic spline through the knot values with given end slopes.

        With the knot values and the two end slopes this spans the whole
        cubic spline space on these knots.
        """
        v = jnp.asarray(knot_values, dtype=jnp.float64)
        times = jnp.asarray(epoch_end_times, dtype=jnp.float64)
        slopes = _spline_slopes(
            v, times, jnp.asarray(slope_start), jnp.asarray(slope_end)
        )
        return cls.hermite(v, slopes, times)

    @classmethod
    def pchip(
        cls, knot_values: jax.Array, epoch_end_times: jax.Array
    ) -> "PiecewiseCubicRate":
        """C1 shape-preserving (Fritsch-Carlson) interpolant of the knot values.

        It never overshoots the knot values, so it is non-negative whenever
        they are.
        """
        v = jnp.asarray(knot_values, dtype=jnp.float64)
        times = jnp.asarray(epoch_end_times, dtype=jnp.float64)
        return cls.hermite(v, _pchip_slopes(v, times), times)

    @property
    def end_time(self) -> jax.Array:
        """Observation time $T$."""
        return self.epoch_end_times[-1]

    @property
    def breakpoints(self) -> jax.Array:
        """Interior epoch boundaries."""
        return self.epoch_end_times[:-1]

    def _local(self, t: jax.Array) -> tuple[jax.Array, ...]:
        """Epoch index, power-basis coefficients and offset ``u`` at ``t``."""
        times = self.epoch_end_times
        starts = _epoch_starts(times)
        k = epoch_index(times, t)
        dur = times[k] - starts[k]
        coef = _cubic_coefficients(
            self.start_values[k],
            self.end_values[k],
            self.start_slopes[k],
            self.end_slopes[k],
            dur,
        )
        return k, dur, (t - starts[k]) / dur, *coef

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        _, _, u, c0, c1, c2, c3 = self._local(jnp.asarray(t))
        return c0 + u * (c1 + u * (c2 + u * c3))

    def derivative(self, t: jax.Array) -> jax.Array:
        """Slope at ``t`` (the left limit at an epoch boundary)."""
        _, dur, u, _, c1, c2, c3 = self._local(jnp.asarray(t))
        return (c1 + u * (2.0 * c2 + 3.0 * u * c3)) / dur

    def flat_unless(self, ok: jax.Array) -> "PiecewiseCubicRate":
        """Return this rate where ``ok`` is true, else the constant rate 1.

        Lets a guard evaluate a harmless rate on infeasible parameters,
        so the gradient through ``jnp.where`` carries no NaN.
        """
        one = jnp.ones_like(self.start_values)
        zero = jnp.zeros_like(self.start_values)
        return PiecewiseCubicRate(
            jnp.where(ok, self.start_values, one),
            jnp.where(ok, self.end_values, one),
            jnp.where(ok, self.start_slopes, zero),
            jnp.where(ok, self.end_slopes, zero),
            self.epoch_end_times,
        )

    def local_coefficients(self) -> np.ndarray:
        r"""Host array ``(n_epochs, 4)`` of local-time coefficients $(a, b, c, d)$.

        Row $k$ is the epoch's polynomial $a + b s + c s^2 + d s^3$ in
        $s = t - \tau_{k-1}$, the form ``bdsurv`` takes for simulation.
        """
        times = np.asarray(self.epoch_end_times, dtype=np.float64)
        dur = times - np.concatenate([[0.0], times[:-1]])
        coef = _cubic_coefficients(
            self.start_values,
            self.end_values,
            self.start_slopes,
            self.end_slopes,
            jnp.asarray(dur),
        )
        c = np.stack([np.asarray(x, dtype=np.float64) for x in coef], axis=1)
        return c / dur[:, None] ** np.arange(4)

    def min_values(self) -> jax.Array:
        """Exact minimum of the rate on each epoch, shape ``(n_epochs,)``.

        Endpoint values and the values at the real roots of $w'$ inside the
        epoch (a quadratic).
        Jittable: no Python branching on traced values.
        """
        times = self.epoch_end_times
        dur = times - _epoch_starts(times)
        c0, c1, c2, c3 = _cubic_coefficients(
            self.start_values,
            self.end_values,
            self.start_slopes,
            self.end_slopes,
            dur,
        )
        # w'(u) = c1 + 2 c2 u + 3 c3 u^2 = qa u^2 + qb u + c1
        qa, qb = 3.0 * c3, 2.0 * c2
        disc = qb * qb - 4.0 * qa * c1
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
        q = -0.5 * (qb + jnp.where(qb >= 0, 1.0, -1.0) * sqrt_disc)
        scale = jnp.maximum(jnp.abs(qa), jnp.abs(qb))
        tiny = 1e-14 * jnp.maximum(scale, jnp.abs(c1)) + 1e-300
        roots = (
            jnp.where(jnp.abs(qa) > tiny, _safe_div(q, qa), jnp.nan),
            jnp.where(jnp.abs(q) > tiny, _safe_div(c1, q), jnp.nan),
        )
        lowest = jnp.minimum(self.start_values, self.end_values)
        for root in roots:
            inside = (disc >= 0) & (root > 0) & (root < 1)
            u = jnp.where(inside, root, 0.0)
            value = c0 + u * (c1 + u * (c2 + u * c3))
            lowest = jnp.minimum(lowest, jnp.where(inside, value, jnp.inf))
        return lowest

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$, as a difference of cumulative integrals."""
        return self._cumulative(jnp.asarray(e)) - self._cumulative(jnp.asarray(s))

    def _cumulative(self, t: jax.Array) -> jax.Array:
        times = self.epoch_end_times
        dur = times - _epoch_starts(times)
        full = (
            dur * 0.5 * (self.start_values + self.end_values)
            + dur * dur * (self.start_slopes - self.end_slopes) / 12.0
        )
        cum_before = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(full)[:-1]]
        )
        t = jnp.clip(t, 0.0, times[-1])
        k, dur_k, u, c0, c1, c2, c3 = self._local(t)
        poly = u * (c0 + u * (c1 / 2.0 + u * (c2 / 3.0 + u * c3 / 4.0)))
        return cum_before[k] + dur_k * poly


# ---------------------------------------------------------------------------
# Sinusoidal
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SinusoidalRate:
    r"""Sinusoidal rate $w(t) = m + A \sin(2\pi t / P + \varphi)$ on $[0, T]$.

    Parameters
    ----------
    mean : float or jax.Array
        Mean level $m$ (scalar, differentiable).
    amplitude : float or jax.Array
        Amplitude $A$ (scalar, differentiable); may be negative.
    period : float
        Period $P > 0$ (static: it fixes the number of breakpoints).
    phase : float or jax.Array
        Phase $\varphi$ in radians (scalar, differentiable), default 0.
    end_time : float
        Observation time $T > 0$ (static, keyword-only).

    Notes
    -----
    Concrete inputs are validated: all values finite, $P > 0$, $T > 0$,
    and $m \ge |A|$ so that the rate is non-negative.
    ``mean``, ``amplitude`` and ``phase`` are pytree data fields;
    ``period`` and ``end_time`` are static meta fields.

    The integral is exact,

    $$
    \int_s^e w = m\,(e - s)
      - \frac{A P}{2\pi}\left[\cos\!\Bigl(\frac{2\pi e}{P} + \varphi\Bigr)
      - \cos\!\Bigl(\frac{2\pi s}{P} + \varphi\Bigr)\right].
    $$

    The sinusoid is smooth, but the quadrature in
    :mod:`bdpinf.models.math_general` uses one Gauss-Legendre panel per segment
    between breakpoints, and a single panel cannot resolve several oscillations.
    ``breakpoints`` therefore returns the interior multiples of half a period,
    $kP/2$ for integers $k$ with $0 < kP/2 < T$,
    so that each panel sees at most half an oscillation.

    """

    mean: jax.Array | float
    amplitude: jax.Array | float
    period: float
    phase: jax.Array | float = 0.0
    end_time: float = dataclasses.field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Coerce, and validate concrete inputs."""
        for name in ("mean", "amplitude", "phase"):
            object.__setattr__(self, name, _as_f64(getattr(self, name)))
        object.__setattr__(self, "period", float(self.period))
        if not (np.isfinite(self.period) and self.period > 0):
            raise ValueError("period must be finite and positive")
        object.__setattr__(self, "end_time", _check_end_time(self.end_time))
        data = (self.mean, self.amplitude, self.phase)
        if not is_concrete(*data):
            return
        mean, amplitude, phase = (np.asarray(x) for x in data)
        if not all(np.all(np.isfinite(x)) for x in (mean, amplitude, phase)):
            raise ValueError("mean, amplitude and phase must be finite")
        if np.any(mean < np.abs(amplitude)):
            raise ValueError(
                f"mean must be at least |amplitude| for a non-negative rate; "
                f"got mean={float(np.min(mean)):.6g}, "
                f"|amplitude|={float(np.max(np.abs(amplitude))):.6g}"
            )

    @property
    def breakpoints(self) -> jax.Array:
        """Interior multiples of half a period, one half-oscillation per panel."""
        half = 0.5 * self.period
        k = np.arange(1, int(np.floor(self.end_time / half)) + 1)
        times = k * half
        return jnp.asarray(times[times < self.end_time], dtype=jnp.float64)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        t = jnp.asarray(t)
        return self.mean + self.amplitude * jnp.sin(
            2.0 * jnp.pi * t / self.period + self.phase
        )

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$ (closed form in the class docstring)."""
        s, e = jnp.asarray(s), jnp.asarray(e)
        omega = 2.0 * jnp.pi / self.period
        return self.mean * (e - s) - self.amplitude / omega * (
            jnp.cos(omega * e + self.phase) - jnp.cos(omega * s + self.phase)
        )


# ---------------------------------------------------------------------------
# Parametric hazards: power (Weibull, Rayleigh), log-logistic, Lomax
# ---------------------------------------------------------------------------

_N_GEOMETRIC = 30


def _geometric_breakpoints(end_time: float) -> jax.Array:
    r"""Ascending $T 2^{-j}$ for $j = 30, \dots, 1$.

    Geometric grading towards 0 lets one Gauss-Legendre panel per segment
    resolve an integrable singularity of the rate at 0.
    """
    j = np.arange(_N_GEOMETRIC, 0, -1)
    return jnp.asarray(float(end_time) * 0.5**j, dtype=jnp.float64)


def _check_end_time(end_time: object) -> np.float64:
    end = np.float64(cast("float", end_time))
    if not (np.isfinite(end) and end > 0):
        raise ValueError("end_time must be finite and positive")
    return end


def _check_finite(**named: object) -> None:
    """Concrete values only: every entry finite."""
    for name, v in named.items():
        if is_concrete_value(v) and not np.all(np.isfinite(np.asarray(v))):
            raise ValueError(f"{name} must be finite")


def _check_positive(**named: object) -> None:
    """Concrete values only: every entry finite and strictly positive."""
    for name, v in named.items():
        if is_concrete_value(v):
            arr = np.asarray(v)
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError(f"{name} must be positive and finite")


def _singular_at_zero(shape: object) -> bool:
    """Whether a rate with this exponent-like parameter is singular at 0.

    Unknown (traced) values are treated as singular, which is always safe.
    """
    if not is_concrete_value(shape):
        return True
    return bool(np.any(np.asarray(shape) < 1.0))


@dataclasses.dataclass(frozen=True)
class PowerRate:
    r"""Power-law rate $w(t) = c\,t^p$ on $[0, T]$, with $c > 0$ and $p > -1$.

    The Weibull hazard with shape $k$ and scale $\eta$ is the case
    $c = k/\eta^k$, $p = k - 1$ (:meth:`weibull`),
    and the Rayleigh hazard is $p = 1$ (:meth:`rayleigh`).

    Parameters
    ----------
    c : float or jax.Array
        Coefficient $c > 0$ (scalar, differentiable).
    p : float or jax.Array
        Exponent $p > -1$ (scalar, differentiable).
    end_time : float
        Observation time $T > 0$ (static, keyword-only).

    Notes
    -----
    Concrete inputs are validated; validation is skipped under tracing.
    ``c`` and ``p`` are pytree data fields and ``end_time`` is a static meta field.

    The integral is $\frac{c}{p+1}(e^{p+1} - s^{p+1})$,
    evaluated as $s^{p+1}\,\mathrm{expm1}((p+1)\ln(e/s))$ for $s > 0$
    so that it stays accurate when $e$ is close to $s$.

    For $p < 0$ the rate is singular at 0 (integrably).
    ``breakpoints`` then returns the geometric times $T 2^{-j}$, $j = 1, \dots, 30$,
    so that the single Gauss-Legendre panel per segment in
    :mod:`bdpinf.models.math_general` resolves the singularity;
    if ``p`` is traced it is not known to be non-negative,
    and the geometric breakpoints are returned as well.
    Otherwise there are no breakpoints.

    """

    c: jax.Array | float
    p: jax.Array | float
    end_time: float = dataclasses.field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Coerce, and validate concrete inputs."""
        object.__setattr__(self, "c", _as_f64(self.c))
        object.__setattr__(self, "p", _as_f64(self.p))
        object.__setattr__(self, "end_time", _check_end_time(self.end_time))
        _check_positive(c=self.c)
        _check_finite(p=self.p)
        if is_concrete_value(self.p) and np.any(np.asarray(self.p) <= -1.0):
            raise ValueError("p must be greater than -1")

    @classmethod
    def weibull(
        cls, shape: float | jax.Array, scale: float | jax.Array, *, end_time: float
    ) -> "PowerRate":
        r"""Weibull hazard $(k/\eta)(t/\eta)^{k-1}$ with $c = k/\eta^k$, $p = k - 1$."""
        k = jnp.asarray(shape, dtype=jnp.float64)
        eta = jnp.asarray(scale, dtype=jnp.float64)
        return cls(k / eta**k, k - 1.0, end_time=end_time)

    @classmethod
    def rayleigh(cls, c: float | jax.Array, *, end_time: float) -> "PowerRate":
        r"""Rayleigh hazard $c\,t$ (the case $p = 1$)."""
        return cls(c, 1.0, end_time=end_time)

    @property
    def breakpoints(self) -> jax.Array:
        """Geometric times $T 2^{-j}$ when singular at 0 ($p < 0$), else empty."""
        if is_concrete_value(self.p) and not np.any(np.asarray(self.p) < 0.0):
            return jnp.zeros(0, dtype=jnp.float64)
        return _geometric_breakpoints(self.end_time)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        return self.c * jnp.asarray(t) ** self.p

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$ (stable form in the class docstring)."""
        s, e = jnp.asarray(s), jnp.asarray(e)
        k = self.p + 1.0
        s_pos, e_pos = s > 0, e > 0
        s_safe = jnp.where(s_pos, s, 1.0)
        e_safe = jnp.where(e_pos, e, 1.0)
        diff = jnp.where(
            s_pos,
            s_safe**k * jnp.expm1(k * (jnp.log(e_safe) - jnp.log(s_safe))),
            jnp.where(e_pos, e_safe**k, 0.0),
        )
        return self.c / k * diff


@dataclasses.dataclass(frozen=True)
class LogLogisticRate:
    r"""Log-logistic hazard on $[0, T]$.

    $$
    w(t) = \frac{(k/\eta)(t/\eta)^{k-1}}{1 + (t/\eta)^k}.
    $$

    Parameters
    ----------
    shape : float or jax.Array
        Shape $k > 0$ (scalar, differentiable).
    scale : float or jax.Array
        Scale $\eta > 0$ (scalar, differentiable).
    end_time : float
        Observation time $T > 0$ (static, keyword-only).

    Notes
    -----
    Concrete inputs are validated; validation is skipped under tracing.
    ``shape`` and ``scale`` are data fields and ``end_time`` is a static meta field.

    The cumulative hazard is $\ln(1 + x)$ with $x = (t/\eta)^k$,
    so the integral is $\mathrm{log1p}((x_e - x_s)/(1 + x_s))$.
    For $k < 1$ the rate is singular at 0,
    and ``breakpoints`` returns the geometric times $T 2^{-j}$,
    as for :class:`PowerRate`.

    """

    shape: jax.Array | float
    scale: jax.Array | float
    end_time: float = dataclasses.field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Coerce, and validate concrete inputs."""
        object.__setattr__(self, "shape", _as_f64(self.shape))
        object.__setattr__(self, "scale", _as_f64(self.scale))
        object.__setattr__(self, "end_time", _check_end_time(self.end_time))
        _check_positive(shape=self.shape, scale=self.scale)

    @property
    def breakpoints(self) -> jax.Array:
        """Geometric times $T 2^{-j}$ when singular at 0 ($k < 1$), else empty."""
        if _singular_at_zero(self.shape):
            return _geometric_breakpoints(self.end_time)
        return jnp.zeros(0, dtype=jnp.float64)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        z = jnp.asarray(t) / self.scale
        return self.shape / self.scale * z ** (self.shape - 1.0) / (1.0 + z**self.shape)

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$ (stable form in the class docstring)."""
        x_s = self._scaled_power(jnp.asarray(s))
        x_e = self._scaled_power(jnp.asarray(e))
        return jnp.log1p((x_e - x_s) / (1.0 + x_s))

    def _scaled_power(self, t: jax.Array) -> jax.Array:
        r"""$(t / \text{scale})^{\text{shape}}$, zero at $t = 0$ with finite gradients.

        For ``shape < 1`` the power has an infinite derivative at ``t = 0``, so
        the direct form gives ``inf * 0 = nan`` in gradients with respect to
        ``scale`` and ``shape``; evaluating through the logarithm on ``t > 0``
        only keeps them finite.
        """
        positive = t > 0
        t_safe = jnp.where(positive, t, 1.0)
        return jnp.where(
            positive,
            jnp.exp(self.shape * (jnp.log(t_safe) - jnp.log(self.scale))),
            0.0,
        )


@dataclasses.dataclass(frozen=True)
class LomaxRate:
    r"""Lomax (Pareto II) hazard $w(t) = k/(\eta + t)$ on $[0, T]$.

    Parameters
    ----------
    shape : float or jax.Array
        Shape $k > 0$ (scalar, differentiable).
    scale : float or jax.Array
        Scale $\eta > 0$ (scalar, differentiable).
    end_time : float
        Observation time $T > 0$ (static, keyword-only).

    Notes
    -----
    Concrete inputs are validated; validation is skipped under tracing.
    The rate is smooth, so there are no breakpoints.
    The integral is $k\,\mathrm{log1p}((e - s)/(\eta + s))$.

    """

    shape: jax.Array | float
    scale: jax.Array | float
    end_time: float = dataclasses.field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Coerce, and validate concrete inputs."""
        object.__setattr__(self, "shape", _as_f64(self.shape))
        object.__setattr__(self, "scale", _as_f64(self.scale))
        object.__setattr__(self, "end_time", _check_end_time(self.end_time))
        _check_positive(shape=self.shape, scale=self.scale)

    @property
    def breakpoints(self) -> jax.Array:
        """No interior breakpoints (the rate is smooth)."""
        return jnp.zeros(0, dtype=jnp.float64)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        return self.shape / (self.scale + jnp.asarray(t))

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$."""
        s, e = jnp.asarray(s), jnp.asarray(e)
        return self.shape * jnp.log1p((e - s) / (self.scale + s))


# ---------------------------------------------------------------------------
# Piecewise composition of arbitrary rate functions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PiecewiseRate:
    r"""Glue rate functions together epoch by epoch, each in local time.

    Piece $k$ applies on $(\tau_{k-1}, \tau_k]$ and is evaluated at the
    *local* time $u = t - \tau_{k-1}$, so each piece is defined on $[0, d_k]$
    with $d_k = \tau_k - \tau_{k-1}$ and need not know where its epoch sits.

    Parameters
    ----------
    pieces : tuple of RateFunction
        One rate function per epoch (a static-length tuple of pytrees).
        The ``end_time`` of piece $k$ must equal $d_k$.
    epoch_end_times : jax.Array
        Strictly increasing epoch end times, shape ``(n_epochs,)``.
        The last entry is the observation time $T$.

    Notes
    -----
    Concrete inputs are validated, including that each piece's ``end_time``
    matches its epoch length; validation is skipped under tracing.
    Values may jump at epoch boundaries.
    ``pieces`` and ``epoch_end_times`` are pytree data fields.

    The integral to time $t$ in epoch $k$ is the sum of the full-epoch integrals
    before it plus ``pieces[k].integral(0, u)``.
    ``breakpoints`` are the interior epoch ends together with each piece's own
    breakpoints shifted by its epoch start, sorted.

    Every piece is evaluated at every time and the right one selected
    (at a harmless in-domain time for the others),
    so this works under ``jit`` and ``grad`` at a cost proportional to the
    number of epochs.

    """

    pieces: tuple[RateFunction, ...]
    epoch_end_times: jax.Array

    def __post_init__(self) -> None:
        """Coerce, and validate concrete inputs."""
        pieces: object = self.pieces
        if isinstance(pieces, list):
            pieces = tuple(pieces)
            object.__setattr__(self, "pieces", pieces)
        object.__setattr__(self, "epoch_end_times", _as_f64(self.epoch_end_times))
        times = self.epoch_end_times
        if not isinstance(pieces, tuple) or not is_concrete(times):  # pyright: ignore[reportUnnecessaryIsInstance]  # pieces may be a non-sequence at runtime; skip validation and let later code reject it
            return
        _validate_epochs(times)
        if len(pieces) != np.asarray(times).shape[0]:
            raise ValueError(
                f"got {len(pieces)} pieces for {np.asarray(times).shape[0]} epochs"
            )
        lengths = np.diff(np.asarray(times), prepend=0.0)
        for k, (piece, length) in enumerate(zip(pieces, lengths, strict=True)):
            end = getattr(piece, "end_time", None)
            if is_concrete_value(end) and not np.isclose(
                float(np.asarray(end)), length, rtol=1e-9, atol=0.0
            ):
                raise ValueError(
                    f"piece {k} has end_time {float(np.asarray(end)):.12g}, "
                    f"but its epoch has length {length:.12g}"
                )

    @property
    def end_time(self) -> jax.Array:
        """Observation time $T$."""
        return self.epoch_end_times[-1]

    @property
    def breakpoints(self) -> jax.Array:
        """Interior epoch ends and the pieces' own breakpoints, sorted."""
        times = self.epoch_end_times
        starts = _epoch_starts(times)
        parts = [times[:-1]]
        parts.extend(
            starts[k] + jnp.asarray(piece.breakpoints, dtype=jnp.float64)
            for k, piece in enumerate(self.pieces)
        )
        return jnp.sort(jnp.concatenate(parts))

    def _local(self, t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Epoch index, local time, and epoch lengths."""
        times = self.epoch_end_times
        starts = _epoch_starts(times)
        k = epoch_index(times, t)
        return k, t - starts[k], times - starts

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t`` (each piece in its local time)."""
        t = jnp.asarray(t, dtype=jnp.float64)
        k, u, dur = self._local(t)
        out = jnp.zeros_like(u)
        for j, piece in enumerate(self.pieces):
            here = k == j
            out = jnp.where(here, piece(jnp.where(here, u, 0.5 * dur[j])), out)
        return out

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact $\int_s^e w$, as a difference of cumulative integrals."""
        return self._cumulative(jnp.asarray(e)) - self._cumulative(jnp.asarray(s))

    def _cumulative(self, t: jax.Array) -> jax.Array:
        times = self.epoch_end_times
        t = jnp.clip(jnp.asarray(t, dtype=jnp.float64), 0.0, times[-1])
        k, u, dur = self._local(t)
        full = jnp.stack(
            [
                piece.integral(jnp.zeros(()), dur[j])
                for j, piece in enumerate(self.pieces)
            ]
        )
        cum_before = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(full)[:-1]]
        )
        out = jnp.zeros_like(u)
        for j, piece in enumerate(self.pieces):
            here = k == j
            u_j = jnp.where(here, u, 0.5 * dur[j])
            out = jnp.where(here, piece.integral(jnp.zeros_like(u_j), u_j), out)
        return cum_before[k] + out


jax.tree_util.register_dataclass(
    PiecewiseConstantRate, data_fields=["values", "epoch_end_times"], meta_fields=[]
)
jax.tree_util.register_dataclass(
    PiecewiseLinearRate,
    data_fields=["start_values", "end_values", "epoch_end_times"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    PiecewiseExponentialRate,
    data_fields=["start_values", "end_values", "epoch_end_times"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    PiecewiseQuadraticRate,
    data_fields=["start_values", "mid_values", "end_values", "epoch_end_times"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    PiecewiseCubicRate,
    data_fields=[
        "start_values",
        "end_values",
        "start_slopes",
        "end_slopes",
        "epoch_end_times",
    ],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    SinusoidalRate,
    data_fields=["mean", "amplitude", "phase"],
    meta_fields=["period", "end_time"],
)
jax.tree_util.register_dataclass(
    PowerRate, data_fields=["c", "p"], meta_fields=["end_time"]
)
jax.tree_util.register_dataclass(
    LogLogisticRate, data_fields=["shape", "scale"], meta_fields=["end_time"]
)
jax.tree_util.register_dataclass(
    LomaxRate, data_fields=["shape", "scale"], meta_fields=["end_time"]
)
jax.tree_util.register_dataclass(
    PiecewiseRate, data_fields=["pieces", "epoch_end_times"], meta_fields=[]
)
