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
Five families are provided: :class:`PiecewiseConstantRate`,
:class:`PiecewiseLinearRate`, :class:`PiecewiseExponentialRate`
(linear on the log scale), :class:`PiecewiseQuadraticRate`
(including C1 quadratic splines) and :class:`SinusoidalRate`.
All are JAX pytrees, so they can be passed through ``jit``/``grad``.
"""

import dataclasses
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "PiecewiseConstantRate",
    "PiecewiseExponentialRate",
    "PiecewiseLinearRate",
    "PiecewiseQuadraticRate",
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


def _is_concrete_array(x: object) -> bool:
    return isinstance(x, np.ndarray | jax.Array) and not isinstance(x, jax.core.Tracer)


def _as_f64(x: object) -> object:
    """Coerce lists, scalars and arrays to float64; leave tracers/sentinels alone."""
    if isinstance(x, jax.core.Tracer):
        return x
    if isinstance(x, list | tuple | float | int | np.ndarray | jax.Array):
        return jnp.asarray(x, dtype=jnp.float64)
    return x


def _epoch_starts(times: jax.Array) -> jax.Array:
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), times[:-1]])


def _epoch_index(times: jax.Array, t: jax.Array) -> jax.Array:
    return jnp.clip(jnp.searchsorted(times, t, side="left"), 0, times.shape[0] - 1)


def _linear_value(
    v0: jax.Array, v1: jax.Array, times: jax.Array, t: jax.Array
) -> jax.Array:
    starts = _epoch_starts(times)
    slope = (v1 - v0) / (times - starts)
    k = _epoch_index(times, t)
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
    k = _epoch_index(times, t)
    d = t - starts[k]
    return cum_before[k] + v0[k] * d + 0.5 * slope[k] * d * d


def _validate_epochs(times: object, *values: object, positive: bool = False) -> None:
    """Value checks; skipped when any input is traced or not a concrete array.

    With ``positive=True`` the values must be strictly positive, not just non-negative.
    """
    arrays = (times, *values)
    if not all(_is_concrete_array(x) for x in arrays):
        return
    t = np.asarray(times)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("epoch_end_times must be a non-empty 1-D array")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("epoch_end_times must be positive and strictly increasing")
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
        return self.values[_epoch_index(self.epoch_end_times, jnp.asarray(t))]

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
        k = _epoch_index(times, t)
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
        k = _epoch_index(times, t)
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


def _validate_quadratic(times: object, start: object, mid: object, end: object) -> None:
    """Finite, and non-negative everywhere on every epoch (vertex included)."""
    if not all(_is_concrete_array(x) for x in (times, start, mid, end)):
        return
    t = np.asarray(times)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("epoch_end_times must be a non-empty 1-D array")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("epoch_end_times must be positive and strictly increasing")
    v0, vm, v1 = (np.asarray(x, dtype=float) for x in (start, mid, end))
    for v in (v0, vm, v1):
        if v.shape != t.shape:
            raise ValueError(f"rate values shape {v.shape} != times shape {t.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("rate values must be finite")
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
    Concrete inputs are validated (and validation is skipped under tracing):
    the values must be finite and the quadratic non-negative on every epoch,
    which is checked at the three stored values and, when the vertex lies
    inside the epoch, at the vertex.
    A rounding allowance of $10^{-12}$ times the largest stored value is made.
    The error message names the offending epoch.

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
            self.epoch_end_times, self.start_values, self.mid_values, self.end_values
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
        return cls(
            knots[:-1],
            jnp.asarray(mid_values, dtype=jnp.float64),
            knots[1:],
            jnp.asarray(epoch_end_times),
        )

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
        Raises ``ValueError`` if the resulting rate dips below zero.
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
        return cls(v0, 0.5 * (v0 + v1), v1, jnp.asarray(epoch_end_times))

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
        k = _epoch_index(times, t)
        a0, a1, a2 = _quadratic_coefficients(
            self.start_values[k], self.mid_values[k], self.end_values[k]
        )
        u = (t - starts[k]) / (times[k] - starts[k])
        return a0 + u * (a1 + u * a2)

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
        k = _epoch_index(times, t)
        a0, a1, a2 = _quadratic_coefficients(
            self.start_values[k], self.mid_values[k], self.end_values[k]
        )
        u = (t - starts[k]) / dur[k]
        return cum_before[k] + dur[k] * u * (a0 + u * (a1 / 2.0 + u * a2 / 3.0))


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
        object.__setattr__(self, "end_time", np.float64(self.end_time))
        if not (np.isfinite(self.period) and self.period > 0):
            raise ValueError("period must be finite and positive")
        if not (np.isfinite(self.end_time) and self.end_time > 0):
            raise ValueError("end_time must be finite and positive")
        data = (self.mean, self.amplitude, self.phase)
        if not all(_is_concrete_array(x) for x in data):
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
    SinusoidalRate,
    data_fields=["mean", "amplitude", "phase"],
    meta_fields=["period", "end_time"],
)
