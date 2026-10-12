r"""Time-varying birth and death rate functions.

A *rate function* maps calendar time :math:`t \in [0, T]` to a non-negative
rate.
The age densities in :mod:`bdpinf.models.math_age_densities` need only three
things from a rate:

- its value :math:`w(t)` (``__call__``);
- its exact integral :math:`\int_s^e w(t)\,dt` (``integral``);
- the interior times at which it is not smooth (``breakpoints``),
  so that numerical quadrature can be split there.

Any class providing these (the :class:`RateFunction` protocol) can be used,
which leaves room for sinusoidal, spline or rational rates.
Two families are provided: :class:`PiecewiseConstantRate` and
:class:`PiecewiseLinearRate`.
Both are JAX pytrees, so they can be passed through ``jit``/``grad``.
"""

import dataclasses
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "PiecewiseConstantRate",
    "PiecewiseLinearRate",
    "RateFunction",
]


@runtime_checkable
class RateFunction(Protocol):
    r"""Protocol for a non-negative rate :math:`w(t)` on :math:`[0, T]`."""

    @property
    def end_time(self) -> jax.Array:
        """Observation time :math:`T` (scalar)."""
        ...

    @property
    def breakpoints(self) -> jax.Array:
        """Increasing interior times where the rate is not smooth, shape ``(k,)``."""
        ...

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at times ``t`` (any shape)."""
        ...

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact :math:`\int_s^e w(t)\,dt` for ``0 <= s <= e <= T``."""
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
    r"""Cumulative integral :math:`\int_0^t w`, with ``t`` clipped to ``[0, T]``."""
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


def _validate_epochs(times: object, *values: object) -> None:
    """Value checks; skipped when any input is traced or not a concrete array."""
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
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
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
        The last entry is the observation time :math:`T`.

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
        """Observation time :math:`T`."""
        return self.epoch_end_times[-1]

    @property
    def breakpoints(self) -> jax.Array:
        """Interior epoch boundaries."""
        return self.epoch_end_times[:-1]

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluate the rate at ``t``."""
        return self.values[_epoch_index(self.epoch_end_times, jnp.asarray(t))]

    def integral(self, s: jax.Array, e: jax.Array) -> jax.Array:
        r"""Exact :math:`\int_s^e w`."""
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

    On epoch :math:`k = (\tau_{k-1}, \tau_k]` the rate interpolates linearly
    from ``start_values[k]`` at :math:`\tau_{k-1}` to ``end_values[k]`` at
    :math:`\tau_k`.
    Setting ``start_values[k+1] != end_values[k]`` gives a jump at
    :math:`\tau_k`; :meth:`continuous` builds the jump-free case.

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
        """Observation time :math:`T`."""
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
        r"""Exact :math:`\int_s^e w`."""
        args = (self.start_values, self.end_values, self.epoch_end_times)
        return _linear_cumulative(*args, jnp.asarray(e)) - _linear_cumulative(
            *args, jnp.asarray(s)
        )


jax.tree_util.register_dataclass(
    PiecewiseConstantRate, data_fields=["values", "epoch_end_times"], meta_fields=[]
)
jax.tree_util.register_dataclass(
    PiecewiseLinearRate,
    data_fields=["start_values", "end_values", "epoch_end_times"],
    meta_fields=[],
)
