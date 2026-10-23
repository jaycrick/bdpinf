r"""Parametric families for the birth and death rates that can be fitted.

A *rate model* maps a vector of natural-scale parameters to a
:class:`~bdpinf.models.rate_functions.RateFunction`.
Each parameter carries a transform (:class:`Param`) that sends it to an
unconstrained scale for optimisation, so non-negativity of a rate (or a
proportion staying in $(0, 1)$) is enforced by construction.

Provided families:

- :class:`PiecewiseConstant`: one rate per epoch;
- :class:`PiecewiseLinear`: values at epoch boundaries (``continuous=True``,
  the default) or separate start and end values per epoch, which allows jumps.
  A jump-form linear model whose start and end values are tied
  (``flat_groups``) is the piecewise-constant model.

:class:`BirthDeathModel` pairs a birth and a death family and owns the full
parameter list.
A new family (for example a sinusoid with amplitude, period and phase
parameters) only needs to satisfy the :class:`RateModel` protocol.
"""

import dataclasses
import itertools
from collections.abc import Sequence
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from bdpinf.models.rate_functions import (
    PiecewiseConstantRate,
    PiecewiseLinearRate,
    RateFunction,
)

__all__ = [
    "BirthDeathModel",
    "Param",
    "PiecewiseConstant",
    "PiecewiseLinear",
    "RateModel",
    "Transform",
]

Transform = Literal["log", "identity", "logit"]
"""How a natural-scale parameter maps to the unconstrained optimisation scale."""

_EDGE_FRACTION = 1e-9

EpochTimes = Sequence[float] | np.ndarray | jax.Array
"""Epoch end times as given by the caller (stored as a tuple of floats)."""


@dataclasses.dataclass(frozen=True)
class Param:
    """A named scalar parameter and its transform."""

    name: str
    transform: Transform = "log"


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

    def build(self, values: jax.Array) -> RateFunction:
        """Return the rate function for natural-scale ``values``."""
        ...

    def constant_values(self, value: float) -> jax.Array:
        """Parameter values for the constant rate ``value``."""
        ...

    def values_from_rate(self, rate: RateFunction) -> jax.Array:
        """Parameter values that approximate ``rate`` (to warm-start a fit)."""
        ...


def _as_epoch_tuple(epoch_end_times: EpochTimes) -> tuple[float, ...]:
    times = tuple(float(t) for t in np.atleast_1d(np.asarray(epoch_end_times)))
    if not times or times[0] <= 0 or any(b <= a for a, b in itertools.pairwise(times)):
        msg = "epoch_end_times must be positive and strictly increasing"
        raise ValueError(msg)
    return times


@dataclasses.dataclass(frozen=True)
class PiecewiseConstant:
    """One non-negative rate per epoch.

    Parameters
    ----------
    epoch_end_times : sequence of float
        Strictly increasing epoch end times; the last is the observation time.

    """

    epoch_end_times: EpochTimes

    def __post_init__(self) -> None:
        """Validate and freeze the epoch end times."""
        object.__setattr__(
            self, "epoch_end_times", _as_epoch_tuple(self.epoch_end_times)
        )

    @property
    def params(self) -> tuple[Param, ...]:
        """One log-scale parameter ``v{k}`` per epoch."""
        return tuple(Param(f"v{k}") for k in range(len(self.epoch_end_times)))

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        return float(self.epoch_end_times[-1])

    def build(self, values: jax.Array) -> PiecewiseConstantRate:
        """Return the piecewise-constant rate with the given epoch values."""
        return PiecewiseConstantRate(values, jnp.asarray(self.epoch_end_times))

    def constant_values(self, value: float) -> jax.Array:
        """Return the same rate in every epoch."""
        return jnp.full(len(self.epoch_end_times), value, dtype=jnp.float64)

    def values_from_rate(self, rate: RateFunction) -> jax.Array:
        """Evaluate ``rate`` at the epoch midpoints."""
        ends = jnp.asarray(self.epoch_end_times)
        starts = jnp.concatenate([jnp.zeros(1), ends[:-1]])
        return jnp.asarray(rate(0.5 * (starts + ends)), dtype=jnp.float64)


@dataclasses.dataclass(frozen=True)
class PiecewiseLinear:
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

    """

    epoch_end_times: EpochTimes
    continuous: bool = True

    def __post_init__(self) -> None:
        """Validate and freeze the epoch end times."""
        object.__setattr__(
            self, "epoch_end_times", _as_epoch_tuple(self.epoch_end_times)
        )

    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return len(self.epoch_end_times)

    @property
    def params(self) -> tuple[Param, ...]:
        """Log-scale knot values, or start and end values per epoch."""
        if self.continuous:
            return tuple(Param(f"knot{k}") for k in range(self.n_epochs + 1))
        return tuple(Param(f"start{k}") for k in range(self.n_epochs)) + tuple(
            Param(f"end{k}") for k in range(self.n_epochs)
        )

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        return float(self.epoch_end_times[-1])

    def build(self, values: jax.Array) -> PiecewiseLinearRate:
        """Return the piecewise-linear rate with the given parameter values."""
        times = jnp.asarray(self.epoch_end_times)
        if self.continuous:
            return PiecewiseLinearRate.continuous(values, times)
        return PiecewiseLinearRate(
            values[: self.n_epochs], values[self.n_epochs :], times
        )

    def constant_values(self, value: float) -> jax.Array:
        """Return a flat rate."""
        n = self.n_epochs + 1 if self.continuous else 2 * self.n_epochs
        return jnp.full(n, value, dtype=jnp.float64)

    def values_from_rate(self, rate: RateFunction) -> jax.Array:
        """Evaluate ``rate`` at the knots (averaging the limits either side)."""
        ends = jnp.asarray(self.epoch_end_times)
        end = ends[-1]
        eps = _EDGE_FRACTION * end
        if self.continuous:
            knots = jnp.concatenate([jnp.zeros(1), ends])
            left = rate(jnp.clip(knots - eps, 0.0, end))
            right = rate(jnp.clip(knots + eps, 0.0, end))
            return jnp.asarray(0.5 * (left + right), dtype=jnp.float64)
        starts = jnp.concatenate([jnp.zeros(1), ends[:-1]])
        return jnp.asarray(
            jnp.concatenate([rate(starts + eps), rate(ends - eps)]), dtype=jnp.float64
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


@dataclasses.dataclass(frozen=True)
class BirthDeathModel:
    """A birth-rate family paired with a death-rate family.

    Parameters are named ``birth.<name>`` and ``death.<name>`` and listed birth
    first.
    """

    birth: RateModel
    death: RateModel

    def __post_init__(self) -> None:
        """Check the two families share an observation window."""
        if not np.isclose(self.birth.end_time, self.death.end_time):
            msg = "birth and death models must share the observation time"
            raise ValueError(msg)

    @classmethod
    def piecewise_constant(cls, epoch_end_times: EpochTimes) -> "BirthDeathModel":
        """Piecewise-constant birth and death rates on shared epochs."""
        return cls(
            PiecewiseConstant(epoch_end_times), PiecewiseConstant(epoch_end_times)
        )

    @classmethod
    def piecewise_linear(
        cls,
        epoch_end_times: EpochTimes,
        *,
        continuous: bool = True,
    ) -> "BirthDeathModel":
        """Piecewise-linear birth and death rates on shared epochs."""
        return cls(
            PiecewiseLinear(epoch_end_times, continuous),
            PiecewiseLinear(epoch_end_times, continuous),
        )

    @property
    def end_time(self) -> float:
        """Observation time ``T``."""
        return self.birth.end_time

    @property
    def n_birth(self) -> int:
        """Number of birth parameters."""
        return len(self.birth.params)

    @property
    def params(self) -> tuple[Param, ...]:
        """All parameters, birth first, with ``birth.``/``death.`` name prefixes."""
        return tuple(
            Param(f"birth.{p.name}", p.transform) for p in self.birth.params
        ) + tuple(Param(f"death.{p.name}", p.transform) for p in self.death.params)

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

    def rates(self, natural: jax.Array) -> tuple[RateFunction, RateFunction]:
        """Return the ``(birth, death)`` rate functions for natural parameters."""
        return (
            self.birth.build(natural[: self.n_birth]),
            self.death.build(natural[self.n_birth :]),
        )

    def constant_natural(self, birth: float, death: float) -> jax.Array:
        """Natural parameters for constant ``birth`` and ``death`` rates."""
        return jnp.concatenate(
            [self.birth.constant_values(birth), self.death.constant_values(death)]
        )

    def natural_from_rates(self, birth: RateFunction, death: RateFunction) -> jax.Array:
        """Natural parameters approximating the given rate functions."""
        return jnp.concatenate(
            [self.birth.values_from_rate(birth), self.death.values_from_rate(death)]
        )

    def swapped(self, natural: jax.Array) -> jax.Array | None:
        """Return ``natural`` with birth and death exchanged, if the layouts match."""
        if self.birth.params != self.death.params:
            return None
        return jnp.concatenate([natural[self.n_birth :], natural[: self.n_birth]])
