"""Constraints on fitted parameters: fixed values and equality groups.

Constraints shrink the optimised parameter space (which can help convergence);
the fitted result is always reported in the full parameter space.
Fixed parameters stay in the likelihood but are not optimised, and parameters
in one equality group share a single fitted value.
Parameters are referred to by name (see
:attr:`bdpinf.inference.rate_models.BirthDeathModel.names`).
"""

import dataclasses
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from bdpinf.inference.rate_models import Param

__all__ = ["CompiledConstraints", "Constraints"]

_MIN_GROUP_SIZE = 2
_CODES = {"identity": 0, "log": 1, "logit": 2}


@dataclasses.dataclass(frozen=True)
class Constraints:
    """Fixed parameters and equality groups, keyed by parameter name.

    Attributes
    ----------
    fixed : Mapping[str, float]
        Parameter name to fixed natural-scale value.
        A fixed zero is allowed for log-scale (rate) parameters.
    equal : Sequence[Sequence[str]]
        Groups of at least two parameter names that must be equal.
        A parameter may not appear in a group and in ``fixed``.

    Examples
    --------
    Pool the first two epochs' birth rates and fix the last death rate:

    >>> constraints = Constraints(
    ...     fixed={"death.v2": 0.05},
    ...     equal=[["birth.v0", "birth.v1"]],
    ... )

    Time-invariant rates over four epochs (two free parameters):

    >>> from bdpinf.inference.rate_models import BirthDeathModel
    >>> model = BirthDeathModel.piecewise_constant([1.0, 2.0, 3.0, 4.0])
    >>> constant = Constraints(equal=[model.birth_names, model.death_names])

    """

    fixed: Mapping[str, float] = dataclasses.field(default_factory=dict)
    equal: Sequence[Sequence[str]] = dataclasses.field(default_factory=list)

    def compile(self, params: Sequence[Param]) -> "CompiledConstraints":
        """Validate against ``params`` and build the expansion maps.

        Raises
        ------
        ValueError
            For unknown names, overlaps, tiny groups, mixed transforms or
            fixed values outside a parameter's domain.

        """
        index = {p.name: i for i, p in enumerate(params)}
        if len(index) != len(params):
            msg = "Parameter names must be unique"
            raise ValueError(msg)
        n = len(params)
        fixed_index: dict[int, float] = {}
        for name, value in self.fixed.items():
            if name not in index:
                msg = f"Unknown parameter '{name}' in fixed"
                raise ValueError(msg)
            transform = params[index[name]].transform
            lower_ok = value >= 0 if transform in ("log", "logit") else True
            upper_ok = value <= 1 if transform == "logit" else True
            if not np.isfinite(value) or not lower_ok or not upper_ok:
                msg = f"Fixed value for '{name}' is outside its domain: {value}"
                raise ValueError(msg)
            fixed_index[index[name]] = float(value)

        representative = list(range(n))
        seen = set(fixed_index)
        for group in self.equal:
            if len(group) < _MIN_GROUP_SIZE:
                msg = "Equality groups must have at least 2 elements"
                raise ValueError(msg)
            members = []
            for name in group:
                if name not in index:
                    msg = f"Unknown parameter '{name}' in equal"
                    raise ValueError(msg)
                i = index[name]
                if i in fixed_index:
                    msg = f"Parameter '{name}' appears in both fixed and equal"
                    raise ValueError(msg)
                if i in seen:
                    msg = f"Parameter '{name}' appears in more than one group"
                    raise ValueError(msg)
                seen.add(i)
                members.append(i)
            if len({params[i].transform for i in members}) > 1:
                msg = f"Equality group {list(group)} mixes parameter transforms"
                raise ValueError(msg)
            head = min(members)
            for i in members:
                representative[i] = head

        free = [i for i in range(n) if i not in fixed_index and representative[i] == i]
        position = {i: k for k, i in enumerate(free)}
        source = np.array(
            [
                position[representative[i]] if i not in fixed_index else 0
                for i in range(n)
            ],
            dtype=np.int64,
        )
        return CompiledConstraints(
            params=tuple(params),
            free_indices=np.array(free, dtype=np.int64),
            source=source,
            is_fixed=np.array([i in fixed_index for i in range(n)]),
            fixed_natural=np.array(
                [fixed_index.get(i, 0.0) for i in range(n)], dtype=np.float64
            ),
            codes=np.array([_CODES[p.transform] for p in params], dtype=np.int64),
        )


@dataclasses.dataclass(frozen=True)
class CompiledConstraints:
    """Index maps between free unconstrained parameters and the full natural vector.

    Built by :meth:`Constraints.compile`.
    """

    params: tuple[Param, ...]
    free_indices: np.ndarray  # positions (in ``params``) of the free parameters
    source: np.ndarray  # for each parameter, the free slot it reads
    is_fixed: np.ndarray
    fixed_natural: np.ndarray
    codes: np.ndarray

    @property
    def n_free(self) -> int:
        """Number of free (optimised) parameters."""
        return len(self.free_indices)

    @property
    def free_names(self) -> tuple[str, ...]:
        """Names of the free parameters (group representatives for equal groups)."""
        return tuple(self.params[i].name for i in self.free_indices)

    @property
    def structure_key(self) -> tuple[tuple[int, ...], ...]:
        """Hashable description of the free/fixed/equal structure (no fixed values)."""
        return (
            tuple(int(v) for v in self.source),
            tuple(int(v) for v in self.is_fixed),
            tuple(int(v) for v in self.codes),
        )

    def expand(
        self, free: jax.Array, fixed_natural: jax.Array | None = None
    ) -> jax.Array:
        """Map free unconstrained values to the full natural-scale vector.

        Differentiable in ``free``.
        ``fixed_natural`` replaces the stored fixed values (same structure),
        so a compiled objective can be reused as the fixed values change.
        """
        if self.n_free == 0:  # every parameter is fixed
            return self.expand_full(jnp.zeros(len(self.params)), fixed_natural)
        return self.expand_full(free[self.source], fixed_natural)

    def expand_full(
        self, unconstrained: jax.Array, fixed_natural: jax.Array | None = None
    ) -> jax.Array:
        """Map a full unconstrained vector to natural scale (fixed entries kept)."""
        u = unconstrained
        natural = jnp.where(
            self.codes == _CODES["log"],
            jnp.exp(u),
            jnp.where(self.codes == _CODES["logit"], jax.nn.sigmoid(u), u),
        )
        fixed = self.fixed_natural if fixed_natural is None else fixed_natural
        return jnp.where(self.is_fixed, fixed, natural)

    def reduce(self, natural: ArrayLike) -> jax.Array:
        """Map a full natural-scale vector to the free unconstrained values.

        Group members are represented by the group's first parameter.
        """
        values = jnp.asarray(natural, dtype=jnp.float64)[self.free_indices]
        codes = self.codes[self.free_indices]
        safe = jnp.where(
            codes == _CODES["identity"], values, jnp.clip(values, 1e-300, None)
        )
        return jnp.where(
            codes == _CODES["log"],
            jnp.log(safe),
            jnp.where(codes == _CODES["logit"], jnp.log(safe / (1.0 - safe)), values),
        )
