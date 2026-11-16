r"""Maximum likelihood fitting of birth-death rate models.

:func:`fit` maximises the exact log-likelihood of a set of ages over the
parameters of a :class:`~bdpinf.inference.rate_models.BirthDeathModel`
(piecewise-constant, piecewise-linear, or any other family satisfying the
:class:`~bdpinf.inference.rate_models.RateModel` protocol), optionally under
:class:`~bdpinf.inference.constraints.Constraints`.

The optimiser is a trust-region Newton method (``scipy`` ``trust-exact``) driven
by a jitted JAX gradient and Hessian.
Newton steps are insensitive to the strong scale differences between
parameters (the information about a growth rate grows like $T^2$ while the
information about a rate grows like the number of individuals), which is
what makes first-order methods slow here.
The Hessian at the optimum also gives the covariance, and certifies that the
optimum is an interior maximum rather than a boundary or saddle point.
"""

import dataclasses
import logging
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from bdpinf.inference.constraints import CompiledConstraints, Constraints
from bdpinf.inference.likelihood import AgeData, log_likelihood_rates
from bdpinf.inference.rate_models import BirthDeathModel
from bdpinf.inference.uncertainty import covariance_from_hessian
from bdpinf.models.math_general import DEFAULT_NODES
from bdpinf.models.rate_functions import RateFunction
from bdpinf.models.validation import normalize_and_validate_inputs
from bdpinf.utils import ensure_1d_tensor

__all__ = [
    "FitResult",
    "StartSummary",
    "fit",
    "get_crude_parameter_estimates",
    "moment_initial_rates",
    "profile_likelihood_interval",
]

logger = logging.getLogger(__name__)

_PENALTY = 1e20
"""Objective value returned where the likelihood is not finite."""

_BOUNDARY_LOG_RATE = -12.0
"""Log-scale parameter values below this count as having hit the boundary.

For a rate with a minimum value this is the log of its excess over the minimum.
"""

_METHODS = ("trust-exact", "L-BFGS-B")
_MIN_GROUP = 2
_THOROUGH_STARTS = 5


@dataclasses.dataclass(frozen=True)
class StartSummary:
    """Outcome of one optimisation start."""

    log_likelihood: float
    converged: bool
    n_iter: int
    swapped: bool = False


@dataclasses.dataclass(frozen=True)
class FitResult:
    """Result of :func:`fit`.

    Attributes
    ----------
    model : BirthDeathModel
        The fitted family.
    constraints : Constraints
        The constraints used.
    params : dict[str, float]
        Fitted natural-scale value of every parameter (``birth.<name>``,
        ``death.<name>``), constraints applied.
    birth, death : RateFunction
        The fitted rate functions.
    log_likelihood : float
        Maximised log-likelihood.
    converged : bool
        Whether the per-observation gradient norm met the tolerance.
    score_norm : float
        Euclidean norm of the gradient of the log-likelihood in the free
        unconstrained parameters.
    min_hessian_eig : float | None
        Smallest eigenvalue of the negative-log-likelihood Hessian there;
        positive at a strict interior maximum.
    at_interior_optimum : bool
        Converged, positive-definite Hessian and no rate at its boundary (zero, or
        the family's ``min_value``).
    covariance_free : NDArray | None
        Covariance of the free unconstrained parameters (inverse observed
        information).
    covariance : NDArray | None
        Covariance of the natural-scale parameters (delta method), ordered as
        ``model.params``; fixed parameters have zero variance.
    covariance_unconstrained : NDArray | None
        Covariance of the full unconstrained vector, ordered as ``model.params``
        (for log-scale parameters this is the covariance of the log rates).
    n_iter, n_function_evals : int
        Optimiser iterations and objective evaluations of the selected start.
    message : str
        Optimiser termination message.
    method : str
        Optimiser that produced the result.
    nll_history : list[float]
        Negative log-likelihood after each iteration of the selected start.
    starts : list[StartSummary]
        One summary per start.
    free_values : NDArray
        Fitted free unconstrained parameters.
    compiled : CompiledConstraints
        The expansion maps for ``free_values``.

    """

    model: BirthDeathModel
    constraints: Constraints
    params: dict[str, float]
    birth: RateFunction
    death: RateFunction
    log_likelihood: float
    converged: bool
    score_norm: float
    min_hessian_eig: float | None
    at_interior_optimum: bool
    covariance_free: NDArray[np.float64] | None
    covariance: NDArray[np.float64] | None
    covariance_unconstrained: NDArray[np.float64] | None
    n_iter: int
    n_function_evals: int
    message: str
    method: str
    nll_history: list[float]
    starts: list[StartSummary]
    free_values: NDArray[np.float64]
    compiled: CompiledConstraints

    @property
    def success(self) -> bool:
        """Alias for :attr:`converged`."""
        return self.converged

    @property
    def natural(self) -> NDArray[np.float64]:
        """Natural-scale parameter vector, ordered as ``model.params``."""
        return np.array([self.params[name] for name in self.model.names])

    @property
    def birth_values(self) -> NDArray[np.float64]:
        """Fitted birth parameters (the epoch rates for a piecewise-constant model)."""
        return np.array([self.params[name] for name in self.model.birth_names])

    @property
    def death_values(self) -> NDArray[np.float64]:
        """Fitted death parameters (the epoch rates for a piecewise-constant model)."""
        return np.array([self.params[name] for name in self.model.death_names])

    @property
    def knot_times(self) -> NDArray[np.float64] | None:
        """Fitted knot times (``None`` when the epochs are fixed)."""
        if self.model.knots is None:
            return None
        return np.array([self.params[name] for name in self.model.knot_names])

    @property
    def standard_errors(self) -> dict[str, float] | None:
        """Delta-method standard errors of the natural-scale parameters.

        Wald standard errors of growth-type parameters are not asymptotically
        valid as the observation window grows (their limit law is not normal);
        prefer :func:`profile_likelihood_interval` there.
        """
        if self.covariance is None:
            return None
        se = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        return dict(zip(self.model.names, (float(v) for v in se), strict=True))

    def confidence_intervals(
        self, level: float = 0.95
    ) -> dict[str, tuple[float, float]] | None:
        """Wald intervals computed on the unconstrained scale and mapped back.

        Rates therefore get positive intervals.
        Fixed parameters get a degenerate interval.
        """
        if self.covariance_unconstrained is None:
            return None
        z = float(scipy.stats.norm.ppf(0.5 + 0.5 * level))
        centre = np.where(
            self.compiled.is_fixed, 0.0, self.free_values[self.compiled.source]
        )
        sd = np.sqrt(np.clip(np.diag(self.covariance_unconstrained), 0.0, None))
        lo = np.asarray(self.compiled.expand_full(jnp.asarray(centre - z * sd)))
        hi = np.asarray(self.compiled.expand_full(jnp.asarray(centre + z * sd)))
        return {
            name: (float(a), float(b))
            for name, a, b in zip(self.model.names, lo, hi, strict=True)
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return {
            "params": dict(self.params),
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "score_norm": self.score_norm,
            "min_hessian_eig": self.min_hessian_eig,
            "at_interior_optimum": self.at_interior_optimum,
            "n_iter": self.n_iter,
            "n_function_evals": self.n_function_evals,
            "message": self.message,
            "method": self.method,
            "nll_history": list(self.nll_history),
            "starts": [dataclasses.asdict(s) for s in self.starts],
            "standard_errors": self.standard_errors,
            "covariance": None if self.covariance is None else self.covariance.tolist(),
            "param_names": list(self.model.names),
            "end_time": self.model.end_time,
        }


# ---------------------------------------------------------------------------
# Initial values
# ---------------------------------------------------------------------------


def _all_ages(data: AgeData) -> NDArray[np.float64]:
    """Every genuine age (spine included) as a host array."""
    rest = np.asarray(data.non_spine_ages, dtype=np.float64)
    if data.n_non_spine is not None:
        rest = rest[: int(data.n_non_spine)]
    if data.upper_ages is not None and data.weights is not None:
        # Interval-censored: range midpoints, repeated by their counts.
        upper = np.asarray(data.upper_ages, dtype=np.float64)[: len(rest)]
        counts = np.rint(np.asarray(data.weights)[: len(rest)]).astype(np.int64)
        return np.repeat(0.5 * (rest + upper), counts)
    if data.spine_ages is None:
        return rest
    return np.concatenate([rest, np.asarray(data.spine_ages, dtype=np.float64)])


def _growth_rate_root(n_bar: float, lam: float, horizon: float) -> float | None:
    """Non-zero root of the moment equation for the growth rate, if one is bracketed."""

    def score(g: float) -> float:
        return g * (n_bar - 1.0) - lam * math.expm1(g * horizon)

    slope = n_bar - 1.0 - lam * horizon
    if abs(slope) < 1e-12:
        return 0.0
    direction = 1.0 if slope > 0 else -1.0
    edge = direction / horizon
    while score(edge) > 0 and abs(edge) * horizon < 500:
        edge *= 2.0
    if score(edge) > 0:
        return None
    lower, upper = (1e-12, edge) if direction > 0 else (edge, -1e-12)
    return scipy.optimize.brentq(score, lower, upper)


def moment_initial_rates(data: AgeData) -> tuple[float, float] | None:
    r"""Constant-rate moment estimators of $(\lambda, \mu)$ from the data.

    For constant rates the non-spine ages tend to $\mathrm{Exp}(\lambda)$,
    so $\tilde\lambda = 1/\bar A$; the population size gives the growth rate
    $\tilde\gamma$ as the non-zero root of
    $\gamma(\bar n - 1) = \tilde\lambda(e^{\gamma T} - 1)$, and
    $\tilde\mu = \tilde\lambda - \tilde\gamma$.
    Returns ``None`` when no sensible root exists (for example one individual).
    """
    ages = _all_ages(data)
    mean_age = float(ages.mean()) if ages.size else 0.0
    n_bar = float(np.mean(np.asarray(data.ns)))
    horizon = data.end_time
    if mean_age <= 0 or n_bar <= 1.0 or data.initial_population != 1:
        return None
    lam = 1.0 / mean_age
    try:
        gamma = _growth_rate_root(n_bar, lam, horizon)
    except ValueError:
        return None
    if gamma is None:
        return None
    mu = lam - gamma
    if not (math.isfinite(lam) and math.isfinite(mu)) or lam <= 0:
        return None
    return lam, max(mu, 0.05 * lam)


def get_crude_parameter_estimates(
    ages: jax.Array | NDArray[np.float64] | float,
    times: jax.Array | NDArray[np.float64] | float,
    *,
    birth_rates: jax.Array | NDArray[np.float64] | float | None = None,
    death_rates: jax.Array | NDArray[np.float64] | float | None = None,
    initial_population: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Get crude initial epoch rates from age data (fallback heuristic)."""
    if birth_rates is not None and death_rates is not None:
        msg = "Cannot provide both birth_rates and death_rates"
        raise ValueError(msg)

    times_arr, ages_arr = normalize_and_validate_inputs(times=times, ages=ages)

    observation_time = times_arr[-1]
    n_individuals = len(ages_arr)
    if n_individuals == 0:
        msg = "Cannot estimate parameters from empty age data"
        raise ValueError(msg)

    mean_age = float(ages_arr.mean())

    if birth_rates is None:
        est_birth_rates = jnp.ones_like(times_arr, dtype=jnp.float64) / mean_age
    else:
        est_birth_rates = ensure_1d_tensor(birth_rates)

    if death_rates is None:
        est_death_rates = jnp.maximum(
            est_birth_rates
            - jnp.log(
                jnp.array(float(n_individuals / initial_population), dtype=jnp.float64),
            )
            / observation_time,
            jnp.full_like(times_arr, 1e-8, dtype=jnp.float64),
        )
    else:
        est_death_rates = ensure_1d_tensor(death_rates)

    if len(times_arr) != len(est_birth_rates) or len(times_arr) != len(est_death_rates):
        msg = (
            f"Size mismatch: times ({len(times_arr)}), "
            f"est_birth_rates ({len(est_birth_rates)}), "
            f"est_death_rates ({len(est_death_rates)}) must all be equal"
        )
        raise ValueError(msg)

    # Pull birth and death rates a quarter of the way toward each other, using the
    # original (pre-shrink) values for both so the pull is symmetric.
    rate_gap = est_death_rates - est_birth_rates
    est_birth_rates = est_birth_rates + rate_gap / 4
    est_death_rates = est_death_rates - rate_gap / 4

    return (est_birth_rates, est_death_rates)


def _default_init(data: AgeData, model: BirthDeathModel) -> Iterator[jax.Array]:
    """Yield candidate natural-scale starting vectors, best first (computed lazily)."""
    moments = moment_initial_rates(data)
    if moments is not None:
        yield model.constant_natural(*moments)
    try:
        birth, death = get_crude_parameter_estimates(
            _all_ages(data), data.end_time, initial_population=data.initial_population
        )
        yield model.constant_natural(float(birth[0]), float(death[0]))
    except ValueError:
        pass
    yield model.constant_natural(1.0 / data.end_time, 0.5 / data.end_time)


InitSpec = (
    Mapping[str, float] | ArrayLike | FitResult | tuple[RateFunction, RateFunction]
)


def _resolve_init(
    init: InitSpec | None, data: AgeData, model: BirthDeathModel
) -> Iterator[jax.Array]:
    """Yield the candidate starting vectors implied by the user's ``init``."""
    if init is None:
        yield from _default_init(data, model)
    elif isinstance(init, FitResult):
        knots = [init.params.get(name) for name in model.knot_names]
        yield model.natural_from_rates(
            init.birth,
            init.death,
            None if None in knots else knots,  # type: ignore[arg-type]
        )
    elif isinstance(init, tuple):
        yield model.natural_from_rates(*init)
    elif isinstance(init, Mapping):
        base = np.array(next(_default_init(data, model)), dtype=np.float64)
        for name, value in init.items():
            if name not in model.names:
                msg = f"Unknown parameter '{name}' in init"
                raise ValueError(msg)
            base[model.names.index(name)] = float(value)
        yield jnp.asarray(base)
    else:
        arr = jnp.asarray(init, dtype=jnp.float64)
        if arr.shape != (len(model.names),):
            msg = f"init has shape {arr.shape}, expected ({len(model.names)},)"
            raise ValueError(msg)
        yield arr


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _RunOutcome:
    x: NDArray[np.float64]
    nll: float
    grad_norm: float
    n_iter: int
    n_evals: int
    message: str
    method: str
    history: list[float]


def _run_optimizer(
    x0: NDArray[np.float64],
    value_and_grad: Any,
    hessian: Any,
    *,
    method: str,
    grad_tol: float,
    max_iter: int,
) -> _RunOutcome:
    """Run one scipy optimiser from ``x0``."""
    evaluations = 0
    seen: dict[bytes, float] = {}

    def fun(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nonlocal evaluations
        evaluations += 1
        value, grad = value_and_grad(jnp.asarray(x))
        seen[np.asarray(x, dtype=np.float64).tobytes()] = float(value)
        return float(value), np.asarray(grad, dtype=np.float64)

    history: list[float] = []

    def record(xk: NDArray[np.float64]) -> None:
        key = np.asarray(xk, dtype=np.float64).tobytes()
        history.append(seen[key] if key in seen else fun(np.asarray(xk))[0])

    if method == "trust-exact":
        result = scipy.optimize.minimize(
            fun,
            x0,
            jac=True,
            hess=lambda x: np.asarray(hessian(jnp.asarray(x)), dtype=np.float64),
            method="trust-exact",
            callback=record,
            options={"gtol": grad_tol, "maxiter": max_iter},
        )
    else:
        result = scipy.optimize.minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            callback=record,
            options={"gtol": grad_tol, "maxiter": 10 * max_iter, "ftol": 1e-15},
        )
    x = np.asarray(result.x, dtype=np.float64)
    nll, grad = fun(x)
    return _RunOutcome(
        x=x,
        nll=nll,
        grad_norm=float(np.linalg.norm(grad)),
        n_iter=int(getattr(result, "nit", len(history))),
        n_evals=evaluations,
        message=str(result.message),
        method=method,
        history=history or [nll],
    )


def _optimise_start(
    x0: NDArray[np.float64],
    value_and_grad: Any,
    hessian: Any,
    *,
    method: str,
    grad_tol: float,
    max_iter: int,
) -> _RunOutcome:
    """Optimise from one start, falling back across methods if it stalls."""

    def attempt(start: NDArray[np.float64], how: str) -> _RunOutcome | None:
        try:
            return _run_optimizer(
                start,
                value_and_grad,
                hessian,
                method=how,
                grad_tol=grad_tol,
                max_iter=max_iter,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            logger.debug("%s failed from %s: %s", how, start, exc)
            return None

    first = attempt(x0, method)
    if first is not None and first.grad_norm <= grad_tol and first.nll < _PENALTY:
        return first
    best = first or _RunOutcome(
        x0, _PENALTY, float("inf"), 0, 0, "optimiser failed", method, []
    )
    other = "L-BFGS-B" if method == "trust-exact" else "trust-exact"
    for fallback in (other, "trust-exact"):
        retry = attempt(best.x, fallback)
        if retry is None:
            continue
        if retry.nll <= best.nll + 1e-12:
            retry.n_iter += best.n_iter
            retry.n_evals += best.n_evals
            retry.history = best.history + retry.history
            best = retry
        if best.grad_norm <= grad_tol:
            break
    return best


_MIN_BUCKET = 32


def _bucket(n: int) -> int:
    """Smallest power of two (at least ``_MIN_BUCKET``) holding ``n`` ages."""
    return max(_MIN_BUCKET, 1 << max(0, int(n) - 1).bit_length())


@dataclasses.dataclass(frozen=True)
class _Objective:
    """Jitted negative log-likelihood, its gradient and its Hessian."""

    nll: Any
    value_and_grad: Any
    hessian: Any


_OBJECTIVES: dict[tuple[Any, ...], _Objective] = {}
_MAX_CACHED_OBJECTIVES = 32


_HESSIAN_MODES = ("fwd-fwd", "fwd-rev", "fd-grad")
_FD_STEP = 1e-5


def _hessian_function(penalised: Callable[..., jax.Array], mode: str) -> Any:
    """Return the Hessian of ``penalised`` in its first argument.

    ``"fwd-fwd"`` is exact and compiles quickly for small models, but its
    compile time and memory grow sharply with the number of rate segments;
    ``"fwd-rev"`` differentiates the reverse-mode gradient forward;
    ``"fd-grad"`` takes central differences of the gradient, which needs no
    second-order compilation at all.
    """
    if mode == "fwd-fwd":
        return jax.jacfwd(jax.jacfwd(penalised))
    if mode == "fwd-rev":
        return jax.jacfwd(jax.grad(penalised))
    if mode == "fd-grad":
        grad = jax.grad(penalised)

        def hessian(free: jax.Array, *args: jax.Array) -> jax.Array:
            step = _FD_STEP * jnp.maximum(1.0, jnp.abs(free))
            moves = jnp.diag(step)

            def up_grad(move: jax.Array) -> jax.Array:
                return grad(free + move, *args)

            def down_grad(move: jax.Array) -> jax.Array:
                return grad(free - move, *args)

            up = jax.vmap(up_grad)(moves)
            down = jax.vmap(down_grad)(moves)
            h = (up - down) / (2.0 * step[:, None])
            return 0.5 * (h + h.T)

        return hessian
    msg = f"Unknown hessian_mode '{mode}'. Choose from {list(_HESSIAN_MODES)}."
    raise ValueError(msg)


def _objective(
    model: BirthDeathModel,
    compiled: CompiledConstraints,
    *,
    has_spine: bool,
    use_unknown_spine: bool,
    initial_population: int,
    end_time: float,
    condition_on_survival: bool,
    n_nodes: int,
    has_ranges: bool = False,
    hessian_mode: str = "fwd-fwd",
) -> _Objective:
    """Return the jitted objective for this model and constraint structure.

    The data arrays and the fixed parameter values are *arguments* of the jitted
    functions, so one compilation serves every fit with the same structure and
    array shapes (bootstrap replicates, profile-likelihood refits, recovery
    studies with equal sample sizes).
    Reverse mode gives the gradient and forward-over-forward the Hessian, which
    compiles about twice as fast as the default for the small parameter counts
    here.
    """
    key = (
        model,
        compiled.structure_key,
        has_spine,
        use_unknown_spine,
        initial_population,
        end_time,
        condition_on_survival,
        n_nodes,
        has_ranges,
        hessian_mode,
    )
    cached = _OBJECTIVES.get(key)
    if cached is not None:
        return cached

    def penalised(
        free: jax.Array,
        non_spine: jax.Array,
        spine: jax.Array,
        ns: jax.Array,
        fixed_natural: jax.Array,
        n_valid: jax.Array,
        upper: jax.Array,
        weights: jax.Array,
    ) -> jax.Array:
        birth, death = model.rates(compiled.expand(free, fixed_natural))
        data = AgeData(
            non_spine_ages=non_spine,
            spine_ages=spine if has_spine else None,
            ns=ns,
            end_time=end_time,
            use_unknown_spine=use_unknown_spine,
            initial_population=initial_population,
            n_non_spine=None if has_ranges else n_valid,
            upper_ages=upper if has_ranges else None,
            weights=weights if has_ranges else None,
        )
        log_lik = log_likelihood_rates(
            data,
            birth,
            death,
            condition_on_survival=condition_on_survival,
            n_nodes=n_nodes,
        )
        return jnp.where(jnp.isfinite(log_lik), -log_lik, _PENALTY)

    objective = _Objective(
        nll=jax.jit(penalised),
        value_and_grad=jax.jit(jax.value_and_grad(penalised)),
        hessian=jax.jit(_hessian_function(penalised, hessian_mode)),
    )
    if len(_OBJECTIVES) >= _MAX_CACHED_OBJECTIVES:
        _OBJECTIVES.clear()
    _OBJECTIVES[key] = objective
    return objective


def _check_method_and_window(
    data: AgeData, model: BirthDeathModel, method: str
) -> None:
    """Raise ``ValueError`` for an unknown optimiser or a window mismatch."""
    if method not in _METHODS:
        msg = f"Unsupported method '{method}'. Choose from {list(_METHODS)}."
        raise ValueError(msg)
    if not np.isclose(model.end_time, data.end_time):
        msg = (
            f"Model window {model.end_time} does not match the data "
            f"observation time {data.end_time}."
        )
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class _Problem:
    """A fit prepared for the optimiser: the objective closed over the runtime data."""

    compiled: CompiledConstraints
    grad_tol: float
    nll: Callable[[jax.Array], jax.Array]
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]]
    hessian: Callable[[jax.Array], jax.Array]


def _prepare_problem(
    data: AgeData,
    model: BirthDeathModel,
    compiled: CompiledConstraints,
    *,
    condition_on_survival: bool,
    n_nodes: int,
    pad_to_bucket: bool,
    hessian_mode: str,
    tol: float,
) -> _Problem:
    """Fetch the jitted objective, build the runtime arrays and bind them."""
    objective = _objective(
        model,
        compiled,
        has_spine=data.spine_ages is not None,
        use_unknown_spine=data.use_unknown_spine,
        initial_population=data.initial_population,
        end_time=data.end_time,
        condition_on_survival=condition_on_survival,
        n_nodes=n_nodes,
        has_ranges=data.has_ranges,
        hessian_mode=hessian_mode,
    )
    n_rest = (
        len(data.non_spine_ages) if data.n_non_spine is None else int(data.n_non_spine)
    )
    padded = (
        data
        if pad_to_bucket is False or data.n_non_spine is not None
        else data.padded(_bucket(n_rest))
    )
    runtime = (
        jnp.asarray(padded.non_spine_ages),
        jnp.zeros(0) if data.spine_ages is None else jnp.asarray(data.spine_ages),
        jnp.asarray(data.ns),
        jnp.asarray(compiled.fixed_natural),
        jnp.asarray(n_rest),
        jnp.zeros(0) if padded.upper_ages is None else jnp.asarray(padded.upper_ages),
        jnp.zeros(0) if padded.weights is None else jnp.asarray(padded.weights),
    )

    def nll(free: jax.Array) -> jax.Array:
        return objective.nll(free, *runtime)

    def value_and_grad(free: jax.Array) -> tuple[jax.Array, jax.Array]:
        return objective.value_and_grad(free, *runtime)

    def hessian(free: jax.Array) -> jax.Array:
        return objective.hessian(free, *runtime)

    n_obs = max(1, data.n_observations)
    return _Problem(
        compiled=compiled,
        grad_tol=tol * n_obs,
        nll=nll,
        value_and_grad=value_and_grad,
        hessian=hessian,
    )


def _first_start(
    problem: _Problem, init: InitSpec | None, data: AgeData, model: BirthDeathModel
) -> tuple[NDArray[np.float64], jax.Array]:
    """Return the first candidate start with a finite likelihood (reduced, natural)."""
    for candidate in _resolve_init(init, data, model):
        u0 = np.asarray(problem.compiled.reduce(candidate), dtype=np.float64)
        if float(problem.nll(jnp.asarray(u0))) < _PENALTY:
            return u0, candidate
    msg = "No starting point has a finite likelihood; pass `init`."
    raise ValueError(msg)


def _starting_points(
    problem: _Problem,
    model: BirthDeathModel,
    first: NDArray[np.float64],
    natural0: jax.Array,
    *,
    n_starts: int,
    perturb_std: float,
    seed: int | None,
) -> list[tuple[NDArray[np.float64], bool]]:
    """Return the starts as ``(unconstrained start, was_swapped)``, in order."""
    starts: list[tuple[NDArray[np.float64], bool]] = [(first, False)]
    if n_starts > 1:
        swapped = model.swapped(natural0)
        if swapped is not None:
            u_swap = np.asarray(problem.compiled.reduce(swapped), dtype=np.float64)
            if float(problem.nll(jnp.asarray(u_swap))) < _PENALTY:
                starts.append((u_swap, True))
        rng = np.random.default_rng(seed)
        while len(starts) < n_starts:
            starts.append(
                (first + rng.normal(0.0, perturb_std, size=first.shape), False)
            )
    return starts


def _run_starts(
    problem: _Problem,
    starts: list[tuple[NDArray[np.float64], bool]],
    *,
    method: str,
    max_iter: int,
) -> tuple[list[_RunOutcome], list[StartSummary]]:
    """Optimise from every start, returning the outcomes and one summary per start."""
    outcomes: list[_RunOutcome] = []
    summaries: list[StartSummary] = []
    for u_start, was_swapped in starts:
        out = _optimise_start(
            u_start,
            problem.value_and_grad,
            problem.hessian,
            method=method,
            grad_tol=problem.grad_tol,
            max_iter=max_iter,
        )
        outcomes.append(out)
        summaries.append(
            StartSummary(
                log_likelihood=-out.nll if out.nll < _PENALTY else float("-inf"),
                converged=out.grad_norm <= problem.grad_tol,
                n_iter=out.n_iter,
                swapped=was_swapped,
            )
        )
    return outcomes, summaries


class _Spread(NamedTuple):
    """Smallest Hessian eigenvalue and the covariances at the optimum."""

    min_eig: float | None
    free: NDArray[np.float64] | None
    natural: NDArray[np.float64] | None
    unconstrained: NDArray[np.float64] | None


def _min_eigenvalue(hess: NDArray[np.float64]) -> float | None:
    """Smallest eigenvalue of the symmetric ``hess``, ``None`` if it is not finite."""
    try:
        return float(np.linalg.eigvalsh(hess).min())
    except np.linalg.LinAlgError:  # non-finite Hessian at a failed fit
        return None


def _free_selector(compiled: CompiledConstraints, n_params: int) -> NDArray[np.float64]:
    """Matrix picking each non-fixed parameter's free coordinate (full vector order)."""
    select = np.zeros((n_params, compiled.n_free))
    for i, k in enumerate(compiled.source):
        if not compiled.is_fixed[i]:
            select[i, k] = 1.0
    return select


def _covariances(
    problem: _Problem,
    x_best: NDArray[np.float64],
    n_params: int,
    *,
    converged: bool,
    compute: bool,
) -> _Spread:
    """Hessian at ``x_best``: smallest eigenvalue and covariance on three scales."""
    compiled = problem.compiled
    if compiled.n_free == 0:
        return _Spread(None, None, None, None)
    hess = np.asarray(problem.hessian(jnp.asarray(x_best)), dtype=np.float64)
    hess = 0.5 * (hess + hess.T)
    min_eig = _min_eigenvalue(hess)
    if not compute or min_eig is None:
        return _Spread(min_eig, None, None, None)
    if not converged:
        # Called straight from ``fit``, so level 3 is the caller of ``fit``.
        warnings.warn(
            "Estimating uncertainty from a fit that did not converge; the "
            "Hessian is evaluated at a possibly non-stationary point, so "
            "the resulting covariance/standard errors may not be valid.",
            UserWarning,
            stacklevel=3,
        )
    cov_free = np.asarray(covariance_from_hessian(jnp.asarray(hess)))
    jac = np.asarray(jax.jacobian(compiled.expand)(jnp.asarray(x_best)))
    select = _free_selector(compiled, n_params)
    return _Spread(
        min_eig, cov_free, jac @ cov_free @ jac.T, select @ cov_free @ select.T
    )


def _at_boundary(compiled: CompiledConstraints, x_best: NDArray[np.float64]) -> bool:
    """Whether a free parameter sits at the edge of its transform's range."""
    # Transform codes are the bare integers of ``_CODES`` in ``constraints.py``
    # (private there, with no named constants): 1 is "log", 3 is "interval".
    free_codes = compiled.codes[compiled.free_indices]
    return bool(
        np.any(x_best[free_codes == 1] < _BOUNDARY_LOG_RATE)
        or np.any(np.abs(x_best[free_codes == 3]) > _BOUNDARY_LOG_RATE)
    )


def fit(
    data: AgeData,
    model: BirthDeathModel,
    *,
    constraints: Constraints | None = None,
    init: InitSpec | None = None,
    method: str = "trust-exact",
    n_starts: int | None = None,
    thorough: bool = False,
    perturb_std: float = 0.3,
    seed: int | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    condition_on_survival: bool = True,
    compute_covariance: bool = True,
    n_nodes: int = DEFAULT_NODES,
    pad_to_bucket: bool = True,
    hessian_mode: str = "fwd-fwd",
) -> FitResult:
    r"""Fit a birth-death rate model to age data by maximum likelihood.

    Parameters
    ----------
    data : AgeData
        Observed ages (see :meth:`AgeData.from_groups`).
    model : BirthDeathModel
        Parametric family for the birth and death rates, on the same
        observation window as ``data``.
    constraints : Constraints, optional
        Fixed parameters and equality groups, by name.
    init : mapping | array-like | FitResult | tuple of RateFunction, optional
        Starting point: natural-scale values by name, a full natural-scale
        vector, a previous :class:`FitResult` (for example a piecewise-constant
        fit warm-starting a piecewise-linear one), or a ``(birth, death)`` pair
        of rate functions.
        By default constant rates from :func:`moment_initial_rates`.
    method : {"trust-exact", "L-BFGS-B"}
        Primary optimiser.
        The other one is the fallback if the primary stalls.
    n_starts : int, optional
        Number of starts (default 1, or 5 with ``thorough``).
        With more than one, the second start exchanges birth and death (when
        the two families have the same layout), because ages alone are
        symmetric in the two rates; further starts perturb the first on the
        unconstrained scale by ``perturb_std``.
        The start with the highest likelihood is returned.
    thorough : bool
        Use 5 starts (the given one, the birth-death swap and three
        perturbed ones) unless ``n_starts`` is given, and keep the best.
    perturb_std : float
        Standard deviation of the unconstrained-scale perturbation.
    seed : int, optional
        Seed for the perturbations.
    tol : float
        Convergence tolerance on the gradient norm per observation.
    max_iter : int
        Iteration limit per optimiser run.
    condition_on_survival : bool
        Condition the population-count term on non-extinction.
    compute_covariance : bool
        Compute the covariance from the observed information.
    n_nodes : int
        Gauss-Legendre nodes per segment for non-piecewise-constant rates.
    pad_to_bucket : bool
        Pad the age vector (masked out of the likelihood) to the next power of
        two so that fits with similar sample sizes share one compilation of the
        gradient and Hessian, which otherwise dominates the run time.
        Results are identical to the unpadded fit.
    hessian_mode : {"fwd-fwd", "fwd-rev", "fd-grad"}
        How the Hessian is computed.
        ``"fd-grad"`` (central differences of the gradient, accurate to about
        1e-10 relative) needs no second-order compilation and was about 4 times
        faster per call than the exact forms on a 13-rate piecewise-linear
        model; ``"fwd-fwd"`` (exact, the default) and ``"fwd-rev"`` are exact.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        If the model and data windows differ, ``method`` is unknown, or no
        starting point has a finite likelihood.

    """
    _check_method_and_window(data, model, method)
    constraints = constraints or Constraints()
    if n_starts is None:
        n_starts = _THOROUGH_STARTS if thorough else 1
    problem = _prepare_problem(
        data,
        model,
        constraints.compile(model.params),
        condition_on_survival=condition_on_survival,
        n_nodes=n_nodes,
        pad_to_bucket=pad_to_bucket,
        hessian_mode=hessian_mode,
        tol=tol,
    )
    compiled = problem.compiled

    summaries: list[StartSummary] = []
    if compiled.n_free == 0:
        value = float(problem.nll(jnp.zeros(0)))
        best = _RunOutcome(
            np.zeros(0), value, 0.0, 0, 1, "no free parameters", method, [value]
        )
        summaries.append(StartSummary(-value, True, 0))
    else:
        first, natural0 = _first_start(problem, init, data, model)
        starts = _starting_points(
            problem,
            model,
            first,
            natural0,
            n_starts=n_starts,
            perturb_std=perturb_std,
            seed=seed,
        )
        outcomes, summaries = _run_starts(
            problem, starts, method=method, max_iter=max_iter
        )
        best = min(outcomes, key=lambda o: o.nll)

    x_best = best.x
    converged = best.grad_norm <= problem.grad_tol and best.nll < _PENALTY
    natural = model.canonical(compiled.expand(jnp.asarray(x_best)))
    birth, death = model.rates(jnp.asarray(natural))
    spread = _covariances(
        problem, x_best, len(natural), converged=converged, compute=compute_covariance
    )
    boundary = _at_boundary(compiled, x_best)
    interior = bool(
        converged
        and not boundary
        and (spread.min_eig > 0 if spread.min_eig is not None else compiled.n_free == 0)
    )

    return FitResult(
        model=model,
        constraints=constraints,
        params=dict(zip(model.names, (float(v) for v in natural), strict=True)),
        birth=birth,
        death=death,
        log_likelihood=-best.nll if best.nll < _PENALTY else float("-inf"),
        converged=converged,
        score_norm=best.grad_norm,
        min_hessian_eig=spread.min_eig,
        at_interior_optimum=interior,
        covariance_free=spread.free,
        covariance=spread.natural,
        covariance_unconstrained=spread.unconstrained,
        n_iter=best.n_iter,
        n_function_evals=best.n_evals,
        message=best.message,
        method=best.method,
        nll_history=best.history,
        starts=summaries,
        free_values=x_best,
        compiled=compiled,
    )


def profile_likelihood_interval(
    result: FitResult,
    data: AgeData,
    name: str,
    *,
    level: float = 0.95,
    span: float = 4.0,
    **fit_kwargs: Any,
) -> tuple[float, float]:
    r"""Profile-likelihood confidence interval for one parameter.

    Re-fits with ``name`` held fixed and finds where twice the drop in
    log-likelihood reaches the $\chi^2_1$ quantile.
    Unlike a Wald interval this stays valid when the sampling distribution is
    not normal (growth-type parameters over long windows).

    Parameters
    ----------
    result : FitResult
        The maximum-likelihood fit.
    data : AgeData
        The data it was fitted to.
    name : str
        Parameter to profile (a name in ``result.model.names``).
    level : float
        Confidence level.
    span : float
        Search range: for log-scale parameters the log of the value moves by up
        to this much either side, otherwise the value moves by this much.
    **fit_kwargs
        Forwarded to :func:`fit` (for example ``n_nodes``).

    Returns
    -------
    tuple[float, float]
        Lower and upper limits; a limit is the parameter's minimum (``0.0`` unless
        the rate family has a ``min_value``) or ``inf`` if not bracketed.

    """
    if name not in result.model.names:
        msg = f"Unknown parameter '{name}'"
        raise ValueError(msg)
    target = result.log_likelihood - 0.5 * float(scipy.stats.chi2.ppf(level, 1))
    centre = result.params[name]
    param = result.model.params[result.model.names.index(name)]
    is_log = param.transform == "log"
    floor = param.lower if is_log else 0.0  # minimum value of a log parameter
    fixed_base = dict(result.constraints.fixed)
    # Profiling frees ``name`` from its equality group; the rest stay tied.
    equal = [
        rest
        for group in result.constraints.equal
        if len(rest := [n for n in group if n != name]) >= _MIN_GROUP
    ]

    def drop(value: float) -> float:
        constraints = Constraints(fixed={**fixed_base, name: value}, equal=equal)
        refit = fit(
            data,
            result.model,
            constraints=constraints,
            init=result,
            compute_covariance=False,
            **fit_kwargs,
        )
        return refit.log_likelihood - target

    def search(direction: int) -> float:
        steps = np.arange(1, 9) / 8.0 * span
        probe = (
            floor + (centre - floor) * np.exp(direction * steps)
            if is_log
            else centre + direction * steps
        )
        previous = centre
        for value in probe:
            if value < floor and is_log:
                break
            if drop(float(value)) < 0:
                a, b = sorted((previous, float(value)))
                return float(
                    scipy.optimize.brentq(drop, a, b, xtol=1e-6 * max(1.0, abs(centre)))
                )
            previous = float(value)
        return floor if direction < 0 else float("inf")

    return search(-1), search(+1)
