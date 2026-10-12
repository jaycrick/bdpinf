"""LBFGS and first-order optimizer runner functions (JAX backend).

Extracted from fitting.py to keep that module below 1000 lines.
Depends only on _fit_types, stdlib, JAX, scipy, and optax.
"""

import dataclasses
import logging
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize

from bdpinf.inference._fit_types import (
    PATIENCE_GRADIENT_MULTIPLIER,
    ConvergenceConfig,
    OptimizerStrategy,
)

logger = logging.getLogger(__name__)

# Solver kwargs that the scipy L-BFGS-B path actually consumes.  Anything else
# (e.g. ``lr``, ``line_search_fn`` — leftovers from the PyTorch LBFGS era) is
# silently ignored, so we log it at debug level rather than failing.
_LBFGS_RECOGNIZED_KWARGS = frozenset(
    {"record_every", "tolerance_grad", "tolerance_change"},
)


def _grad_norm(grads: jax.Array) -> float:
    """Return L2 norm of a gradient array as a Python float."""
    return float(jnp.sqrt(jnp.sum(grads**2)))


def _rel_tol(absolute_tolerance: float, reference: float) -> float:
    """Scale an absolute NLL tolerance to a relative one.

    Returns ``absolute_tolerance * (1 + |reference|)`` so that the convergence
    thresholds are invariant to the magnitude of the negative log-likelihood
    (which grows with the number of observations).
    """
    return absolute_tolerance * (1.0 + abs(reference))


def _scaled_config(  # pyright: ignore[reportUnusedFunction]
    convergence_config: ConvergenceConfig,
    n_observations: int,
) -> ConvergenceConfig:
    """Scale the gradient threshold by the observation count.

    The objective is a *sum* over observations, so its gradient norm grows with
    ``n_observations``.  Scaling the threshold makes the gradient stopping
    criterion sample-size invariant (a per-observation gradient threshold).
    """
    factor = float(max(1, n_observations))
    return dataclasses.replace(
        convergence_config,
        gradient_threshold=convergence_config.gradient_threshold * factor,
    )


def _check_convergence_criteria(
    optimizer_name: str,
    *,
    grad_norm: float | None = None,
    nll_history: list[float] | None = None,
    convergence_config: ConvergenceConfig | None = None,
    iterations_without_improvement: int | None = None,
) -> bool:
    """Check if optimizer meets convergence criteria.

    ``convergence_config.gradient_threshold`` is expected to already be scaled
    for the observation count (see :func:`_scaled_config`).  The NLL-change test
    is relative to the current NLL magnitude (see :func:`_rel_tol`).
    """
    if convergence_config is None:
        convergence_config = ConvergenceConfig.default()

    if grad_norm is not None and grad_norm <= convergence_config.gradient_threshold:
        return True

    min_nll_history_len = 2

    if optimizer_name.lower() == "lbfgs":
        if nll_history is None or len(nll_history) < min_nll_history_len:
            return False
        # Converged when improvement from previous step fell below a relative
        # threshold (nll[-2] - nll[-1] is positive when NLL decreasing, small at
        # plateau).  Relative scaling avoids rejecting valid convergence on
        # large-NLL problems.
        return nll_history[-2] - nll_history[-1] < _rel_tol(
            convergence_config.absolute_tolerance, nll_history[-1]
        )

    if iterations_without_improvement is None:
        return False

    # Patience (no-improvement) path: require the gradient to also be small, so a
    # noisy first-order plateau far from a stationary point is not mistaken for
    # convergence.
    if iterations_without_improvement < convergence_config.patience:
        return False
    patience_grad_ceiling = (
        PATIENCE_GRADIENT_MULTIPLIER * convergence_config.gradient_threshold
    )
    return grad_norm is not None and grad_norm <= patience_grad_ceiling


def run_lbfgs_optimizer(
    strategy: OptimizerStrategy,
    log_params: jax.Array,
    max_iters: int,
    nll_history: list[float],
    grad_norms: list[float],
    param_history: list[list[float]],
    timestamps: list[float],
    convergence_config: ConvergenceConfig,
    val_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
) -> tuple[bool, Exception | None, jax.Array | None, int]:
    """Run L-BFGS-B optimization via scipy.optimize with JAX gradients.

    Parameters
    ----------
    val_and_grad_fn : Callable
        Pre-built ``(params) -> (nll, grad)`` callable.  Pass a
        ``jax.jit``-compiled callable for maximum throughput; the caller is
        responsible for compilation.

    Returns
    -------
        (did_converge, current_exception, best_log_params, iterations)

    """
    start_time = time.perf_counter()
    unrecognized = set(strategy.solver_kwargs) - _LBFGS_RECOGNIZED_KWARGS
    if unrecognized:
        logger.debug(
            "Ignoring unrecognized LBFGS solver_kwargs %s "
            "(scipy L-BFGS-B only consults %s)",
            sorted(unrecognized),
            sorted(_LBFGS_RECOGNIZED_KWARGS),
        )
    record_every = int(strategy.solver_kwargs.get("record_every", 1))
    best_nll = float("inf")
    best_log_params: jax.Array | None = None
    improved_once = False
    iterations = 0

    val_and_grad = val_and_grad_fn

    def scipy_obj(x: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (nll_float, grad_numpy) for scipy.optimize."""
        x_jax = jnp.array(x, dtype=jnp.float64)
        nll_jax, grads_jax = val_and_grad(x_jax)
        return float(nll_jax), np.array(grads_jax, dtype=np.float64)

    try:
        x0 = np.array(log_params, dtype=np.float64)

        # Evaluate at the initial point so nll_history is never empty
        # and first_nll is always set before scipy runs.
        init_nll_val, init_grad_np = scipy_obj(x0)
        first_nll = init_nll_val
        best_nll = init_nll_val
        best_log_params = jnp.array(x0, dtype=jnp.float64)
        init_gn = float(np.linalg.norm(init_grad_np))

        nll_history.append(init_nll_val)
        grad_norms.append(init_gn)
        param_history.append(x0.tolist())
        timestamps.append(0.0)

        # Gradient already satisfies threshold → converge immediately.
        if init_gn <= convergence_config.gradient_threshold:
            return True, None, best_log_params, 0

        iteration_count = [0]

        def callback(x: np.ndarray) -> None:
            """Record per-iteration diagnostics."""
            nonlocal best_nll, best_log_params, improved_once
            iteration_count[0] += 1
            it = iteration_count[0]

            x_jax = jnp.array(x, dtype=jnp.float64)
            nll_jax, grads_jax = val_and_grad(x_jax)
            nll_val = float(nll_jax)
            gn = _grad_norm(grads_jax)

            nll_history.append(nll_val)

            if nll_val < first_nll - _rel_tol(
                convergence_config.absolute_tolerance, first_nll
            ):
                improved_once = True

            if nll_val < best_nll:
                best_nll = nll_val
                best_log_params = x_jax

            if (it % record_every) == 0:
                grad_norms.append(gn)
                param_history.append(x.tolist())
                timestamps.append(time.perf_counter() - start_time)

        gtol = strategy.solver_kwargs.get(
            "tolerance_grad", convergence_config.gradient_threshold
        )
        ftol = strategy.solver_kwargs.get(
            "tolerance_change", convergence_config.absolute_tolerance
        )
        opt_result = scipy.optimize.minimize(  # pyright: ignore[reportCallIssue]
            scipy_obj,
            x0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": max_iters,
                "gtol": gtol,
                "ftol": ftol,
            },
        )
        iterations = iteration_count[0]

        x_final = jnp.array(opt_result.x, dtype=jnp.float64)
        nll_final_jax, grads_final = val_and_grad(x_final)
        gn_final = _grad_norm(grads_final)
        nll_final = float(nll_final_jax)

        # scipy's reported optimum (opt_result.x) need not coincide with the
        # last callback iterate — append it explicitly so nll_history/best_nll
        # and the plateau check reflect the actual optimum, not just the last
        # callback point.
        if not nll_history or nll_history[-1] != nll_final:
            nll_history.append(nll_final)
            grad_norms.append(gn_final)
            param_history.append(opt_result.x.tolist())
            timestamps.append(time.perf_counter() - start_time)

        if nll_final < best_nll:
            best_nll = nll_final
            best_log_params = x_final

        if nll_final < first_nll - _rel_tol(
            convergence_config.absolute_tolerance, first_nll
        ):
            improved_once = True

        # Primary convergence check — gradient criterion takes priority.
        # `improved_once` is only required when relying on NLL-change criterion.
        # Also accept scipy's own convergence signal: L-BFGS-B sets success=True
        # when either gtol (gradient) or ftol (relative function change) is met.
        # Our absolute NLL-change check is stricter than scipy's relative one and
        # would incorrectly reject valid convergence for large-NLL problems.
        converged = _check_convergence_criteria(
            optimizer_name=strategy.name,
            grad_norm=gn_final,
            nll_history=nll_history,
            convergence_config=convergence_config,
        ) or (opt_result.success and improved_once)
        if not converged and not improved_once:
            return (
                False,
                RuntimeError(
                    f"Objective did not improve from initial value"
                    f" with {strategy.name}."
                ),
                best_log_params,
                iterations,
            )
        if not converged:
            return (
                False,
                RuntimeError(
                    f"Convergence not reached with {strategy.name}: "
                    f"max_iters={max_iters}, "
                    f"history_len={len(nll_history)}, "
                    f"final_nll={nll_final:.6f}, "
                    f"grad_norm={gn_final:.6f}, "
                    f"gradient_threshold={convergence_config.gradient_threshold}"
                ),
                best_log_params,
                iterations,
            )
        return True, None, best_log_params, iterations

    except Exception as exc:
        return False, exc, best_log_params, max(iterations, 1)


def run_first_order_optimizer(
    strategy: OptimizerStrategy,
    log_params: jax.Array,
    max_iters: int,
    nll_history: list[float],
    grad_norms: list[float],
    param_history: list[list[float]],
    timestamps: list[float],
    convergence_config: ConvergenceConfig,
    val_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
) -> tuple[bool, Exception | None, jax.Array | None, int]:
    """Run first-order optimizer (Adam/SGD) via optax with a Python-level loop.

    Parameters
    ----------
    val_and_grad_fn : Callable
        Pre-built ``(params) -> (nll, grad)`` callable.  Pass a
        ``jax.jit``-compiled callable for maximum throughput; the caller is
        responsible for compilation.

    Returns
    -------
        (did_converge, current_exception, best_log_params, iterations)

    """
    lr = float(strategy.solver_kwargs.get("lr", 0.1))
    record_every = int(strategy.solver_kwargs.get("record_every", 1))
    optimizer_name_lower = strategy.name.lower()

    tx = optax.sgd(lr) if optimizer_name_lower == "sgd" else optax.adam(lr)

    opt_state = tx.init(log_params)
    params = log_params

    val_and_grad = val_and_grad_fn

    best_nll = float("inf")
    best_params: jax.Array | None = None
    first_nll: float | None = None
    improved_once = False
    iterations_without_improvement = 0
    current_exception: Exception | None = None
    iterations = 0
    start_time = time.perf_counter()

    # Already-converged guard (mirrors the LBFGS early-return): if the initial
    # point already satisfies the gradient criterion, there is nothing to
    # optimize and the "no improvement" exception below would otherwise fire
    # spuriously on an already-optimal start.  This check is side-effect free
    # (no history append) so it does not duplicate the loop's own first record.
    init_nll_jax, init_grads = val_and_grad(params)
    init_nll_val = float(init_nll_jax)
    if not np.isfinite(init_nll_val):
        return (
            False,
            RuntimeError(
                f"Non-finite objective encountered during optimization"
                f" with {strategy.name}."
            ),
            None,
            0,
        )
    if _grad_norm(init_grads) <= convergence_config.gradient_threshold:
        nll_history.append(init_nll_val)
        grad_norms.append(_grad_norm(init_grads))
        param_history.append(np.array(params).tolist())
        timestamps.append(0.0)
        return True, None, jnp.asarray(params, dtype=jnp.float64), 0

    for iteration in range(max_iters):
        iterations = iteration + 1

        try:
            nll_jax, grads = val_and_grad(params)
        except Exception as exc:
            current_exception = exc
            break

        nll_val = float(nll_jax)

        if not np.isfinite(nll_val):
            current_exception = RuntimeError(
                f"Non-finite objective encountered during optimization"
                f" with {strategy.name}."
            )
            return False, current_exception, None, iterations

        gn = _grad_norm(grads)
        nll_history.append(nll_val)

        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if (iteration % record_every) == 0:
            grad_norms.append(gn)
            param_history.append(np.array(params).tolist())
            timestamps.append(time.perf_counter() - start_time)

        if first_nll is None:
            first_nll = nll_val
        elif nll_val < first_nll - _rel_tol(
            convergence_config.absolute_tolerance, first_nll
        ):
            improved_once = True

        if nll_val >= best_nll:
            iterations_without_improvement += 1
        else:
            improvement = best_nll - nll_val
            iterations_without_improvement = (
                0
                if improvement
                >= _rel_tol(convergence_config.absolute_tolerance, best_nll)
                else iterations_without_improvement + 1
            )
            best_nll = nll_val
            best_params = jnp.asarray(params, dtype=jnp.float64)

        if improved_once and _check_convergence_criteria(
            optimizer_name=strategy.name,
            grad_norm=gn,
            nll_history=nll_history,
            convergence_config=convergence_config,
            iterations_without_improvement=iterations_without_improvement,
        ):
            return True, None, best_params, iterations

    if not improved_once:
        current_exception = RuntimeError(
            f"Objective did not improve from initial value with {strategy.name}."
        )
        return False, current_exception, None, iterations

    first_nll_value = first_nll if first_nll is not None else float("nan")
    final_nll_value = nll_history[-1] if nll_history else float("nan")
    improvement = first_nll_value - final_nll_value
    last_gn = grad_norms[-1] if grad_norms else float("nan")
    current_exception = RuntimeError(
        f"Convergence not reached with {strategy.name}: "
        f"max_iters={max_iters}, "
        f"iterations={iterations}, "
        f"history_len={len(nll_history)}, "
        f"first_nll={first_nll_value:.6f}, "
        f"best_nll={best_nll:.6f}, "
        f"final_nll={final_nll_value:.6f}, "
        f"improvement={improvement:.6f}, "
        f"last_grad_norm={last_gn}, "
        f"iterations_without_improvement={iterations_without_improvement}, "
        f"patience={convergence_config.patience}, "
        f"gradient_threshold={convergence_config.gradient_threshold}"
    )
    return False, current_exception, best_params, iterations
