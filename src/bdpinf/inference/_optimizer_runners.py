"""LBFGS and first-order optimizer runner functions (JAX backend).

Extracted from fitting.py to keep that module below 1000 lines.
Depends only on _fit_types, stdlib, JAX, scipy, and optax.
"""

import logging
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize

from inf_bp_texts.inference._fit_types import (
    ConvergenceConfig,
    OptimizerStrategy,
)

logger = logging.getLogger(__name__)


def _grad_norm(grads: jax.Array) -> float:
    """Return L2 norm of a gradient array as a Python float."""
    return float(jnp.sqrt(jnp.sum(grads**2)))


def _check_convergence_criteria(
    optimizer_name: str,
    *,
    grad_norm: float | None = None,
    nll_history: list[float] | None = None,
    convergence_config: ConvergenceConfig | None = None,
    iterations_without_improvement: int | None = None,
) -> bool:
    """Check if optimizer meets convergence criteria."""
    if convergence_config is None:
        convergence_config = ConvergenceConfig.default()

    if grad_norm is not None and grad_norm <= convergence_config.gradient_threshold:
        return True

    min_nll_history_len = 2

    if optimizer_name.lower() == "lbfgs":
        if nll_history is None or len(nll_history) < min_nll_history_len:
            return False
        # Converged when improvement from previous step fell below threshold
        # (nll[-2] - nll[-1] is positive when NLL decreasing, small at plateau).
        return nll_history[-2] - nll_history[-1] < convergence_config.absolute_tolerance

    if iterations_without_improvement is None:
        return False

    return iterations_without_improvement >= convergence_config.patience


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

            if nll_val < first_nll - convergence_config.absolute_tolerance:
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
                "ftol": ftol / 1e-8,
            },
        )
        iterations = iteration_count[0]

        x_final = jnp.array(opt_result.x, dtype=jnp.float64)
        _, grads_final = val_and_grad(x_final)
        gn_final = _grad_norm(grads_final)
        nll_final = float(val_and_grad(x_final)[0])

        if nll_final < first_nll - convergence_config.absolute_tolerance:
            improved_once = True

        # Primary convergence check — gradient criterion takes priority.
        # `improved_once` is only required when relying on NLL-change criterion.
        converged = _check_convergence_criteria(
            optimizer_name=strategy.name,
            grad_norm=gn_final,
            nll_history=nll_history,
            convergence_config=convergence_config,
        )
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

    if optimizer_name_lower == "sgd":
        tx = optax.sgd(lr)
    else:
        tx = optax.adam(lr)

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
        elif nll_val < first_nll - convergence_config.absolute_tolerance:
            improved_once = True

        if nll_val >= best_nll:
            iterations_without_improvement += 1
        else:
            improvement = best_nll - nll_val
            iterations_without_improvement = (
                0
                if improvement >= convergence_config.absolute_tolerance
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
