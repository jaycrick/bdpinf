"""LBFGS and first-order optimizer runner functions.

Extracted from fitting.py to keep that module below 1000 lines.
Depends only on _fit_types, stdlib, and torch.
"""

import logging
import time
from collections.abc import Callable

import torch

from inf_bp_texts.inference._fit_types import (
    ConvergenceConfig,
    OptimizerStrategy,
)

logger = logging.getLogger(__name__)


def _capture_grad_norm(log_params: torch.nn.Parameter) -> float:
    """Safely compute gradient norm, returning inf on failure."""
    try:
        return (
            float(torch.norm(log_params.grad).item())
            if log_params.grad is not None
            else float("inf")
        )
    except Exception as exc:
        logger.warning("grad norm capture failed: %s", exc)
        return float("inf")


def _capture_param_snapshot(log_params: torch.nn.Parameter) -> list[float] | None:
    """Safely snapshot current parameters, returning None on failure."""
    try:
        result: list[float] = log_params.detach().cpu().numpy().tolist()
        return result
    except Exception as exc:
        logger.warning("param snapshot failed: %s", exc)
        return None


def _check_convergence_criteria(
    optimizer_name: str,
    *,
    log_params: torch.nn.Parameter | None = None,
    nll_history: list[float] | None = None,
    convergence_config: ConvergenceConfig | None = None,
    iterations_without_improvement: int | None = None,
) -> bool:
    """Check if optimizer meets convergence criteria.

    Evaluates convergence based on optimizer type and stopping criteria.

    Args:
        optimizer_name: Name of optimizer type ("LBFGS", "Adam", "SGD").
        log_params: Parameter tensor to compute gradient norm from.
        nll_history: History of negative log likelihood values.
        convergence_config: Convergence configuration with thresholds.
        iterations_without_improvement: For non-LBFGS, iterations without improvement.

    Returns
    -------
        True if convergence criteria are met, False otherwise.

    """
    # Compute gradient norm if log_params provided
    grad_norm = float("inf")
    if log_params is not None and log_params.grad is not None:
        grad_norm = float(torch.norm(log_params.grad).item())

    if convergence_config is None:
        convergence_config = ConvergenceConfig.default()

    if grad_norm <= convergence_config.gradient_threshold:
        return True

    min_nll_history_len = 2

    if optimizer_name.lower() == "lbfgs":
        if nll_history is None or len(nll_history) < min_nll_history_len:
            return False
        return nll_history[-1] - nll_history[-2] < convergence_config.absolute_tolerance

    if iterations_without_improvement is None:
        return False

    return iterations_without_improvement >= convergence_config.patience


def _run_lbfgs_optimizer(
    strategy: OptimizerStrategy,
    log_params: torch.nn.Parameter,
    optimizer: torch.optim.LBFGS,
    max_iters: int,
    nll_history: list[float],
    grad_norms: list[float],
    param_history: list[list[float]],
    timestamps: list[float],
    convergence_config: ConvergenceConfig,
    compute_nll_fn: Callable[[], torch.Tensor],
) -> tuple[bool, Exception | None, torch.Tensor | None, int]:
    """Run LBFGS optimization loop.

    Returns
    -------
        (did_converge, current_exception, best_log_params, iterations)

    """
    last_nll_value: float | None = None
    last_grad_norm: float | None = None
    last_grad_norm_local: float | None = None
    last_param_snapshot_local: list[float] | None = None
    start_time = time.perf_counter()
    record_every = int(strategy.optimizer_kwargs.get("record_every", 1))
    min_iterations = max(1, int(strategy.optimizer_kwargs.get("min_iterations", 1)))
    best_nll = float("inf")
    best_log_params: torch.Tensor | None = None
    first_nll: float | None = None
    improved_once = False
    current_exception: Exception | None = None
    iterations = 0

    for iteration in range(max_iters):
        iterations = iteration + 1

        def closure() -> torch.Tensor:
            nonlocal last_nll_value
            nonlocal nll_history
            nonlocal best_nll
            nonlocal best_log_params
            nonlocal last_grad_norm_local
            nonlocal last_param_snapshot_local
            optimizer.zero_grad()
            nll = compute_nll_fn()
            nll.backward()  # type: ignore[no-untyped-call]
            last_nll_value = float(nll.item())
            # capture grad norm and parameter snapshot inside closure
            last_grad_norm_local = _capture_grad_norm(log_params)
            last_param_snapshot_local = _capture_param_snapshot(log_params)
            if last_nll_value < best_nll:
                best_nll = last_nll_value
                best_log_params = log_params.detach().clone()
            nll_history.append(last_nll_value)
            return nll

        optimizer.step(closure)  # type: ignore[no-untyped-call]

        # append diagnostics once per iteration according to record_every
        if (iteration % record_every) == 0:
            grad_to_append = (
                float(last_grad_norm_local)
                if last_grad_norm_local is not None
                else (
                    float(torch.norm(log_params.grad).item())
                    if log_params.grad is not None
                    else float("inf")
                )
            )
            param_to_append = (
                last_param_snapshot_local
                if last_param_snapshot_local is not None
                else log_params.detach().cpu().numpy().tolist()
            )
            grad_norms.append(grad_to_append)
            param_history.append(param_to_append)
            timestamps.append(time.perf_counter() - start_time)

        # LBFGS may evaluate the closure multiple times per "iteration"
        # (for line-search). Use the recorded `nll_history` to detect
        # whether the objective has improved relative to the initial
        # recorded value. This is more robust than relying on the last
        # closure evaluation alone.
        try:
            if nll_history:
                if first_nll is None:
                    first_nll = float(nll_history[0])
                if min(nll_history) < first_nll - convergence_config.absolute_tolerance:
                    improved_once = True
        except Exception as exc:
            logger.warning("NLL history inspection failed: %s", exc)

        if last_nll_value is None:
            current_exception = RuntimeError(
                f"Optimizer {strategy.name} did not produce a valid objective value.",
            )
            break

        if first_nll is None:
            first_nll = last_nll_value
        elif last_nll_value < first_nll - convergence_config.absolute_tolerance:
            improved_once = True

        if (
            improved_once
            and iterations >= min_iterations
            and _check_convergence_criteria(
                optimizer_name=strategy.name,
                log_params=log_params,
                nll_history=nll_history,
                convergence_config=convergence_config,
            )
        ):
            return True, None, best_log_params, iterations

        last_grad_norm = (
            float(torch.norm(log_params.grad).item())
            if log_params.grad is not None
            else float("inf")
        )

    if not improved_once:
        current_exception = RuntimeError(
            f"Objective did not improve from initial value with {strategy.name}.",
        )

    if current_exception is not None:
        return False, current_exception, best_log_params, iterations

    first_nll_value = first_nll if first_nll is not None else float("nan")
    final_nll_value = last_nll_value if last_nll_value is not None else float("nan")
    improvement = first_nll_value - final_nll_value
    current_exception = RuntimeError(
        "Convergence not reached with "
        f"{strategy.name}: max_iters={max_iters}, "
        f"history_len={len(nll_history)}, "
        f"first_nll={first_nll_value:.6f}, "
        f"final_nll={final_nll_value:.6f}, "
        f"improvement={improvement:.6f}, "
        f"last_grad_norm={last_grad_norm}, "
        f"gradient_threshold={convergence_config.gradient_threshold}",
    )
    return False, current_exception, best_log_params, iterations


def _run_first_order_optimizer(
    strategy: OptimizerStrategy,
    log_params: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    max_iters: int,
    nll_history: list[float],
    grad_norms: list[float],
    param_history: list[list[float]],
    timestamps: list[float],
    convergence_config: ConvergenceConfig,
    compute_nll_fn: Callable[[], torch.Tensor],
) -> tuple[bool, Exception | None, torch.Tensor | None, int]:
    """Run first-order optimizer (Adam/SGD) loop.

    Returns
    -------
        (did_converge, current_exception, best_log_params, iterations)

    """
    best_nll = float("inf")
    iterations_without_improvement = 0
    last_grad_norm = None
    best_log_params: torch.Tensor | None = None
    first_nll: float | None = None
    improved_once = False
    current_exception: Exception | None = None
    absolute_tolerance = convergence_config.absolute_tolerance
    iterations = 0

    for iteration in range(max_iters):
        iterations = iteration + 1
        nll = compute_nll_fn()
        optimizer.zero_grad()

        if not torch.isfinite(nll):
            current_exception = RuntimeError(
                f"Non-finite objective encountered during optimization with "
                f"{strategy.name}.",
            )
            return False, current_exception, None, iterations

        nll.backward()  # type: ignore[no-untyped-call]

        # record gradient norm (pre-step) and optional parameter snapshot (post-step)
        grad_norm_pre = _capture_grad_norm(log_params)

        optimizer.step()

        try:
            nll_value = float(nll.item())
        except Exception as exc:
            logger.warning("NLL value extraction failed: %s", exc)
            nll_value = float("nan")
        nll_history.append(nll_value)

        # append diagnostics
        record_every = int(strategy.optimizer_kwargs.get("record_every", 1))
        if (iteration % record_every) == 0:
            grad_to_append = grad_norm_pre
            param_snapshot = _capture_param_snapshot(log_params)
            grad_norms.append(grad_to_append)
            param_history.append(param_snapshot if param_snapshot is not None else [])
            timestamps.append(time.perf_counter())

        if first_nll is None:
            first_nll = nll_value
        elif nll_value < first_nll - absolute_tolerance:
            improved_once = True

        if nll_value >= best_nll:
            iterations_without_improvement += 1
        else:
            improvement = best_nll - nll_value
            if improvement < absolute_tolerance:
                iterations_without_improvement += 1
            else:
                iterations_without_improvement = 0

            best_nll = nll_value
            best_log_params = log_params.detach().clone()

        if improved_once and _check_convergence_criteria(
            optimizer_name=strategy.name,
            log_params=log_params,
            nll_history=nll_history,
            convergence_config=convergence_config,
            iterations_without_improvement=iterations_without_improvement,
        ):
            return True, None, best_log_params, iterations

        last_grad_norm = (
            float(torch.norm(log_params.grad).item())
            if log_params.grad is not None
            else None
        )

    if not improved_once:
        current_exception = RuntimeError(
            f"Objective did not improve from initial value with {strategy.name}.",
        )
        return False, current_exception, None, iterations

    first_nll_value = first_nll if first_nll is not None else float("nan")
    final_nll_value = nll_history[-1] if nll_history else float("nan")
    improvement = first_nll_value - final_nll_value
    current_exception = RuntimeError(
        "Convergence not reached with "
        f"{strategy.name}: max_iters={max_iters}, "
        f"iterations={iterations}, "
        f"history_len={len(nll_history)}, "
        f"first_nll={first_nll_value:.6f}, "
        f"best_nll={best_nll:.6f}, "
        f"final_nll={final_nll_value:.6f}, "
        f"improvement={improvement:.6f}, "
        f"last_grad_norm={last_grad_norm}, "
        f"iterations_without_improvement={iterations_without_improvement}, "
        f"patience={convergence_config.patience}, "
        f"gradient_threshold={convergence_config.gradient_threshold}",
    )
    return False, current_exception, best_log_params, iterations
