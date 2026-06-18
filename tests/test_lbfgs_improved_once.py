"""Tests verifying LBFGS makes at least one improvement step from initial values."""

import numpy as np

from inf_bp_texts.inference import ConvergenceConfig
from inf_bp_texts.inference.fitting import fit_birth_death_processes


def _make_simple_ages():
    # all ages must be <= observation time (1.0)
    return np.array([0.1, 0.2, 0.5, 0.7, 0.95], dtype=np.float64)


def _run_lbfgs(line_search_fn: str | None = None):
    optimizer_kwargs = {"lbfgs": {"record_every": 1}}
    if line_search_fn is not None:
        optimizer_kwargs["lbfgs"]["line_search_fn"] = line_search_fn

    return fit_birth_death_processes(
        list_of_ages=[_make_simple_ages()],
        times=1.0,
        optimizers=["LBFGS"],
        optimizer_kwargs=optimizer_kwargs,
        perturb_initial_params_std=0.0,
        random_seed=42,
        convergence_config=ConvergenceConfig.default(),
        estimate_uncertainty=False,
    )


def _assert_improved(result: dict):
    attempts = result["optimization_result"]["attempts"]
    assert attempts, "No optimizer attempts recorded"
    attempt = attempts[0]
    first_nll = attempt.get("first_nll")
    best_nll = attempt.get("best_nll")
    nll_history = attempt.get("nll_history", [])

    assert len(nll_history) >= 2, "Insufficient nll history recorded"
    assert first_nll is not None and best_nll is not None
    assert best_nll < first_nll - 1e-8, (
        f"Expected improvement: best_nll < first_nll, got best={best_nll}, first={first_nll}"
    )


class TestLbfgsImprovedOnce:
    """Test that LBFGS improves from the initial NLL value within one pass."""

    def test_lbfgs_improved_once_default(self):
        result = _run_lbfgs()
        _assert_improved(result)

    def test_lbfgs_improved_once_strong_wolfe(self):
        result = _run_lbfgs(line_search_fn="strong_wolfe")
        _assert_improved(result)
