"""Test that LBFGS improves on its starting point and reports it truthfully."""

import numpy as np
import pytest

from bdpinf.inference import ConvergenceConfig
from bdpinf.inference.fitting import fit_birth_death_processes


def _make_simple_ages() -> np.ndarray:
    # all ages must be <= observation time (1.0)
    return np.array([0.1, 0.2, 0.5, 0.7, 0.95], dtype=np.float64)


def _run_lbfgs(line_search_fn: str | None = None) -> dict:
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


class TestLbfgs:
    """Test the first LBFGS attempt records a real improvement and no false error."""

    @pytest.mark.parametrize("line_search_fn", [None, "strong_wolfe"])
    def test_improves_from_initial_value(self, line_search_fn: str | None) -> None:
        attempt = _run_lbfgs(line_search_fn)["optimization_result"]["attempts"][0]

        first_nll, best_nll = attempt["first_nll"], attempt["best_nll"]
        assert len(attempt.get("nll_history", [])) >= 2
        assert best_nll < first_nll - 1e-8
        # An improving run must never carry the "did not improve" exception.
        assert "did not improve from initial value" not in (
            attempt.get("exception") or ""
        )
