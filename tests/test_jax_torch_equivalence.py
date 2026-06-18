"""JAX ↔ torch equivalence gate for core likelihood functions.

Tests in this module form the blocking Phase 2a gate: the JAX spike
(``inf_bp_texts.models._jax``) must numerically agree with the existing
torch implementations before any in-place Phase 2b rewrite begins.

Three test groups:

1. **Golden reproduction** — JAX reproduces the ``_GOLDEN_*`` float
   constants from ``test_regression_likelihoods.py`` at ``_ATOL_GOLDEN = 1e-4``.
   This is the same tolerance used by the existing regression suite, so a
   JAX failure here would imply a systematic error rather than floating-point
   noise.

2. **JAX ↔ torch equivalence** — function-by-function agreement at
   ``_ATOL_EQUIV = 1e-8`` across single-epoch, two-epoch, and a parameter
   grid covering supercritical / subcritical / critical regimes.

3. **Gradient equivalence** — ``jax.grad`` against ``torch.autograd.grad``
   at ``_ATOL_GRAD = 1e-6``.

   Functions covered by gradient tests (fully ``jax.grad``-safe — no Python
   control flow on abstract traced values):

   * ``calculate_extinction_pr``
   * ``calculate_non_spine_ages_log_lik`` (single-epoch path)
   * ``calculate_spine_ages_log_lik`` (single-epoch path)

   Not yet grad-safe in the Phase 2a spike and therefore **not** covered here
   (Phase 2b will remove the Python guards):

   * ``calculate_populations_log_lik`` — uses ``float(alpha_t)`` guards
   * top-level ``log_likelihood`` — multiple Python guards on traced arrays

``JAX_PLATFORMS=cpu`` is set in ``conftest.py`` before any JAX import so the
A40-12Q vGPU (which does not support CUDA VMM required by XLA) is bypassed.
"""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from inf_bp_texts.inference.likelihood import log_likelihood as torch_log_likelihood
from inf_bp_texts.models._jax import (
    calculate_extinction_pr as jax_ext_pr,
    calculate_non_spine_ages_log_lik as jax_non_spine,
    calculate_populations_log_lik as jax_pop,
    calculate_spine_ages_log_lik as jax_spine,
    log_likelihood as jax_log_likelihood,
)
from inf_bp_texts.models.math_ages import (
    calculate_non_spine_ages_log_lik as torch_non_spine,
    calculate_spine_ages_log_lik as torch_spine,
)
from inf_bp_texts.models.math_extinction import calculate_extinction_pr as torch_ext_pr
from inf_bp_texts.models.math_populations import calculate_populations_log_lik as torch_pop

# ---------------------------------------------------------------------------
# JAX parameter arrays — single epoch (canonical supercritical)
# ---------------------------------------------------------------------------
_B = jnp.array([0.05], dtype=jnp.float64)
_D = jnp.array([0.03], dtype=jnp.float64)
_T = jnp.array([1500.0], dtype=jnp.float64)
_AGE_750 = jnp.array([750.0], dtype=jnp.float64)
_NS_5 = jnp.array([5.0], dtype=jnp.float64)
_AGES_10 = jnp.array(
    [100.0, 200.0, 300.0, 450.0, 600.0, 750.0, 900.0, 1050.0, 1200.0, 1350.0],
    dtype=jnp.float64,
)
_AGES_10_NP = np.array(
    [100.0, 200.0, 300.0, 450.0, 600.0, 750.0, 900.0, 1050.0, 1200.0, 1350.0]
)

# JAX parameter arrays — two-epoch (supercritical → subcritical)
_B2 = jnp.array([0.05, 0.01], dtype=jnp.float64)
_D2 = jnp.array([0.03, 0.02], dtype=jnp.float64)
_T2 = jnp.array([750.0, 1500.0], dtype=jnp.float64)
_AGE_300 = jnp.array([300.0], dtype=jnp.float64)
_NS_2 = jnp.array([3.0, 5.0], dtype=jnp.float64)
_AGES_10_MULTI = jnp.array(
    [80.0, 200.0, 350.0, 500.0, 650.0, 720.0, 900.0, 1050.0, 1200.0, 1400.0],
    dtype=jnp.float64,
)
_AGES_10_MULTI_NP = np.array(
    [80.0, 200.0, 350.0, 500.0, 650.0, 720.0, 900.0, 1050.0, 1200.0, 1400.0]
)

# Torch equivalents (same numerical values, float64 dtype)
_B_T = torch.tensor([0.05], dtype=torch.float64)
_D_T = torch.tensor([0.03], dtype=torch.float64)
_T_T = torch.tensor([1500.0], dtype=torch.float64)
_AGE_750_T = torch.tensor([750.0], dtype=torch.float64)
_NS_5_T = torch.tensor([5], dtype=torch.int64)
_B2_T = torch.tensor([0.05, 0.01], dtype=torch.float64)
_D2_T = torch.tensor([0.03, 0.02], dtype=torch.float64)
_T2_T = torch.tensor([750.0, 1500.0], dtype=torch.float64)
_AGE_300_T = torch.tensor([300.0], dtype=torch.float64)
_NS_2_T = torch.tensor([3, 5], dtype=torch.int64)

# ---------------------------------------------------------------------------
# Golden values (mirrored from test_regression_likelihoods.py)
# The JAX path must reproduce these to _ATOL_GOLDEN before Phase 2b begins.
# ---------------------------------------------------------------------------
_GOLDEN_EXTINCTION_SUPERCRITICAL = 0.6000000000
_GOLDEN_EXTINCTION_SUBCRITICAL = 1.0000000000
_GOLDEN_EXTINCTION_CRITICAL = 0.9868421053
_GOLDEN_SPINE_LL_750 = -25.0902674713
_GOLDEN_NON_SPINE_LL_750 = -40.4957324571
_GOLDEN_POPULATIONS_LL_N5 = -31.8330083712
_GOLDEN_COMBINED_LL_10AGES = -381.6393679271
_GOLDEN_SPINE_LL_300_MULTI = -6.9630921864
_GOLDEN_NON_SPINE_LL_300_MULTI = -7.6053693664
_GOLDEN_POPULATIONS_LL_NS_2_MULTI = -18.6682576565
_GOLDEN_COMBINED_LL_10AGES_MULTI = -177.7867323299

_ATOL_GOLDEN = 1e-4   # same as the existing regression golden tolerance
_ATOL_EQUIV = 1e-8    # JAX ↔ torch must agree to this tightness
_ATOL_GRAD = 1e-6     # gradient equivalence tolerance


def _f(x: object) -> float:
    """Extract a Python float from a JAX array, torch Tensor, or numpy scalar."""
    return float(x)  # type: ignore[arg-type]


@pytest.mark.regression
class TestJaxTorchEquivalence:
    """JAX ↔ torch numerical equivalence gate for core likelihood functions."""

    # =========================================================================
    # Group 1 — Golden value reproduction (JAX path reproduces _GOLDEN_* at 1e-4)
    # =========================================================================

    def test_golden_extinction_supercritical(self) -> None:
        assert abs(_f(jax_ext_pr(_B, _D, _T)) - _GOLDEN_EXTINCTION_SUPERCRITICAL) < _ATOL_GOLDEN

    def test_golden_extinction_subcritical(self) -> None:
        b = jnp.array([0.03], dtype=jnp.float64)
        d = jnp.array([0.05], dtype=jnp.float64)
        assert abs(_f(jax_ext_pr(b, d, _T)) - _GOLDEN_EXTINCTION_SUBCRITICAL) < _ATOL_GOLDEN

    def test_golden_extinction_critical(self) -> None:
        b = jnp.array([0.05], dtype=jnp.float64)
        d = jnp.array([0.05], dtype=jnp.float64)
        assert abs(_f(jax_ext_pr(b, d, _T)) - _GOLDEN_EXTINCTION_CRITICAL) < _ATOL_GOLDEN

    def test_golden_spine_ll_single_age(self) -> None:
        assert abs(_f(jax_spine(_B, _D, _AGE_750, _T)) - _GOLDEN_SPINE_LL_750) < _ATOL_GOLDEN

    def test_golden_non_spine_ll_single_age(self) -> None:
        assert abs(_f(jax_non_spine(_B, _D, _AGE_750, _T)) - _GOLDEN_NON_SPINE_LL_750) < _ATOL_GOLDEN

    def test_golden_populations_ll_n5(self) -> None:
        result = jax_pop(_B, _D, _NS_5, _T, initial_population=1, unbias=False)
        assert abs(_f(result) - _GOLDEN_POPULATIONS_LL_N5) < _ATOL_GOLDEN

    def test_golden_combined_ll_ten_ages(self) -> None:
        result = jax_log_likelihood(_B, _D, _AGES_10, _T)
        assert abs(_f(result) - _GOLDEN_COMBINED_LL_10AGES) < _ATOL_GOLDEN

    def test_golden_spine_ll_multi_epoch(self) -> None:
        assert abs(_f(jax_spine(_B2, _D2, _AGE_300, _T2)) - _GOLDEN_SPINE_LL_300_MULTI) < _ATOL_GOLDEN

    def test_golden_non_spine_ll_multi_epoch(self) -> None:
        assert abs(_f(jax_non_spine(_B2, _D2, _AGE_300, _T2)) - _GOLDEN_NON_SPINE_LL_300_MULTI) < _ATOL_GOLDEN

    def test_golden_populations_ll_multi_epoch(self) -> None:
        result = jax_pop(_B2, _D2, _NS_2, _T2, initial_population=1, unbias=False)
        assert abs(_f(result) - _GOLDEN_POPULATIONS_LL_NS_2_MULTI) < _ATOL_GOLDEN

    def test_golden_combined_ll_multi_epoch(self) -> None:
        result = jax_log_likelihood(_B2, _D2, _AGES_10_MULTI, _T2)
        assert abs(_f(result) - _GOLDEN_COMBINED_LL_10AGES_MULTI) < _ATOL_GOLDEN

    # =========================================================================
    # Group 2 — JAX ↔ torch equivalence at _ATOL_EQUIV = 1e-8
    # =========================================================================

    # --- calculate_extinction_pr ---

    def test_equiv_extinction_supercritical(self) -> None:
        jax_val = _f(jax_ext_pr(_B, _D, _T))
        torch_val = _f(torch_ext_pr(_B_T, _D_T, _T_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_extinction_subcritical(self) -> None:
        b_j = jnp.array([0.03], dtype=jnp.float64)
        d_j = jnp.array([0.05], dtype=jnp.float64)
        b_t = torch.tensor([0.03], dtype=torch.float64)
        d_t = torch.tensor([0.05], dtype=torch.float64)
        assert abs(_f(jax_ext_pr(b_j, d_j, _T)) - _f(torch_ext_pr(b_t, d_t, _T_T))) < _ATOL_EQUIV

    def test_equiv_extinction_critical(self) -> None:
        b_j = jnp.array([0.05], dtype=jnp.float64)
        d_j = jnp.array([0.05], dtype=jnp.float64)
        b_t = torch.tensor([0.05], dtype=torch.float64)
        d_t = torch.tensor([0.05], dtype=torch.float64)
        assert abs(_f(jax_ext_pr(b_j, d_j, _T)) - _f(torch_ext_pr(b_t, d_t, _T_T))) < _ATOL_EQUIV

    def test_equiv_extinction_multi_epoch(self) -> None:
        jax_val = _f(jax_ext_pr(_B2, _D2, _T2))
        torch_val = _f(torch_ext_pr(_B2_T, _D2_T, _T2_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    # --- calculate_spine_ages_log_lik ---

    def test_equiv_spine_ll_single_age(self) -> None:
        jax_val = _f(jax_spine(_B, _D, _AGE_750, _T))
        torch_val = _f(torch_spine(_B_T, _D_T, _AGE_750_T, _T_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_spine_ll_single_age_per_element(self) -> None:
        """Per-element (return_array=True) values agree element-wise."""
        jax_arr = np.array(jax_spine(_B, _D, _AGE_750, _T, return_array=True))
        torch_arr = np.array(torch_spine(_B_T, _D_T, _AGE_750_T, _T_T, return_array=True))
        assert np.allclose(jax_arr, torch_arr, atol=_ATOL_EQUIV, rtol=0.0)

    def test_equiv_spine_ll_multi_epoch(self) -> None:
        jax_val = _f(jax_spine(_B2, _D2, _AGE_300, _T2))
        torch_val = _f(torch_spine(_B2_T, _D2_T, _AGE_300_T, _T2_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_spine_ll_at_final_time(self) -> None:
        """Boundary mass at age == T — both give the same (finite) value."""
        age_at_T = jnp.array([1500.0], dtype=jnp.float64)
        age_at_T_t = torch.tensor([1500.0], dtype=torch.float64)
        jax_val = _f(jax_spine(_B, _D, age_at_T, _T))
        torch_val = _f(torch_spine(_B_T, _D_T, age_at_T_t, _T_T))
        assert np.isfinite(jax_val)
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    # --- calculate_non_spine_ages_log_lik ---

    def test_equiv_non_spine_ll_single_age(self) -> None:
        jax_val = _f(jax_non_spine(_B, _D, _AGE_750, _T))
        torch_val = _f(torch_non_spine(_B_T, _D_T, _AGE_750_T, _T_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_non_spine_ll_single_age_per_element(self) -> None:
        jax_arr = np.array(jax_non_spine(_B, _D, _AGE_750, _T, return_array=True))
        torch_arr = np.array(torch_non_spine(_B_T, _D_T, _AGE_750_T, _T_T, return_array=True))
        assert np.allclose(jax_arr, torch_arr, atol=_ATOL_EQUIV, rtol=0.0)

    def test_equiv_non_spine_ll_multi_epoch(self) -> None:
        jax_val = _f(jax_non_spine(_B2, _D2, _AGE_300, _T2))
        torch_val = _f(torch_non_spine(_B2_T, _D2_T, _AGE_300_T, _T2_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_non_spine_ll_critical(self) -> None:
        """Critical (λ ≈ μ) branch path agrees."""
        b_j = jnp.array([0.05], dtype=jnp.float64)
        d_j = jnp.array([0.05], dtype=jnp.float64)
        b_t = torch.tensor([0.05], dtype=torch.float64)
        d_t = torch.tensor([0.05], dtype=torch.float64)
        jax_val = _f(jax_non_spine(b_j, d_j, _AGE_750, _T))
        torch_val = _f(torch_non_spine(b_t, d_t, _AGE_750_T, _T_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    # --- calculate_populations_log_lik ---

    def test_equiv_populations_ll_n5(self) -> None:
        jax_val = _f(jax_pop(_B, _D, _NS_5, _T))
        torch_val = _f(torch_pop(_B_T, _D_T, _NS_5_T, _T_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_populations_ll_multi_epoch(self) -> None:
        jax_val = _f(jax_pop(_B2, _D2, _NS_2, _T2))
        torch_val = _f(torch_pop(_B2_T, _D2_T, _NS_2_T, _T2_T))
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    # --- top-level log_likelihood ---

    def test_equiv_combined_ll_single_epoch(self) -> None:
        jax_val = _f(jax_log_likelihood(_B, _D, _AGES_10, _T))
        torch_val = _f(
            torch_log_likelihood(birth_rates=_B_T, death_rates=_D_T, ages=_AGES_10_NP, times=1500.0)
        )
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_combined_ll_multi_epoch(self) -> None:
        jax_val = _f(jax_log_likelihood(_B2, _D2, _AGES_10_MULTI, _T2))
        torch_val = _f(
            torch_log_likelihood(birth_rates=_B2_T, death_rates=_D2_T, ages=_AGES_10_MULTI_NP, times=_T2_T)
        )
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_combined_ll_is_ordered(self) -> None:
        """First age is treated as spine with is_ordered=True."""
        ages_np = np.array([1350.0, 100.0, 300.0, 600.0, 900.0])
        ages_j = jnp.array([1350.0, 100.0, 300.0, 600.0, 900.0], dtype=jnp.float64)
        jax_val = _f(jax_log_likelihood(_B, _D, ages_j, _T, is_ordered=True))
        torch_val = _f(
            torch_log_likelihood(birth_rates=_B_T, death_rates=_D_T, ages=ages_np, times=1500.0, is_ordered=True)
        )
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    def test_equiv_combined_ll_explicit_spine_ages(self) -> None:
        """Passing spine_ages explicitly produces the same result as is_ordered."""
        non_spine_np = np.array([100.0, 300.0, 600.0, 900.0])
        non_spine_j = jnp.array([100.0, 300.0, 600.0, 900.0], dtype=jnp.float64)
        spine_j = jnp.array([300.0], dtype=jnp.float64)
        spine_t = torch.tensor([300.0], dtype=torch.float64)
        jax_val = _f(jax_log_likelihood(_B, _D, non_spine_j, _T, spine_ages=spine_j))
        torch_val = _f(
            torch_log_likelihood(birth_rates=_B_T, death_rates=_D_T, ages=non_spine_np, times=1500.0, spine_ages=spine_t)
        )
        assert abs(jax_val - torch_val) < _ATOL_EQUIV

    # =========================================================================
    # Group 3 — Parameter grid sweeps (extinction + non-spine, 1e-8)
    # =========================================================================

    @pytest.mark.parametrize(
        "b_val, d_val, t_val",
        [
            (0.05, 0.03, 1500.0),    # canonical supercritical
            (0.03, 0.05, 1500.0),    # subcritical
            (0.05, 0.05, 1500.0),    # critical (λ = μ)
            (0.10, 0.02, 800.0),     # strongly supercritical
            (0.02, 0.10, 800.0),     # strongly subcritical
            (0.001, 0.001, 2000.0),  # very small rates
        ],
    )
    def test_equiv_extinction_param_grid(self, b_val: float, d_val: float, t_val: float) -> None:
        b_j = jnp.array([b_val], dtype=jnp.float64)
        d_j = jnp.array([d_val], dtype=jnp.float64)
        t_j = jnp.array([t_val], dtype=jnp.float64)
        b_t = torch.tensor([b_val], dtype=torch.float64)
        d_t = torch.tensor([d_val], dtype=torch.float64)
        t_t = torch.tensor([t_val], dtype=torch.float64)
        assert abs(_f(jax_ext_pr(b_j, d_j, t_j)) - _f(torch_ext_pr(b_t, d_t, t_t))) < _ATOL_EQUIV

    @pytest.mark.parametrize(
        "b_val, d_val, t_val",
        [
            (0.05, 0.03, 1500.0),
            (0.10, 0.02, 800.0),
            (0.02, 0.08, 1000.0),
        ],
    )
    def test_equiv_non_spine_param_grid(self, b_val: float, d_val: float, t_val: float) -> None:
        ages_np = np.array([0.2 * t_val, 0.5 * t_val, 0.8 * t_val])
        b_j = jnp.array([b_val], dtype=jnp.float64)
        d_j = jnp.array([d_val], dtype=jnp.float64)
        t_j = jnp.array([t_val], dtype=jnp.float64)
        ages_j = jnp.array(ages_np, dtype=jnp.float64)
        b_t = torch.tensor([b_val], dtype=torch.float64)
        d_t = torch.tensor([d_val], dtype=torch.float64)
        t_t = torch.tensor([t_val], dtype=torch.float64)
        ages_t = torch.tensor(ages_np, dtype=torch.float64)
        assert abs(_f(jax_non_spine(b_j, d_j, ages_j, t_j)) - _f(torch_non_spine(b_t, d_t, ages_t, t_t))) < _ATOL_EQUIV

    @pytest.mark.parametrize(
        "b_val, d_val, t_val",
        [
            (0.05, 0.03, 1500.0),
            (0.10, 0.02, 800.0),
            (0.02, 0.08, 1000.0),
        ],
    )
    def test_equiv_spine_param_grid(self, b_val: float, d_val: float, t_val: float) -> None:
        ages_np = np.array([0.2 * t_val, 0.5 * t_val, 0.8 * t_val])
        b_j = jnp.array([b_val], dtype=jnp.float64)
        d_j = jnp.array([d_val], dtype=jnp.float64)
        t_j = jnp.array([t_val], dtype=jnp.float64)
        ages_j = jnp.array(ages_np, dtype=jnp.float64)
        b_t = torch.tensor([b_val], dtype=torch.float64)
        d_t = torch.tensor([d_val], dtype=torch.float64)
        t_t = torch.tensor([t_val], dtype=torch.float64)
        ages_t = torch.tensor(ages_np, dtype=torch.float64)
        assert abs(_f(jax_spine(b_j, d_j, ages_j, t_j)) - _f(torch_spine(b_t, d_t, ages_t, t_t))) < _ATOL_EQUIV

    # =========================================================================
    # Group 4 — Gradient equivalence (jax.grad vs torch.autograd vs analytic)
    # =========================================================================

    def test_grad_extinction_vs_torch(self) -> None:
        """∂extinction_pr/∂(b, d): jax.grad on spike vs jax.grad on main module."""
        b_j = jnp.array([0.05], dtype=jnp.float64)
        d_j = jnp.array([0.03], dtype=jnp.float64)
        t_j = jnp.array([1500.0], dtype=jnp.float64)
        spike_gb, spike_gd = jax.grad(jax_ext_pr, argnums=(0, 1))(b_j, d_j, t_j)
        main_gb, main_gd = jax.grad(torch_ext_pr, argnums=(0, 1))(b_j, d_j, t_j)

        assert abs(_f(spike_gb[0]) - _f(main_gb[0])) < _ATOL_GRAD
        assert abs(_f(spike_gd[0]) - _f(main_gd[0])) < _ATOL_GRAD

    def test_grad_non_spine_vs_torch(self) -> None:
        """∂non_spine_ll/∂(b, d): jax.grad on spike vs jax.grad on main module."""
        b_j = jnp.array([0.05], dtype=jnp.float64)
        d_j = jnp.array([0.03], dtype=jnp.float64)
        t_j = jnp.array([1500.0], dtype=jnp.float64)
        ages_j = jnp.array([750.0], dtype=jnp.float64)
        spike_gb, spike_gd = jax.grad(jax_non_spine, argnums=(0, 1))(b_j, d_j, ages_j, t_j)
        main_gb, main_gd = jax.grad(torch_non_spine, argnums=(0, 1))(b_j, d_j, ages_j, t_j)

        assert abs(_f(spike_gb[0]) - _f(main_gb[0])) < _ATOL_GRAD
        assert abs(_f(spike_gd[0]) - _f(main_gd[0])) < _ATOL_GRAD

    def test_grad_spine_vs_torch(self) -> None:
        """∂spine_ll/∂(b, d): jax.grad on spike vs jax.grad on main module."""
        b_j = jnp.array([0.05], dtype=jnp.float64)
        d_j = jnp.array([0.03], dtype=jnp.float64)
        t_j = jnp.array([1500.0], dtype=jnp.float64)
        ages_j = jnp.array([750.0], dtype=jnp.float64)
        spike_gb, spike_gd = jax.grad(jax_spine, argnums=(0, 1))(b_j, d_j, ages_j, t_j)
        main_gb, main_gd = jax.grad(torch_spine, argnums=(0, 1))(b_j, d_j, ages_j, t_j)

        assert abs(_f(spike_gb[0]) - _f(main_gb[0])) < _ATOL_GRAD
        assert abs(_f(spike_gd[0]) - _f(main_gd[0])) < _ATOL_GRAD

    def test_grad_non_spine_multi_ages_vs_torch(self) -> None:
        """Gradient over multiple ages: jax.grad on spike vs jax.grad on main module."""
        b_j = jnp.array([0.05], dtype=jnp.float64)
        d_j = jnp.array([0.03], dtype=jnp.float64)
        t_j = jnp.array([1500.0], dtype=jnp.float64)
        ages_j = jnp.array([300.0, 600.0, 900.0, 1200.0], dtype=jnp.float64)
        spike_gb, spike_gd = jax.grad(jax_non_spine, argnums=(0, 1))(b_j, d_j, ages_j, t_j)
        main_gb, main_gd = jax.grad(torch_non_spine, argnums=(0, 1))(b_j, d_j, ages_j, t_j)

        assert abs(_f(spike_gb[0]) - _f(main_gb[0])) < _ATOL_GRAD
        assert abs(_f(spike_gd[0]) - _f(main_gd[0])) < _ATOL_GRAD
