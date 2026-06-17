"""JAX port of models/math_ages.py — Phase 2a spike.

Control-flow notes vs torch original:

- ``is_critical = torch.isclose(...).item()`` becomes a compile-time static
  dispatch only when called outside ``jit``; inside ``jit`` the critical and
  non-critical paths are merged with ``jnp.where``.  Since the equivalence gate
  calls these functions eagerly (not under ``jit``) this is fine.  The full
  in-place port in Phase 2b will consolidate into a single ``jnp.where`` path.
- In-place ``log_numerator[at_boundary] = ...`` → ``jnp.where``.
- Multiple ``return _ninf`` → single ``jnp.where``-guarded result at the end.
- ``torch.any(not_at_boundary)`` guards → dropped; the ``jnp.where`` handles
  the empty-mask case correctly (selects the other branch everywhere).

The single-epoch branch of both functions is a direct translation.  The
multi-epoch branch dispatches to the JAX extinction layer via ``_reverse_time``
(same transformation as the torch original).
"""

import jax
import jax.numpy as jnp

from inf_bp_texts.models._jax.math_extinction import (
    calculate_log_lik_initial_lifetimes_given_extinction,
    calculate_log_lik_initial_lifetimes_given_non_extinction,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reverse_time(
    birth_rates: jax.Array,
    death_rates: jax.Array,
    times: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Time-reversal transform for multi-epoch likelihood.

    Mirrors ``math_ages._reverse_time`` exactly.

    Parameters
    ----------
    birth_rates, death_rates : jax.Array
        Shape (n_epochs,) float64.
    times : jax.Array
        Epoch end times, shape (n_epochs,) float64.

    Returns
    -------
    tuple
        (inverted_times, reflected_birth_rates, reflected_death_rates)

    """
    end_time = times[-1]
    epoch_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), times[:-1]])
    inverted_times = jnp.flip(end_time - epoch_starts)
    reflected_birth_rates = jnp.flip(death_rates)
    reflected_death_rates = jnp.flip(birth_rates)
    return inverted_times, reflected_birth_rates, reflected_death_rates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_non_spine_ages_log_lik(
    birth_rates: jax.Array,
    death_rates: jax.Array,
    ages: jax.Array,
    times: jax.Array,
    *,
    return_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Log-likelihood of non-spine manuscript ages.

    For the single-epoch case uses the Kück & Schuhmacher (2018) eq. 3.6
    density; for multi-epoch delegates via the time-reversal trick.

    Parameters
    ----------
    birth_rates, death_rates : jax.Array
        Shape (n_epochs,) float64, all non-negative.
    ages : jax.Array
        Shape (n,) float64, 0 ≤ age ≤ times[-1].
    times : jax.Array
        Shape (n_epochs,) float64, strictly increasing.
    return_array : bool
        Per-age values if True, summed scalar if False.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    end_time = times[-1]
    n_epochs = len(times)

    if n_epochs > 1:
        inv_times, refl_br, refl_dr = _reverse_time(birth_rates, death_rates, times)
        return calculate_log_lik_initial_lifetimes_given_extinction(
            ages, refl_br, refl_dr, inv_times, as_array=return_array
        )

    # Single-epoch case
    birth_rate = birth_rates[0]
    death_rate = death_rates[0]
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    # Guard against invalid rates — functional path returns -inf array
    valid = (birth_rate > 0) & (death_rate > 0)

    is_critical = jnp.abs(birth_rate - death_rate) < 1e-12 * jnp.maximum(
        jnp.abs(birth_rate), jnp.ones_like(birth_rate)
    )

    # ---- Critical branch (λ ≈ μ) ----
    avg_rate = (birth_rate + death_rate) / 2.0
    log_num_crit = -avg_rate * ages + jnp.log1p(avg_rate * (end_time - ages))
    log_den_crit = jnp.log(end_time)
    log_density_crit = log_num_crit - log_den_crit

    # ---- Non-critical branch (λ ≠ μ) ----
    growth_rate = birth_rate - death_rate
    prod = growth_rate * end_time
    num1 = birth_rate * jnp.exp(-birth_rate * ages)
    num2 = death_rate * jnp.exp(-prod) * jnp.exp(-death_rate * ages)
    numerator = num1 - num2
    denominator = 1.0 - jnp.exp(-prod)

    # Signs must agree for a valid log; if not, -inf
    sign_ok = jnp.sign(numerator) == jnp.sign(denominator)
    safe_num = jnp.where(sign_ok, jnp.abs(numerator), jnp.ones_like(numerator))
    safe_den = jnp.where(jnp.abs(denominator) > 0, jnp.abs(denominator), jnp.ones_like(denominator))
    log_num_nc = jnp.log(safe_num)
    log_den_nc = jnp.log(safe_den)
    log_density_nc = jnp.where(sign_ok, log_num_nc - log_den_nc, _ninf)

    log_density = jnp.where(is_critical, log_density_crit, log_density_nc)

    # Final: -inf for invalid rates
    log_density = jnp.where(valid, log_density, _ninf)

    # -inf if any non-finite result
    all_finite = jnp.all(jnp.isfinite(log_density))
    log_density = jnp.where(all_finite, log_density, jnp.full_like(log_density, -jnp.inf))

    return log_density if return_array else log_density.sum()


def calculate_spine_ages_log_lik(
    birth_rates: jax.Array,
    death_rates: jax.Array,
    ages: jax.Array,
    times: jax.Array,
    *,
    return_array: bool = False,
    _skip_normalization: bool = False,
) -> jax.Array:
    r"""Log-likelihood of spine manuscript ages (with boundary mass at T).

    For the single-epoch case implements Kück & Schuhmacher (2018) eq. 3.8;
    multi-epoch delegates via the time-reversal trick.

    Parameters
    ----------
    birth_rates, death_rates : jax.Array
        Shape (n_epochs,) float64.
    ages : jax.Array
        Shape (n,) float64, 0 ≤ age ≤ times[-1].
    times : jax.Array
        Shape (n_epochs,) float64, strictly increasing.
    return_array : bool
        Per-age values if True, summed scalar if False.

    Returns
    -------
    jax.Array
        Scalar or (n,) float64.

    """
    end_time = times[-1]
    n_epochs = len(times)

    if n_epochs > 1:
        inv_times, refl_br, refl_dr = _reverse_time(birth_rates, death_rates, times)
        return calculate_log_lik_initial_lifetimes_given_non_extinction(
            ages, refl_br, refl_dr, inv_times, as_array=return_array
        )

    # Single-epoch case
    birth_rate = birth_rates[0]
    death_rate = death_rates[0]
    _ninf = jnp.array(-jnp.inf, dtype=jnp.float64)

    # Guard against negative rates
    valid = (birth_rate >= 0.0) & (death_rate >= 0.0)

    at_boundary = ages >= end_time
    not_at_boundary = ~at_boundary

    is_critical = jnp.abs(birth_rate - death_rate) < 1e-12 * jnp.maximum(
        jnp.abs(birth_rate), jnp.ones_like(birth_rate)
    )

    # ---- Critical branch (λ ≈ μ): single formula covers both at/not-at boundary ----
    avg_rate = (birth_rate + death_rate) / 2.0
    boundary_val = jnp.log1p(avg_rate * end_time) - avg_rate * end_time
    interior_val = 2.0 * jnp.log(avg_rate) + jnp.log(ages) - avg_rate * ages
    log_density_crit = jnp.where(at_boundary, boundary_val, interior_val)

    # ---- Non-critical branch (λ ≠ μ) ----
    denom_t = birth_rate - death_rate
    sign_denom = jnp.sign(denom_t)
    log_abs_denom = jnp.log(jnp.abs(denom_t))

    # Boundary term
    num_bdy = birth_rate * jnp.exp(-death_rate * end_time) - death_rate * jnp.exp(
        -birth_rate * end_time
    )
    sign_ok_bdy = jnp.sign(num_bdy) == sign_denom
    log_num_bdy = jnp.where(
        sign_ok_bdy,
        jnp.log(jnp.abs(jnp.where(jnp.abs(num_bdy) > 0, num_bdy, jnp.ones_like(num_bdy)))),
        _ninf,
    )

    # Interior term
    num_int = jnp.exp(-death_rate * ages) - jnp.exp(-birth_rate * ages)
    sign_ok_int = jnp.sign(num_int) == sign_denom
    log_num_int = jnp.where(
        sign_ok_int,
        (
            jnp.log(birth_rate)
            + jnp.log(death_rate)
            + jnp.log(jnp.abs(jnp.where(jnp.abs(num_int) > 0, num_int, jnp.ones_like(num_int))))
        ),
        _ninf,
    )

    log_num_nc = jnp.where(at_boundary, log_num_bdy, log_num_int)
    log_density_nc = log_num_nc - log_abs_denom

    log_density = jnp.where(is_critical, log_density_crit, log_density_nc)
    log_density = jnp.where(valid, log_density, _ninf)

    all_finite = jnp.all(jnp.isfinite(log_density))
    log_density = jnp.where(all_finite, log_density, jnp.full_like(log_density, -jnp.inf))

    return log_density if return_array else log_density.sum()
