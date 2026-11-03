"""Cross-check the maths duplicated in bdsurv against bdpinf's own functions."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bdsurv import (
    compute_level_replicates,
    select_level_times_for_band,
    simulate_birth_death_lifespans,
    simulate_birth_death_process,
)
from bdsurv._extinction import pr_extinction
from bdsurv.lifespans import (
    _sample_piecewise_exponential_death_time,  # pyright: ignore[reportPrivateUsage]
)
from bdsurv.linear_rates import (
    _as_linear,  # pyright: ignore[reportPrivateUsage]
    _next_event_offset,  # pyright: ignore[reportPrivateUsage]
    _segments,  # pyright: ignore[reportPrivateUsage]
)
from bdsurv.splitting import (
    _log_survival_probability,  # pyright: ignore[reportPrivateUsage]
)

from bdpinf.models.math_extinction import calculate_extinction_pr
from bdpinf.models.math_nested import nested_terms
from bdpinf.models.rate_functions import PiecewiseConstantRate, PiecewiseLinearRate

N_STANDARD_ERRORS = 4.0


def _bdpinf_log_survival(birth_rate: float, death_rate: float, time: float) -> float:
    """Return ``log Pr(Z_t > 0)`` as ``-softplus(log I)`` from the nested terms."""
    return -float(
        jax.nn.softplus(nested_terms(birth_rate, death_rate, time).log_i_total)
    )


SURVIVAL_CASES = [
    (0.1, 1.0, 5.0),
    (0.9, 1.0, 50.0),
    (1.0, 2.0, 20.0),
    (1.0, 10.0, 20.0),  # deep subcritical: 1 - P0 is tiny
    (1.0, 10.0, 100.0),
    (0.5, 0.5, 4.0),  # critical case
    (0.5 - 1e-6, 0.5, 4.0),  # just below critical threshold
    (0.5 + 1e-6, 0.5, 4.0),  # just above critical threshold
    (2.0, 0.5, 3.0),  # supercritical
    (0.05, 1.0, 0.1),  # short horizon
]


class TestExtinctionCrossCheck:
    """Pin bdsurv's closed-form extinction probability against bdpinf's JAX one.

    bdsurv reimplements the single-epoch special case of
    bdpinf.models.math_extinction.calculate_extinction_pr as a pure-numpy
    closed form (see bdsurv._extinction.pr_extinction), so the simulation
    package can avoid a JAX dependency. This test is the guard against the
    two implementations drifting apart.
    """

    @pytest.mark.parametrize(
        ("birth_rate", "death_rate", "terminal_time"),
        [
            (0.1, 1.0, 5.0),
            (0.9, 1.0, 50.0),
            (1.0, 2.0, 20.0),
            (0.5, 0.5, 4.0),  # critical case
            (0.5 - 1e-6, 0.5, 4.0),  # just below critical threshold
            (0.5 + 1e-6, 0.5, 4.0),  # just above critical threshold
            (2.0, 0.5, 3.0),  # supercritical
            (0.05, 1.0, 0.1),  # short horizon
        ],
    )
    def test_matches_bdsurv_closed_form(
        self, birth_rate: float, death_rate: float, terminal_time: float
    ) -> None:
        expected = float(calculate_extinction_pr(birth_rate, death_rate, terminal_time))
        actual = pr_extinction(birth_rate, death_rate, terminal_time)
        assert actual == pytest.approx(expected, abs=1e-8, rel=1e-6)

    def test_matches_across_random_grid(self) -> None:
        rng = np.random.default_rng(2026)
        birth_rates = rng.uniform(0.01, 5.0, size=50)
        death_rates = rng.uniform(0.01, 5.0, size=50)
        terminal_times = rng.uniform(0.1, 100.0, size=50)

        for b, d, t in zip(birth_rates, death_rates, terminal_times, strict=True):
            expected = float(calculate_extinction_pr(float(b), float(d), float(t)))
            actual = pr_extinction(float(b), float(d), float(t))
            assert actual == pytest.approx(expected, abs=1e-6, rel=1e-5)


class TestLogSurvivalCrossCheck:
    """Test bdsurv's log survival probability against bdpinf's nested terms."""

    @pytest.mark.parametrize(("birth_rate", "death_rate", "time"), SURVIVAL_CASES)
    def test_matches_nested_terms(
        self, birth_rate: float, death_rate: float, time: float
    ) -> None:
        expected = _bdpinf_log_survival(birth_rate, death_rate, time)
        actual = _log_survival_probability(
            birth_rate=birth_rate, death_rate=death_rate, time=time
        )
        assert actual == pytest.approx(expected, abs=1e-9, rel=1e-8)

    def test_non_positive_time_is_certain_survival(self) -> None:
        assert (
            _log_survival_probability(birth_rate=1.0, death_rate=2.0, time=0.0) == 0.0
        )


class TestLevelReplicatesDefinition:
    """Test bdsurv's replicate counts against their definition, using bdpinf."""

    @pytest.mark.parametrize(
        ("birth_rate", "death_rate", "terminal_time"),
        [(1.0, 1.1, 20.0), (1.0, 2.0, 20.0), (1.0, 10.0, 20.0), (0.9, 1.0, 100.0)],
    )
    def test_smallest_count_keeping_one_expected_line(
        self, birth_rate: float, death_rate: float, terminal_time: float
    ) -> None:
        levels = select_level_times_for_band(
            birth_rate=birth_rate, death_rate=death_rate, terminal_time=terminal_time
        )
        replicates = compute_level_replicates(
            birth_rate=birth_rate, death_rate=death_rate, level_times=levels
        )
        log_cumulative = 0.0
        for count, time in zip(replicates, levels[1:], strict=True):
            log_pr = _bdpinf_log_survival(birth_rate, death_rate, float(time))
            assert count >= 2
            log_cumulative += math.log(int(count))
            assert log_cumulative + log_pr >= -1e-6
            if count > 2:
                one_fewer = log_cumulative - math.log(int(count)) + math.log(count - 1)
                assert one_fewer + log_pr < 1e-6


class TestLevelBandCrossCheck:
    """Test bdsurv's level grid against the survival ratios from bdpinf."""

    @pytest.mark.parametrize(
        ("birth_rate", "death_rate", "terminal_time"),
        [(1.0, 1.1, 20.0), (1.0, 2.0, 20.0), (1.0, 10.0, 20.0), (0.9, 1.0, 100.0)],
    )
    def test_no_ratio_below_the_band(
        self, birth_rate: float, death_rate: float, terminal_time: float
    ) -> None:
        levels = select_level_times_for_band(
            birth_rate=birth_rate, death_rate=death_rate, terminal_time=terminal_time
        )
        assert levels[0] == 0.0
        assert levels[-1] == pytest.approx(terminal_time)
        log_pr = [
            0.0 if t == 0.0 else _bdpinf_log_survival(birth_rate, death_rate, float(t))
            for t in levels
        ]
        log_ratios = np.diff(log_pr)
        assert np.all(log_ratios >= -2.0 - 1e-9)
        assert np.all(log_ratios < 0.0)


class TestPiecewiseExponentialDeathTime:
    """Test the lifespan sampler's death-time law against bdpinf's integral."""

    def test_empirical_cdf_matches_cumulative_hazard(self) -> None:
        hazards = jnp.array([0.3, 1.2, 0.5])
        ends = jnp.array([2.0, 4.0, 7.0])
        birth_time = jnp.float64(1.0)
        n = 4_000
        keys = jax.random.split(jax.random.key(7), n)
        draws = np.asarray(
            jax.vmap(
                lambda k: _sample_piecewise_exponential_death_time(
                    birth_time, hazards, ends, ends[-1], k
                )
            )(keys)
        )
        rate = PiecewiseConstantRate(hazards, ends)
        for t in (1.5, 2.5, 3.5, 5.0, 6.5, 7.0):
            expected = 1.0 - math.exp(-float(rate.integral(birth_time, jnp.float64(t))))
            observed = float(np.mean(draws <= t))
            se = math.sqrt(expected * (1.0 - expected) / n)
            assert abs(observed - expected) < N_STANDARD_ERRORS * se + 1e-12
        survive = math.exp(-float(rate.integral(birth_time, ends[-1])))
        se = math.sqrt(survive * (1.0 - survive) / n)
        assert abs(np.mean(np.isinf(draws)) - survive) < N_STANDARD_ERRORS * se


class TestLinearEventOffset:
    """Test bdsurv's linear-rate event solver against bdpinf's rate integrals."""

    def _rates(self) -> tuple[PiecewiseLinearRate, PiecewiseConstantRate]:
        birth = PiecewiseLinearRate(
            jnp.array([0.3, 0.4]), jnp.array([0.4, 0.2]), jnp.array([5.0, 10.0])
        )
        death = PiecewiseConstantRate(jnp.array([0.2, 0.3]), jnp.array([4.0, 10.0]))
        return birth, death

    def test_segments_are_union_of_breakpoints_with_matching_integrals(self) -> None:
        birth, death = self._rates()
        segments = _segments(_as_linear(birth), _as_linear(death))
        edges = sorted(
            {
                0.0,
                *np.asarray(birth.breakpoints),
                *np.asarray(death.breakpoints),
                float(birth.end_time),
            }
        )
        np.testing.assert_allclose([s[0] for s in segments], edges[:-1])
        np.testing.assert_allclose([s[1] for s in segments], edges[1:])
        for start, end, b0, b1, d0, d1 in segments:
            width = end - start
            for rate, v0, slope in ((birth, b0, b1), (death, d0, d1)):
                exact = float(rate.integral(jnp.float64(start), jnp.float64(end)))
                assert v0 * width + slope * width**2 / 2.0 == pytest.approx(
                    exact, rel=1e-10
                )

    @pytest.mark.parametrize("n_alive", [1, 3, 10])
    def test_offset_solves_the_cumulative_hazard(self, n_alive: int) -> None:
        birth, death = self._rates()
        segments = _segments(_as_linear(birth), _as_linear(death))
        for start, end, b0, b1, d0, d1 in segments:
            a = n_alive * (b0 + d0)
            b = n_alive * (b1 + d1)
            target = 0.05
            offset = _next_event_offset(a, b, target)
            assert 0.0 < offset < end - start
            hazard = n_alive * sum(
                float(r.integral(jnp.float64(start), jnp.float64(start + offset)))
                for r in (birth, death)
            )
            assert hazard == pytest.approx(target, rel=1e-10)


class TestSimulatedExtinctionFrequency:
    """Test simulated extinction frequencies against bdpinf's extinction probability."""

    BIRTH = np.array([0.5, 0.3])
    DEATH = np.array([0.4, 0.5])
    TIMES = np.array([3.0, 6.0])

    def _assert_extinct_fraction(self, extinct: list[bool]) -> None:
        expected = float(
            calculate_extinction_pr(
                jnp.asarray(self.BIRTH),
                jnp.asarray(self.DEATH),
                jnp.asarray(self.TIMES),
            )
        )
        n = len(extinct)
        se = math.sqrt(expected * (1.0 - expected) / n)
        assert abs(np.mean(extinct) - expected) < N_STANDARD_ERRORS * se

    def test_lifespans_simulator(self) -> None:
        extinct = [
            not simulate_birth_death_lifespans(
                self.BIRTH, self.DEATH, self.TIMES, random_seed=seed
            )
            .records["survived"]
            .any()
            for seed in range(2_000)
        ]
        self._assert_extinct_fraction(extinct)

    def test_gillespie_simulator(self) -> None:
        extinct = [
            not simulate_birth_death_process(
                self.BIRTH, self.DEATH, self.TIMES, random_seed=seed
            )
            .records["survived"]
            .any()
            for seed in range(1_000)
        ]
        self._assert_extinct_fraction(extinct)
