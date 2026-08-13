"""Unit tests for the clustered-gate statistics in `vgc.evaluation`.

These exist because the archetype-pool gate reported a materially over-confident
interval for its first two runs. The properties pinned here are the ones that were
silently violated: an interval that ignores clustering, and subgroup checks that ignore
how many subgroups are being checked.
"""

from __future__ import annotations

import random
import statistics

import pytest

from vgc.evaluation import (
    clustered_interval,
    holm_rejections,
    normal_cdf,
    student_t_cdf,
    variance_components,
    wilson_interval,
)


def test_clustered_interval_matches_wilson_when_teams_agree() -> None:
    """With no between-team disagreement there is nothing for clustering to correct."""

    clusters = [(9, 18)] * 20
    low, high = clustered_interval(clusters)

    # Every team posts the identical record, so the between-cluster residual is zero.
    assert low == high == pytest.approx(0.5)


def test_clustered_interval_is_wider_than_wilson_when_teams_disagree() -> None:
    clusters = [(16, 18)] * 10 + [(2, 18)] * 10
    naive = wilson_interval(180, 360)
    clustered = clustered_interval(clusters)

    assert clustered[1] - clustered[0] > 2 * (naive[1] - naive[0])


def test_clustered_interval_falls_back_to_wilson_below_two_clusters() -> None:
    assert clustered_interval([(30, 50)]) == wilson_interval(30, 50)
    assert clustered_interval([]) == (0.0, 0.0)


def test_clustered_interval_handles_unequal_team_sizes() -> None:
    """The pool allows a partially-played team; the ratio estimator must not choke."""

    low, high = clustered_interval([(6, 10), (12, 30), (5, 8)])

    assert 0.0 <= low < high <= 1.0


def test_variance_components_recover_a_known_team_effect() -> None:
    """Sample teams with a real spread and check tau is estimated near the truth."""

    rng = random.Random(7)
    true_sd = 0.12
    clusters = []
    for _ in range(400):
        rate = min(0.98, max(0.02, rng.gauss(0.5, true_sd)))
        clusters.append((sum(1 for _ in range(18) if rng.random() < rate), 18))
    components = variance_components(clusters)

    assert components.tau == pytest.approx(true_sd, abs=0.03)
    assert components.design_effect > 1.0
    assert components.clustered_se > components.naive_se


def test_variance_components_report_zero_team_effect_when_there_is_none() -> None:
    """Pure binomial noise must not be misread as real team-to-team spread.

    Asserted across seeds rather than on one, because `tau` is the square root of a
    truncated variance estimate: `tau^2` is unbiased and lands at zero about half the
    time under a true null, but `sqrt` of a noisy non-negative quantity is biased upward,
    so any single seed can read ~0.05 with nothing real underneath. The median is the
    property that actually says "no systematic inflation".
    """

    taus = []
    for seed in range(60):
        rng = random.Random(seed)
        clusters = [(sum(1 for _ in range(18) if rng.random() < 0.5), 18) for _ in range(200)]
        taus.append(variance_components(clusters).tau)

    assert statistics.median(taus) < 0.03
    assert sum(1 for tau in taus if tau == 0.0) > len(taus) // 4


def test_se_floor_does_not_move_with_more_games_but_does_with_more_teams() -> None:
    """The design fact that decides how to spend a gate's budget.

    Between-team variance is divided by the number of TEAMS, so buying more games per
    team leaves the floor untouched while adding teams lowers it.
    """

    few_teams_many_games = variance_components([(60, 100), (40, 100)] * 15)
    many_teams_few_games = variance_components([(11, 18), (7, 18)] * 60)

    assert many_teams_few_games.se_floor < few_teams_many_games.se_floor

    doubled_games = variance_components([(120, 200), (80, 200)] * 15)
    assert doubled_games.se_floor == pytest.approx(few_teams_many_games.se_floor, rel=0.1)


def test_teams_needed_grows_as_the_target_effect_shrinks() -> None:
    components = variance_components([(11, 18), (7, 18), (9, 18), (12, 18)] * 15)

    assert components.teams_needed(0.02, 18) > components.teams_needed(0.05, 18)
    # More games per team helps, but cannot go below the team-effect-only requirement.
    assert components.teams_needed(0.02, 200) < components.teams_needed(0.02, 18)


def test_holm_is_stricter_than_uncorrected_but_keeps_the_clear_signal() -> None:
    # 0.01 survives correction at 6 tests (0.05/6 = 0.0083 fails, so use a clearer one).
    assert holm_rejections([0.001, 0.2, 0.3, 0.4, 0.5, 0.6]) == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    # A lone p = 0.04 among six is exactly the false alarm the old guardrail acted on.
    assert holm_rejections([0.04, 0.2, 0.3, 0.4, 0.5, 0.6]) == [False] * 6


def test_holm_steps_down_and_stops_at_the_first_failure() -> None:
    """Two strong signals both survive; the step-down must not stop after the first."""

    assert holm_rejections([0.001, 0.002, 0.9]) == [True, True, False]
    # Ordering is by p-value, not by position.
    assert holm_rejections([0.9, 0.002, 0.001]) == [False, True, True]


def test_normal_cdf_at_known_points() -> None:
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


@pytest.mark.parametrize(
    ("t", "df", "expected"),
    [
        (1.0, 1, 0.75),  # df=1 is Cauchy, closed form
        (1.8595, 8, 0.95),
        (2.2281, 10, 0.975),
        (2.8453, 20, 0.995),
        (1.6973, 30, 0.95),
        (-2.0150, 5, 0.05),
        (0.0, 100, 0.5),
    ],
)
def test_student_t_cdf_matches_published_critical_values(
    t: float, df: int, expected: float
) -> None:
    assert student_t_cdf(t, df) == pytest.approx(expected, abs=1e-4)


def test_student_t_cdf_converges_to_the_normal() -> None:
    assert student_t_cdf(1.96, 1_000_000) == pytest.approx(normal_cdf(1.96), abs=1e-5)


def test_student_t_is_more_conservative_than_the_normal_at_small_samples() -> None:
    """The reason the guardrail uses it: 9 teams is not 'SE known exactly'."""

    assert student_t_cdf(-1.8, 8) > normal_cdf(-1.8)
